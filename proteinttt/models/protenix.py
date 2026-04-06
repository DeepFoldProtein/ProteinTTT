"""ProtenixTTT: ProteinTTT wrapper for Protenix (F00 design).

Protenix is an AlphaFold3-style structure predictor. This TTT integration
follows the same philosophy as ESM2/ESMFold: attach a linear LM head to the
model's final token representation and do masked residue-type prediction.

Architecture
------------
  ESM2:     transformer hidden state s  →  LM head  →  vocab logits
  Protenix: pairformer output       s  →  LM head  →  restype logits  (this file)

The pairformer output `s` has shape [N_token, c_s=384] and is the natural
equivalent of ESM2's last-layer hidden state. The existing confidence head
and diffusion module already read from `s`, so this is the correct place.

TTT hook methods (base loop calls these):
  _ttt_tokenize()       restype one-hot → integer index [1, N_token]
  _ttt_sample_batch()   masks feat dict; stashes in _ttt_feat_cache
  _ttt_predict_logits() runs get_pairformer_output() with fixed MSA seed

ttt() is overridden to add:
  - grad_norm tracking (per optimizer step)
  - wandb logging (ttt/loss, ttt/grad_norm, ttt/pseudo_perplexity, ttt/lr)
  - MSA seed fixed for reproducibility

Usage
-----
    ttt_model = ProtenixTTT.from_protenix(protenix_model, ttt_cfg=cfg)
    ttt_model.eval()
    results = ttt_model.ttt(input_feature_dict=feat_dict)
    prediction = ttt_model(input_feature_dict=feat_dict, ...)
    ttt_model.ttt_reset()
"""

import sys
import time
import typing as T
import warnings
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

from proteinttt.base import TTTModule, TTTConfig
from proteinttt.utils.torch import preserve_model_state

# ---------------------------------------------------------------------------
# Protenix path setup
# ---------------------------------------------------------------------------

_MTTT_ROOT = Path(__file__).parent.parent.parent  # .../MTTT
_PROTENIX_REPO = _MTTT_ROOT / "protenix"          # .../MTTT/protenix/

if str(_PROTENIX_REPO) not in sys.path:
    sys.path.insert(0, str(_PROTENIX_REPO))

try:
    from protenix.model.protenix import Protenix, update_input_feature_dict  # type: ignore
except ImportError as exc:
    raise ImportError(
        f"Cannot import Protenix: {exc}\n"
        f"Expected the Protenix repo at: {_PROTENIX_REPO}\n"
        "Run `pip install -e .` inside that folder, or verify the path."
    )

# wandb는 optional
try:
    import wandb
    _WANDB_AVAILABLE = True
except ImportError:
    _WANDB_AVAILABLE = False

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

N_RESTYPE = 32
_MASK_SENTINEL = N_RESTYPE
_C_S = 384

DEFAULT_PROTENIX_TTT_CFG = TTTConfig(
    lr=1e-3,
    ags=1,
    steps=5,
    batch_size=1,
    mask_ratio=0.15,
    bert_leave_prob=0.0,
    bert_replace_prob=0.0,
    optimizer="adamw",
    weight_decay=0.0,
    lora_rank=0,
    eval_each_step=False,
    initial_state_reset=True,
    automatic_best_state_reset=False,
    score_seq_kind=None,
)

# ---------------------------------------------------------------------------
# ProtenixTTT
# ---------------------------------------------------------------------------

class ProtenixTTT(TTTModule, Protenix):
    """ProteinTTT wrapper for Protenix (F00 design).

    Attaches a linear LM head to the pairformer single representation s
    [N_token, c_s=384] and trains it via masked residue-type prediction.

    Trainable modules: pairformer_stack + ttt_lm_head
    Frozen modules: distogram_head, confidence_head, diffusion_module

    wandb logging (if wandb.run is active):
        ttt/loss, ttt/grad_norm, ttt/pseudo_perplexity, ttt/lr
    """

    ttt_default_cfg = DEFAULT_PROTENIX_TTT_CFG

    def __init__(
        self,
        configs,
        ttt_cfg: T.Optional[TTTConfig] = None,
        c_s: int = _C_S,
    ):
        Protenix.__init__(self, configs)

        # LM head must be created before TTTModule.__init__()
        # because TTTModule.__init__() calls _ttt_get_trainable_modules()
        actual_c_s = getattr(configs.model, "c_s", c_s)
        self.ttt_lm_head = nn.Linear(actual_c_s, N_RESTYPE)

        TTTModule.__init__(self, ttt_cfg=ttt_cfg)

        self._ttt_feat_cache: T.Optional[T.Dict[str, T.Any]] = None
        self._ttt_prepared_feat: T.Optional[T.Dict[str, T.Any]] = None

        # Fixed seed for MSA subsampling — ensures reproducible pseudo-perplexity
        self._ttt_msa_seed: int = (
            ttt_cfg.seed if (ttt_cfg and ttt_cfg.seed is not None) else 0
        )

    # ------------------------------------------------------------------
    # TTTModule hook methods
    # ------------------------------------------------------------------

    def _ttt_tokenize(
        self, seq: T.Optional[str] = None, **kwargs
    ) -> torch.Tensor:
        feat = kwargs.get("input_feature_dict")
        if feat is None:
            raise ValueError(
                "ProtenixTTT._ttt_tokenize requires input_feature_dict as a keyword arg."
            )

        device = next(self.parameters()).device
        feat = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in feat.items()
        }
        feat = self.relative_position_encoding.generate_relp(feat)
        feat = update_input_feature_dict(feat)
        feat = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in feat.items()
        }
        self._ttt_prepared_feat = feat

        restype = feat["restype"]
        has_res = restype.sum(-1) > 0
        idx = torch.full(
            (restype.shape[0],), -1, dtype=torch.long, device=restype.device
        )
        idx[has_res] = restype[has_res].argmax(-1).long()
        return idx.unsqueeze(0)

    def _ttt_mask_token(self, token: int) -> int:
        return _MASK_SENTINEL

    def _ttt_get_non_special_tokens(self) -> T.List[int]:
        return list(range(20))

    def _ttt_get_trainable_modules(self) -> T.List[nn.Module]:
        # pairformer_stack: backbone trunk (representation learning)
        # ttt_lm_head: linear probe for masked residue-type prediction
        # distogram_head / confidence_head / diffusion_module: frozen (structure module)
        return [self.pairformer_stack, self.ttt_lm_head]

    def _ttt_sample_batch(
        self, x: torch.Tensor
    ) -> T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, T.Optional[torch.Tensor]]:
        feat = self._ttt_prepared_feat
        restype = feat["restype"]
        N_token = restype.shape[0]
        device = restype.device

        has_res = restype.sum(-1) > 0
        targets = torch.full((N_token,), -1, dtype=torch.long, device=device)
        targets[has_res] = restype[has_res].argmax(-1).long()

        candidate_idx = has_res.nonzero(as_tuple=True)[0]
        n_mask = max(1, int(len(candidate_idx) * self.ttt_cfg.mask_ratio))
        perm = torch.randperm(len(candidate_idx), generator=self.ttt_generator)
        chosen = candidate_idx[perm[:n_mask]]

        mask_pos = torch.zeros(N_token, dtype=torch.bool, device=device)
        mask_pos[chosen] = True

        self._ttt_feat_cache = self._ttt_mask_feat(feat, mask_pos)

        batch_masked = targets.clone()
        batch_masked[mask_pos] = _MASK_SENTINEL

        return (
            batch_masked.unsqueeze(0),
            targets.unsqueeze(0),
            mask_pos.unsqueeze(0),
            None,
        )

    def _ttt_predict_logits(
        self,
        batch: torch.Tensor,
        start_indices: T.Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run pairformer with fixed MSA seed and return restype logits.

        MSA seed is fixed to self._ttt_msa_seed so that pseudo-perplexity
        is comparable across TTT steps.

        Returns:
            logits: [1, N_token, N_RESTYPE]
        """
        if self._ttt_feat_cache is None:
            raise RuntimeError(
                "_ttt_feat_cache is None — "
                "_ttt_sample_batch() must be called before _ttt_predict_logits()."
            )

        # Fix MSA subsampling seed for reproducibility
        with torch.random.fork_rng(enabled=True):
            torch.manual_seed(self._ttt_msa_seed)
            _, s, _ = self.get_pairformer_output(
                input_feature_dict=self._ttt_feat_cache,
                N_cycle=1,
                inplace_safe=False,
            )

        logits = self.ttt_lm_head(s)
        return logits.unsqueeze(0)

    def _ttt_eval_step(
        self,
        step: int,
        loss: T.Optional[float],
        perplexity: T.Optional[float],
        all_log_probs: T.Optional[torch.Tensor],
        seq: T.Optional[str],
        msa_pth: T.Optional[Path],
        **kwargs,
    ) -> T.Tuple[dict, dict, T.Optional[float]]:
        """Run Protenix structure prediction and return pLDDT as confidence."""
        feat = kwargs.get("input_feature_dict", self._ttt_prepared_feat)
        if feat is None:
            return {}, {}, None

        with torch.no_grad():
            pred_dict, _, _ = self(
                input_feature_dict=self._ttt_clone_feat_dict(feat),
                label_full_dict=None,
                label_dict=None,
                mode="inference",
            )

        plddt_tensor = pred_dict.get("plddt")
        plddt = float(plddt_tensor.mean()) if plddt_tensor is not None else None

        return (
            {"pred_dict": pred_dict},
            {"plddt": plddt},
            plddt,
        )

    # ------------------------------------------------------------------
    # ttt() override — adds grad_norm tracking and wandb logging
    # ------------------------------------------------------------------

    def ttt(
        self,
        seq: T.Optional[str] = None,
        msa_pth: T.Optional[Path] = None,
        **kwargs,
    ) -> T.Dict[str, T.Any]:
        """TTT loop with grad_norm tracking and wandb logging.

        Extends base TTTModule.ttt() with:
          - grad_norm computed after backward, before optimizer step
          - wandb logging: ttt/loss, ttt/grad_norm, ttt/pseudo_perplexity, ttt/lr
          - MSA seed fixed in _ttt_predict_logits() for reproducibility
        """
        # Tokenize (preprocesses feat dict, stores _ttt_prepared_feat)
        x = self._ttt_tokenize(seq, **kwargs)

        # MSA (Protenix doesn't use external MSA files — already in feat dict)
        msa = None

        parameters = self._ttt_get_parameters()
        optimizer = self._ttt_get_optimizer(parameters)
        optimizer.zero_grad()

        df = []
        ttt_step_data = defaultdict(dict)
        score_seq_time = None
        eval_step_time = None

        if self.ttt_cfg.automatic_best_state_reset:
            best_confidence = 0
            best_state = None

        device = next(self.parameters()).device
        non_blocking = device.type == "cuda"
        cached_trainable_params = [p for p in self.parameters() if p.requires_grad]

        # LR scheduler
        scheduler = None
        if self.ttt_cfg.lr_scheduler == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, T_max=self.ttt_cfg.steps, eta_min=self.ttt_cfg.lr_min
            )
        elif self.ttt_cfg.lr_scheduler == "cosine_warmup":
            import math
            warmup = max(0, int(self.ttt_cfg.lr_warmup_steps))
            min_factor = (
                self.ttt_cfg.lr_min / self.ttt_cfg.lr if self.ttt_cfg.lr > 0 else 0.0
            )
            def lr_mult(step_idx):
                if warmup > 0 and step_idx < warmup:
                    return (step_idx + 1) / max(1, warmup)
                progress = (step_idx - warmup) / max(1, self.ttt_cfg.steps - warmup)
                return min_factor + 0.5 * (1.0 - min_factor) * (1.0 + math.cos(math.pi * progress))
            scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_mult)

        loss = None
        grad_norm = None
        self.eval()

        for step in range(self.ttt_cfg.steps * self.ttt_cfg.ags + 1):
            batch_masked, targets, mask, start_indices = self._ttt_sample_batch(x)

            if step % self.ttt_cfg.ags == 0:
                if step == 0:
                    last_step_time = time.time()
                ttt_step_time = time.time() - last_step_time

                # Score sequence (pseudo-perplexity)
                all_log_probs, perplexity = None, None
                should_score = self.ttt_cfg.score_seq_kind is not None and (
                    self.ttt_cfg.score_seq_steps_list is None
                    or (step // self.ttt_cfg.ags) in self.ttt_cfg.score_seq_steps_list
                )
                if should_score:
                    score_seq_start_time = time.time()
                    seq_to_score = x[0:1, :]
                    all_log_probs, perplexity = self._ttt_score_seq(seq_to_score, **kwargs)
                    score_seq_time = time.time() - score_seq_start_time
                    all_log_probs = [prob.detach().cpu() for prob in all_log_probs]
                    ttt_step_data[step // self.ttt_cfg.ags]["all_log_probs"] = all_log_probs
                else:
                    score_seq_time = 0.0

                # Eval step
                if self.ttt_cfg.eval_each_step:
                    eval_step_start_time = time.time()
                    eval_step_preds, eval_step_metric_dict, confidence = self._ttt_eval_step(
                        step=step // self.ttt_cfg.ags,
                        loss=loss.item() if loss is not None else None,
                        perplexity=perplexity,
                        all_log_probs=all_log_probs,
                        seq=seq,
                        msa_pth=msa_pth,
                        **kwargs,
                    )
                    eval_step_time = time.time() - eval_step_start_time
                    if self.ttt_cfg.automatic_best_state_reset and confidence is not None:
                        if confidence > best_confidence:
                            best_confidence = confidence
                            best_state = self._ttt_get_state()
                else:
                    eval_step_metric_dict = {}
                    eval_step_preds = None

                if eval_step_preds is not None:
                    ttt_step_data[step // self.ttt_cfg.ags]["eval_step_preds"] = eval_step_preds

                # Build log row
                row = dict(
                    step=step // self.ttt_cfg.ags,
                    accumulated_step=step,
                    loss=loss.item() if loss is not None else None,
                    grad_norm=grad_norm,
                    perplexity=perplexity,
                    ttt_step_time=ttt_step_time,
                    score_seq_time=score_seq_time,
                    eval_step_time=eval_step_time,
                    **eval_step_metric_dict,
                )
                if scheduler is not None:
                    row["lr"] = optimizer.param_groups[0]["lr"]
                df.append(row)

                # Console log
                log_row = ", ".join([
                    f"{k}: {v:.5f}" if isinstance(v, float) else f"{k}: {v}"
                    for k, v in row.items()
                ])
                self.ttt_logger.info(log_row)

                # wandb log
                self._wandb_log(row)

                last_step_time = time.time()

                # Early stopping
                if (
                    self.ttt_cfg.perplexity_early_stopping is not None
                    and perplexity is not None
                    and perplexity < self.ttt_cfg.perplexity_early_stopping
                ):
                    self.ttt_logger.info(
                        f"Early stopping at step {step} with perplexity {perplexity}"
                    )
                    break

            if step == self.ttt_cfg.steps * self.ttt_cfg.ags:
                break

            # Move to device
            batch_masked = batch_masked.to(device, non_blocking=non_blocking)
            targets = targets.to(device, non_blocking=non_blocking)
            mask = mask.to(device, non_blocking=non_blocking)

            # Forward pass
            self.train()
            logits = self._ttt_predict_logits(batch_masked, start_indices, **kwargs)

            # Loss
            loss = self._ttt_cross_entropy_loss(logits, targets, mask)

            # Backward
            loss.backward()

            if (step + 1) % self.ttt_cfg.ags == 0:
                trainable_params = [p for p in cached_trainable_params if p.grad is not None]

                # Compute grad_norm before clipping
                grad_norm = float(
                    torch.norm(
                        torch.stack([p.grad.norm() for p in trainable_params])
                    ).item()
                ) if trainable_params else 0.0

                # NaN/Inf check
                has_nan_grad = any(
                    torch.isnan(p.grad).any() or torch.isinf(p.grad).any()
                    for p in trainable_params
                )

                if has_nan_grad:
                    optimizer.zero_grad(set_to_none=True)
                    warnings.warn(f"NaN/Inf gradient detected at step {step}, skipping update")
                    grad_norm = float("nan")
                else:
                    if self.ttt_cfg.gradient_clip and trainable_params:
                        torch.nn.utils.clip_grad_norm_(
                            trainable_params,
                            max_norm=self.ttt_cfg.gradient_clip_max_norm,
                        )
                    optimizer.step()
                    if scheduler is not None:
                        scheduler.step()
                    optimizer.zero_grad(set_to_none=True)

            self.eval()

        if self.ttt_cfg.automatic_best_state_reset and best_state is not None:
            self._ttt_set_state(best_state)

        df = pd.DataFrame(df)
        return dict(ttt_step_data=ttt_step_data, df=df)

    # ------------------------------------------------------------------
    # wandb helper
    # ------------------------------------------------------------------

    def _wandb_log(self, row: dict) -> None:
        """Log a metrics row to wandb if available and initialized."""
        if not _WANDB_AVAILABLE:
            return
        try:
            if wandb.run is None:
                return
        except Exception:
            return

        log_dict = {}
        key_map = {
            "loss": "ttt/loss",
            "grad_norm": "ttt/grad_norm",
            "perplexity": "ttt/pseudo_perplexity",
            "lr": "ttt/lr",
            "plddt": "ttt/plddt",
        }
        for src_key, wandb_key in key_map.items():
            val = row.get(src_key)
            if val is not None:
                log_dict[wandb_key] = val

        if log_dict:
            wandb.log(log_dict, step=int(row.get("step", 0)))

    def _ttt_clone_feat_dict(self, feat: dict) -> dict:
        """Clone tensors so inference-time in-place cleanup does not mutate caller state."""
        return {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in feat.items()
        }

    # ------------------------------------------------------------------
    # Protenix-specific helpers
    # ------------------------------------------------------------------

    def _ttt_mask_feat(self, feat: dict, mask_pos: torch.Tensor) -> dict:
        """Shallow-copy feat and zero all features that leak residue identity."""
        masked = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in feat.items()
        }

        # 1. Token-level sequence features
        masked["restype"][mask_pos] = 0.0
        if "profile" in masked:
            masked["profile"][mask_pos] = 0.0
        if "deletion_mean" in masked:
            masked["deletion_mean"][mask_pos] = 0.0

        # 2. Atom-level reference features
        atom_to_token = masked["atom_to_token_idx"]
        masked_token_idx = mask_pos.nonzero(as_tuple=True)[0]
        atom_mask = torch.isin(atom_to_token, masked_token_idx)

        masked["ref_pos"][atom_mask] = 0.0
        masked["ref_element"][atom_mask] = 0.0
        masked["ref_atom_name_chars"][atom_mask] = 0.0
        masked["ref_charge"][atom_mask] = 0.0
        if "ref_mask" in masked:
            masked["ref_mask"][atom_mask] = 0.0
        if "ref_space_uid" in masked:
            sentinel = int(masked["ref_space_uid"].max().item()) + 1
            masked["ref_space_uid"][atom_mask] = sentinel

        masked = update_input_feature_dict(masked)

        # 3. MSA features
        if "msa" in masked:
            masked["msa"][:, mask_pos] = 0
        if "has_deletion" in masked:
            masked["has_deletion"][:, mask_pos] = 0.0
        if "deletion_value" in masked:
            masked["deletion_value"][:, mask_pos] = 0.0

        # 4. Template features
        if "template_aatype" in masked:
            masked["template_aatype"][:, mask_pos] = 0

        # Move new tensors (d_lm, v_lm) to GPU
        device = masked["restype"].device
        masked = {
            k: v.to(device) if isinstance(v, torch.Tensor) else v
            for k, v in masked.items()
        }
        return masked

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------

    @classmethod
    def from_protenix(
        cls,
        model: "Protenix",
        ttt_cfg: T.Optional[TTTConfig] = None,
        c_s: int = _C_S,
    ) -> "ProtenixTTT":
        """Wrap an existing Protenix instance with TTT capabilities."""
        cfg = ttt_cfg or cls.ttt_default_cfg or DEFAULT_PROTENIX_TTT_CFG
        instance = cls(model.configs, ttt_cfg=cfg, c_s=c_s)
        model_param = next(model.parameters(), None)
        if model_param is not None:
            instance = instance.to(device=model_param.device, dtype=model_param.dtype)
        instance.load_state_dict(model.state_dict(), strict=False)
        if instance.ttt_cfg.initial_state_reset:
            instance._ttt_initial_state = instance._ttt_get_state()
        return instance
