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

Compared to the P4_2 (MSA intermediate) approach:
  - No MSAModuleCapture subclass needed
  - No _swap_msa_module() needed
  - Gradient flows through get_pairformer_output() directly

TTT hook methods (base loop calls these; no ttt() override needed):
  _ttt_tokenize()       restype one-hot → integer index [1, N_token]
                        also preprocesses feat dict and stores it
  _ttt_sample_batch()   masks feat dict; stashes in _ttt_feat_cache;
                        returns dummy token tensors for base loop
  _ttt_predict_logits() runs get_pairformer_output() on masked feat;
                        applies ttt_lm_head; returns [1, N_token, N_RESTYPE]

All base TTTModule features are active: NaN/Inf guard, LR scheduler,
automatic best-state reset, eval_each_step, early stopping, df logging.

Usage
-----
    ttt_model = ProtenixTTT.from_protenix(protenix_model, ttt_cfg=cfg)
    ttt_model.eval()
    results = ttt_model.ttt(input_feature_dict=feat_dict)
    prediction = ttt_model(input_feature_dict=feat_dict, ...)
    ttt_model.ttt_reset()
"""

import sys
import typing as T
from pathlib import Path

import torch
import torch.nn as nn

from proteinttt.base import TTTModule, TTTConfig

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

# ---------------------------------------------------------------------------
# Constants / defaults
# ---------------------------------------------------------------------------

N_RESTYPE = 32          # Protenix restype vocabulary size
_MASK_SENTINEL = N_RESTYPE  # integer sentinel for masked positions in token tensor
_C_S = 384              # pairformer single representation dim (c_s)

DEFAULT_PROTENIX_TTT_CFG = TTTConfig(
    lr=1e-4,
    ags=1,
    steps=5,
    batch_size=1,
    mask_ratio=0.15,
    bert_leave_prob=0.0,
    bert_replace_prob=0.0,
    optimizer="adamw",
    weight_decay=0.0,
    lora_rank=0,
    eval_each_step=False,              # set True to get pLDDT at each step
    initial_state_reset=True,
    automatic_best_state_reset=False,  # set True together with eval_each_step
    score_seq_kind=None,
)

# ---------------------------------------------------------------------------
# ProtenixTTT
# ---------------------------------------------------------------------------

class ProtenixTTT(TTTModule, Protenix):
    """ProteinTTT wrapper for Protenix (F00 design).

    Attaches a linear LM head to the pairformer single representation s
    [N_token, c_s=384] and trains it via masked residue-type prediction.

    This mirrors how ESM2TTT attaches its LM head to the transformer's final
    hidden state. The trainable parameters are pairformer_stack + ttt_lm_head
    (configurable via _ttt_get_trainable_modules).
    """

    ttt_default_cfg = DEFAULT_PROTENIX_TTT_CFG

    def __init__(
        self,
        configs,
        ttt_cfg: T.Optional[TTTConfig] = None,
        c_s: int = _C_S,
    ):
        Protenix.__init__(self, configs)

        # LM head: pairformer s [N_token, c_s] → restype logits [N_token, N_RESTYPE]
        # TTTModule.__init__()이 _ttt_get_trainable_modules()를 호출하므로
        # ttt_lm_head를 반드시 먼저 생성해야 함
        actual_c_s = getattr(configs.model, "c_s", c_s)
        self.ttt_lm_head = nn.Linear(actual_c_s, N_RESTYPE)

        TTTModule.__init__(self, ttt_cfg=ttt_cfg)

        # Shared state between _ttt_sample_batch() and _ttt_predict_logits()
        self._ttt_feat_cache: T.Optional[T.Dict[str, T.Any]] = None
        self._ttt_prepared_feat: T.Optional[T.Dict[str, T.Any]] = None

    # ------------------------------------------------------------------
    # TTTModule hook methods
    # ------------------------------------------------------------------

    def _ttt_tokenize(
        self, seq: T.Optional[str] = None, **kwargs
    ) -> torch.Tensor:
        """Preprocess feat dict and return restype class indices [1, N_token].

        Stores the preprocessed feat dict in self._ttt_prepared_feat so that
        _ttt_sample_batch() can reuse it without repeating preprocessing.
        """
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
        feat = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in feat.items()}
        self._ttt_prepared_feat = feat  # reused by _ttt_sample_batch

        restype = feat["restype"]  # [N_token, N_RESTYPE]
        has_res = restype.sum(-1) > 0
        idx = torch.full(
            (restype.shape[0],), -1, dtype=torch.long, device=restype.device
        )
        idx[has_res] = restype[has_res].argmax(-1).long()
        return idx.unsqueeze(0)  # [1, N_token]

    def _ttt_mask_token(self, token: int) -> int:
        return _MASK_SENTINEL

    def _ttt_get_non_special_tokens(self) -> T.List[int]:
        return list(range(20))  # 20 standard amino acids

    def _ttt_get_trainable_modules(self) -> T.List[nn.Module]:
        # pairformer_stack is the "backbone trunk" — equivalent to ESM2's transformer layers
        return [self.pairformer_stack, self.ttt_lm_head]

    def _ttt_sample_batch(
        self, x: torch.Tensor
    ) -> T.Tuple[torch.Tensor, torch.Tensor, torch.Tensor, T.Optional[torch.Tensor]]:
        """Build a masked feat dict and return dummy token tensors for the base loop.

        Stashes the masked feat dict in self._ttt_feat_cache so that
        _ttt_predict_logits() can retrieve it.

        Returns (base loop format):
            batch_masked : [1, N_token]  token indices, masked positions = _MASK_SENTINEL
            targets      : [1, N_token]  original class indices (-1 for padding)
            mask         : [1, N_token]  bool, True at masked positions
            start_indices: None
        """
        feat = self._ttt_prepared_feat
        restype = feat["restype"]  # [N_token, N_RESTYPE]
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
            batch_masked.unsqueeze(0),  # [1, N_token]
            targets.unsqueeze(0),       # [1, N_token]
            mask_pos.unsqueeze(0),      # [1, N_token]
            None,
        )

    def _ttt_predict_logits(
        self,
        batch: torch.Tensor,
        start_indices: T.Optional[torch.Tensor] = None,
        **kwargs,
    ) -> torch.Tensor:
        """Run pairformer on the masked feat dict and return restype logits.

        Ignores `batch` (dummy token tensor from base loop).
        Uses self._ttt_feat_cache set by _ttt_sample_batch().

        Returns:
            logits: [1, N_token, N_RESTYPE]
        """
        if self._ttt_feat_cache is None:
            raise RuntimeError(
                "_ttt_feat_cache is None — "
                "_ttt_sample_batch() must be called before _ttt_predict_logits()."
            )

        # s: [N_token, c_s=384] — pairformer final single representation
        _, s, _ = self.get_pairformer_output(
            input_feature_dict=self._ttt_feat_cache,
            N_cycle=1,          # single cycle during TTT for speed
            inplace_safe=False,
        )

        logits = self.ttt_lm_head(s)  # [N_token, N_RESTYPE]
        return logits.unsqueeze(0)    # [1, N_token, N_RESTYPE]

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
        """Run Protenix structure prediction and return pLDDT as confidence.

        Called by base loop when eval_each_step=True.
        Confidence drives automatic_best_state_reset.
        """
        feat = kwargs.get("input_feature_dict", self._ttt_prepared_feat)
        if feat is None:
            return {}, {}, None

        with torch.no_grad():
            pred_dict, _, _ = self(
                input_feature_dict=feat,
                label_full_dict=None,
                label_dict=None,
                mode="inference",
            )

        plddt_tensor = pred_dict.get("plddt")
        plddt = float(plddt_tensor.mean()) if plddt_tensor is not None else None

        return (
            {"pred_dict": pred_dict},
            {"plddt": plddt},
            plddt,  # confidence for automatic_best_state_reset
        )

    # ------------------------------------------------------------------
    # Protenix-specific helpers
    # ------------------------------------------------------------------

    def _ttt_mask_feat(self, feat: dict, mask_pos: torch.Tensor) -> dict:
        """Shallow-copy feat and zero all features that leak residue identity.

        Three categories of leakage are blocked:

        1. Token-level sequence features (InputEmbedder Line 2):
               restype, profile, deletion_mean

        2. Atom-level reference features (InputEmbedder -> AtomAttentionEncoder):
               ref_pos [N_atom, 3], ref_element [N_atom, 128],
               ref_atom_name_chars [N_atom, 4, 64], ref_charge [N_atom],
               ref_mask [N_atom], ref_space_uid [N_atom]
           These ideal-geometry tensors uniquely identify residue type.
           mask_pos is token-level; mapped to atom-level via atom_to_token_idx.
           d_lm / v_lm are derived from ref_pos / ref_space_uid, so they are
           recomputed after zeroing via update_input_feature_dict().

        3. MSA features [N_msa, N_token]:
               msa, has_deletion, deletion_value

        4. Template features:
               template_aatype [N_templ, N_token]

        Args:
            feat:     Preprocessed input_feature_dict (relp already computed).
            mask_pos: Bool tensor [N_token], True at positions to mask.

        Returns:
            New feat dict with all identity-leaking features zeroed at mask_pos.
        """
        masked = {
            k: v.clone() if isinstance(v, torch.Tensor) else v
            for k, v in feat.items()
        }

        # ------------------------------------------------------------------
        # 1. Token-level sequence features
        # ------------------------------------------------------------------
        masked["restype"][mask_pos] = 0.0
        if "profile" in masked:
            masked["profile"][mask_pos] = 0.0
        if "deletion_mean" in masked:
            masked["deletion_mean"][mask_pos] = 0.0

        # ------------------------------------------------------------------
        # 2. Atom-level reference features
        #    atom_to_token_idx [N_atom]: maps each atom to its token index
        # ------------------------------------------------------------------
        atom_to_token = masked["atom_to_token_idx"]              # [N_atom]
        masked_token_idx = mask_pos.nonzero(as_tuple=True)[0]    # [N_masked]
        atom_mask = torch.isin(atom_to_token, masked_token_idx)  # [N_atom] bool

        masked["ref_pos"][atom_mask] = 0.0            # [N_atom, 3]
        masked["ref_element"][atom_mask] = 0.0        # [N_atom, 128]
        masked["ref_atom_name_chars"][atom_mask] = 0.0  # [N_atom, 4, 64]
        masked["ref_charge"][atom_mask] = 0.0         # [N_atom]
        if "ref_mask" in masked:
            masked["ref_mask"][atom_mask] = 0.0       # [N_atom]

        # ref_space_uid drives v_lm (same-residue indicator in AtomAttentionEncoder).
        # Assign a unique sentinel so masked atoms don't match any real atom.
        if "ref_space_uid" in masked:
            sentinel = int(masked["ref_space_uid"].max().item()) + 1
            masked["ref_space_uid"][atom_mask] = sentinel

        # Recompute d_lm / v_lm / pad_info from the zeroed ref_pos / ref_space_uid
        masked = update_input_feature_dict(masked)

        # ------------------------------------------------------------------
        # 3. MSA features  [N_msa, N_token]
        # ------------------------------------------------------------------
        if "msa" in masked:
            masked["msa"][:, mask_pos] = 0
        if "has_deletion" in masked:
            masked["has_deletion"][:, mask_pos] = 0.0
        if "deletion_value" in masked:
            masked["deletion_value"][:, mask_pos] = 0.0

        # ------------------------------------------------------------------
        # 4. Template features  [N_templ, N_token]
        # ------------------------------------------------------------------
        if "template_aatype" in masked:
            masked["template_aatype"][:, mask_pos] = 0

        # update_input_feature_dict()가 CPU에 생성하는 텐서를 GPU로 올림
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
        """Wrap an existing Protenix instance with TTT capabilities.

        Args:
            model:   Pretrained Protenix model.
            ttt_cfg: TTT configuration (defaults to DEFAULT_PROTENIX_TTT_CFG).
            c_s:     Pairformer single-rep dim. Read from model.configs if present.

        Returns:
            ProtenixTTT with weights copied from model.
        """
        cfg = ttt_cfg or cls.ttt_default_cfg or DEFAULT_PROTENIX_TTT_CFG
        instance = cls(model.configs, ttt_cfg=cfg, c_s=c_s)
        model_param = next(model.parameters(), None)
        if model_param is not None:
            instance = instance.to(device=model_param.device, dtype=model_param.dtype)
        instance.load_state_dict(model.state_dict(), strict=False)
        if instance.ttt_cfg.initial_state_reset:
            instance._ttt_initial_state = instance._ttt_get_state()
        return instance
