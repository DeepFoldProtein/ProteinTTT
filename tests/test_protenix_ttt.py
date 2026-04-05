
"""
Tests for ProtenixTTT (F00 design).

Test suite:
    test_protenix_ttt_improves_plddt   — TTT가 pLDDT를 개선하는지 (핵심 실험)
    test_protenix_ttt_reset            — ttt_reset()이 가중치를 완전히 복원하는지
    test_protenix_ttt_loss_decreases   — TTT 중 loss가 감소하는지

데이터:
    tests/data/sampled_monomers.csv  (pdb_id, chain_id, sequence, length, length_bin)
"""

import copy
import csv
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

from argparse import Namespace
torch.serialization.add_safe_globals([Namespace])

# ---------------------------------------------------------------------------
# Fixtures / helpers
# ---------------------------------------------------------------------------

DATA_CSV = Path(__file__).parent / "data" / "sampled_monomers.csv"

pytestmark = [
    pytest.mark.gpu,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available"),
]


def load_sequences(length_bin: str = None, n: int = None) -> list[dict]:
    """Load sequences from the benchmark CSV, optionally filtered by length_bin."""
    rows = []
    with open(DATA_CSV, newline="") as f:
        for row in csv.DictReader(f):
            if length_bin is None or row["length_bin"] == length_bin:
                rows.append(row)
    if n is not None:
        rows = rows[:n]
    return rows


def load_protenix_model(device: str):
    """Load a Protenix model and its configs. Skips if not available."""
    try:
        from argparse import Namespace
        import sys
        sys.path.insert(0, str(Path(__file__).parent.parent / "protenix"))

        from configs.configs_base import configs as configs_base
        from configs.configs_data import data_configs
        from configs.configs_inference import inference_configs
        from configs.configs_model_type import model_configs
        from protenix.config.config import parse_configs
        from protenix.model.protenix import Protenix
    except ImportError as e:
        pytest.skip(f"Protenix not available: {e}")

    # inference runner와 동일한 방식으로 configs 구성
    # data_configs는 "data" 키로 감싸야 함
    base_configs = {**configs_base, **{"data": data_configs}, **inference_configs}

    # 1차 파싱으로 model_name 확인
    configs = parse_configs(
        configs=base_configs,
        arg_str=[],
        fill_required_with_null=True,
    )
    model_name = configs.model_name

    # model_configs에서 해당 모델 스펙 병합
    from collections.abc import Mapping
    def deep_update(d, u):
        for k, v in u.items():
            if isinstance(v, Mapping) and k in d and isinstance(d[k], Mapping):
                deep_update(d[k], v)
            else:
                d[k] = v
        return d

    base_configs = {**configs_base, **{"data": data_configs}, **inference_configs}
    if model_name in model_configs:
        deep_update(base_configs, model_configs[model_name])

    # 2차 파싱
    configs = parse_configs(
        configs=base_configs,
        arg_str=[],
        fill_required_with_null=True,
    )

    ckpt_path = Path.home() / "checkpoints" / "protenix" / f"{configs.model_name}.pt"
    if not ckpt_path.exists():
        pytest.skip(f"Checkpoint not found: {ckpt_path}")

    model = Protenix(configs).to(device)
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state = ckpt["model"]
    if next(iter(state)).startswith("module."):
        state = {k[len("module."):]: v for k, v in state.items()}
    model.load_state_dict(state, strict=True)
    model.eval()
    return model, configs


def build_feat_dict(sequence: str, configs, device: str) -> dict:
    """Build a minimal input_feature_dict for a single protein chain."""
    try:
        from protenix.data.inference.infer_dataloader import get_inference_dataloader
        from protenix.utils.torch_utils import to_device
        import tempfile, json
    except ImportError as e:
        pytest.skip(f"Protenix data pipeline not available: {e}")

    input_data = [{"sequences": [{"proteinChain": {"sequence": sequence, "count": 1}}],
                   "name": "test"}]
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(input_data, f)
        tmp_path = f.name

    configs.input_json_path = tmp_path
    dataloader = get_inference_dataloader(configs=configs)
    batch = next(iter(dataloader))
    from protenix.utils.torch_utils import to_device
    feat = batch[0][0]["input_feature_dict"]
    return to_device(feat, device)


def run_inference(model, feat: dict) -> dict:
    with torch.no_grad():
        pred_dict, _, _ = model(
            input_feature_dict=feat,
            label_full_dict=None,
            label_dict=None,
            mode="inference",
        )
    return pred_dict


def mean_plddt(pred_dict: dict) -> float:
    import torch.nn.functional as F
    plddt = pred_dict.get("plddt")
    assert plddt is not None, "plddt not found in pred_dict"
    if plddt.dim() == 3:
        bins = torch.linspace(0, 1, plddt.shape[-1], device=plddt.device)
        plddt = (F.softmax(plddt, dim=-1) * bins).sum(-1)
    return float(plddt.mean())


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

@pytest.mark.slow
def test_protenix_ttt_improves_plddt():
    """
    TTT가 pLDDT를 개선하는지 확인한다.

    sampled_monomers.csv의 short 시퀀스 3개에 대해:
      - baseline pLDDT 측정
      - TTT 적용
      - adapted pLDDT 측정
      - 평균 delta > 0 임을 확인
    """
    device = "cuda"
    model, configs = load_protenix_model(device)

    from proteinttt.models.protenix import ProtenixTTT, DEFAULT_PROTENIX_TTT_CFG

    ttt_cfg = copy.deepcopy(DEFAULT_PROTENIX_TTT_CFG)
    ttt_cfg.seed = 0
    ttt_cfg.steps = 5
    ttt_cfg.lr = 1e-4

    ttt_model = ProtenixTTT.from_protenix(model, ttt_cfg=ttt_cfg)
    ttt_model.eval()

    rows = load_sequences(length_bin="short", n=3)
    assert len(rows) > 0, f"No short sequences found in {DATA_CSV}"

    deltas = []
    for row in rows:
        feat = build_feat_dict(row["sequence"], configs, device)

        plddt_before = mean_plddt(run_inference(ttt_model, feat))
        ttt_model.ttt(input_feature_dict=feat)
        plddt_after = mean_plddt(run_inference(ttt_model, feat))
        ttt_model.ttt_reset()

        delta = plddt_after - plddt_before
        deltas.append(delta)
        print(f"  {row['pdb_id']}  before={plddt_before:.4f}  after={plddt_after:.4f}  delta={delta:+.4f}")

    mean_delta = sum(deltas) / len(deltas)
    assert mean_delta > 0, (
        f"TTT did not improve pLDDT on average. "
        f"mean delta={mean_delta:.4f}, per-sample deltas={[f'{d:.4f}' for d in deltas]}"
    )


def test_protenix_ttt_reset():
    """
    ttt_reset()이 pairformer_stack과 ttt_lm_head 가중치를 완전히 복원하는지 확인한다.

    절차:
      1. 시퀀스 A로 logits 측정 (before)
      2. 시퀀스 B로 TTT 적용 (모델이 달라져야 함)
      3. ttt_reset() 호출
      4. 시퀀스 A로 logits 재측정 (after reset)
      5. before == after reset 확인
    """
    device = "cuda"
    model, configs = load_protenix_model(device)

    from proteinttt.models.protenix import ProtenixTTT, DEFAULT_PROTENIX_TTT_CFG

    ttt_cfg = copy.deepcopy(DEFAULT_PROTENIX_TTT_CFG)
    ttt_cfg.seed = 0
    ttt_cfg.steps = 3
    ttt_cfg.initial_state_reset = True

    ttt_model = ProtenixTTT.from_protenix(model, ttt_cfg=ttt_cfg)
    ttt_model.eval()

    assert ttt_model._ttt_initial_state is not None, "_ttt_initial_state가 저장되지 않았습니다."
    assert len(ttt_model._ttt_initial_state) > 0, "_ttt_initial_state가 비어있습니다."

    rows = load_sequences(n=2)
    assert len(rows) >= 2, f"테스트에 시퀀스가 2개 이상 필요합니다."

    feat_a = build_feat_dict(rows[0]["sequence"], configs, device)
    feat_b = build_feat_dict(rows[1]["sequence"], configs, device)

    # Before TTT: tokenize해서 _ttt_prepared_feat 세팅 후 logits 추출
    with torch.no_grad():
        ttt_model._ttt_tokenize(input_feature_dict=feat_a)
        ttt_model._ttt_feat_cache = ttt_model._ttt_mask_feat(
            ttt_model._ttt_prepared_feat,
            torch.zeros(
                ttt_model._ttt_prepared_feat["restype"].shape[0],
                dtype=torch.bool, device=device
            )
        )
        logits_before = ttt_model._ttt_predict_logits(batch=None)

    # TTT on B (모델 가중치가 달라져야 함)
    ttt_model.ttt(input_feature_dict=feat_b)

    # After TTT: A에 대한 logits가 달라졌는지 확인 (sanity check)
    with torch.no_grad():
        ttt_model._ttt_tokenize(input_feature_dict=feat_a)
        ttt_model._ttt_feat_cache = ttt_model._ttt_mask_feat(
            ttt_model._ttt_prepared_feat,
            torch.zeros(
                ttt_model._ttt_prepared_feat["restype"].shape[0],
                dtype=torch.bool, device=device
            )
        )
        logits_after_ttt = ttt_model._ttt_predict_logits(batch=None)

    diff_after_ttt = torch.abs(logits_before - logits_after_ttt).max().item()
    assert diff_after_ttt > 1e-6, (
        f"TTT 후 logits가 변하지 않았습니다 (max diff={diff_after_ttt:.2e}). "
        "학습이 실제로 일어나지 않은 것 같습니다."
    )

    # Reset
    ttt_model.ttt_reset()

    # After reset: 원래 logits로 돌아왔는지 확인
    with torch.no_grad():
        ttt_model._ttt_tokenize(input_feature_dict=feat_a)
        ttt_model._ttt_feat_cache = ttt_model._ttt_mask_feat(
            ttt_model._ttt_prepared_feat,
            torch.zeros(
                ttt_model._ttt_prepared_feat["restype"].shape[0],
                dtype=torch.bool, device=device
            )
        )
        logits_after_reset = ttt_model._ttt_predict_logits(batch=None)

    max_diff = torch.abs(logits_before - logits_after_reset).max().item()
    mean_diff = torch.abs(logits_before - logits_after_reset).mean().item()

    tolerance = 1e-5
    assert max_diff < tolerance, (
        "ttt_reset()이 가중치를 완전히 복원하지 못했습니다.\n"
        f"  max diff  : {max_diff:.6e}  (tolerance: {tolerance:.6e})\n"
        f"  mean diff : {mean_diff:.6e}\n"
        f"  TTT 후 max diff (sanity): {diff_after_ttt:.6e}"
    )


def test_protenix_ttt_loss_decreases():
    """
    TTT 중 loss가 전반적으로 감소하는지 확인한다.

    첫 스텝 loss의 평균 > 마지막 스텝 loss의 평균 이어야 한다.
    """
    device = "cuda"
    model, configs = load_protenix_model(device)

    from proteinttt.models.protenix import ProtenixTTT, DEFAULT_PROTENIX_TTT_CFG

    ttt_cfg = copy.deepcopy(DEFAULT_PROTENIX_TTT_CFG)
    ttt_cfg.seed = 0
    ttt_cfg.steps = 10
    ttt_cfg.lr = 1e-4

    ttt_model = ProtenixTTT.from_protenix(model, ttt_cfg=ttt_cfg)
    ttt_model.eval()

    rows = load_sequences(length_bin="short", n=1)
    assert len(rows) > 0
    feat = build_feat_dict(rows[0]["sequence"], configs, device)

    result = ttt_model.ttt(input_feature_dict=feat)

    df = result.get("df")
    assert df is not None and not df.empty and "loss" in df.columns, (
        "ttt()가 loss 정보를 담은 df를 반환하지 않았습니다."
    )

    losses = df["loss"].dropna().tolist()
    assert len(losses) >= 2, "스텝이 너무 적어 loss 추이를 확인할 수 없습니다."

    first_half_avg = sum(losses[:len(losses)//2]) / (len(losses)//2)
    second_half_avg = sum(losses[len(losses)//2:]) / (len(losses) - len(losses)//2)

    assert second_half_avg < first_half_avg, (
        f"Loss가 감소하지 않았습니다.\n"
        f"  전반부 평균 loss: {first_half_avg:.4f}\n"
        f"  후반부 평균 loss: {second_half_avg:.4f}\n"
        f"  전체 loss: {[f'{l:.4f}' for l in losses]}"
    )
