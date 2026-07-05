"""OrderHead 등변성 + 디코딩 규칙 + 손실 방향성 검증."""

import torch

from src.track_a.decode import (
    IDENTITY_IDX,
    calibrate_tau,
    decode,
    perm_nll,
    perm_scores,
)
from src.track_a.model import OrderHead
from src.utils.permutation import ALL_PERMUTATIONS, IDENTITY


def _random_inputs(b=3, n_events=6, d=768, seed=0):
    g = torch.Generator().manual_seed(seed)
    img = torch.nn.functional.normalize(torch.randn(b, 4, d, generator=g), dim=-1)
    scalars = torch.randn(b, 4, 8, generator=g)
    events = torch.nn.functional.normalize(torch.randn(b, n_events, d, generator=g), dim=-1)
    mask = torch.zeros(b, n_events, dtype=torch.bool)
    mask[:, :3] = True
    caption = torch.nn.functional.normalize(torch.randn(b, d, generator=g), dim=-1)
    return img, scalars, events, mask, caption


def test_equivariance_and_gate_invariance():
    """프레임 입력을 섞으면 rank 로짓도 같은 방식으로 섞이고, 게이트는 불변."""
    torch.manual_seed(0)
    model = OrderHead().eval()
    img, scalars, events, mask, caption = _random_inputs()

    with torch.inference_mode():
        base_logits, base_gate = model(img, scalars, events, mask, caption)
        for perm in ([1, 0, 3, 2], [3, 2, 1, 0], [2, 0, 3, 1]):
            p = torch.tensor(perm)
            logits, gate = model(img[:, p], scalars[:, p], events, mask, caption)
            assert torch.allclose(logits, base_logits[:, p], atol=1e-5), perm
            assert torch.allclose(gate, base_gate, atol=1e-5), perm


def test_equivariance_without_events_or_scalars():
    torch.manual_seed(1)
    model = OrderHead(use_events=False, use_scalars=False).eval()
    img, scalars, events, mask, caption = _random_inputs(seed=2)
    p = torch.tensor([2, 3, 0, 1])
    with torch.inference_mode():
        base_logits, base_gate = model(img, scalars, events, mask, caption)
        logits, gate = model(img[:, p], scalars[:, p], events, mask, caption)
    assert torch.allclose(logits, base_logits[:, p], atol=1e-5)
    assert torch.allclose(gate, base_gate, atol=1e-5)


def _one_hot_logits(rank, scale=10.0):
    """rank(1-indexed)를 강하게 지지하는 로짓 (1,4,4)."""
    logits = torch.zeros(1, 4, 4)
    for i, r in enumerate(rank):
        logits[0, i, r - 1] = scale
    return logits


def test_perm_scores_pick_true_rank():
    rank = [3, 1, 4, 2]
    scores = perm_scores(_one_hot_logits(rank))
    assert ALL_PERMUTATIONS[int(scores.argmax())] == rank


def test_decode_identity_ban_and_gate():
    logits = _one_hot_logits(IDENTITY)  # 모델이 identity를 가장 선호하는 상황
    # 게이트 낮음 -> identity 금지 -> 비-identity 차선책
    pred = decode(logits, gate_probs=torch.tensor([0.1]), tau=0.5)[0]
    assert pred != IDENTITY and sorted(pred) == IDENTITY
    # 게이트 높음 -> identity 허용
    pred = decode(logits, gate_probs=torch.tensor([0.9]), tau=0.5)[0]
    assert pred == IDENTITY


def test_calibrate_tau_perfect_gate():
    """게이트가 no_ordering을 완벽 분리하면 EM 1.0인 τ가 존재."""
    ranks = [[2, 1, 3, 4], [3, 4, 1, 2], IDENTITY, IDENTITY]
    logits = torch.cat([_one_hot_logits(r) for r in ranks])
    gate = torch.tensor([0.05, 0.10, 0.95, 0.90])
    tau, em, curve = calibrate_tau(logits, gate, ranks)
    assert em == 1.0
    assert 0.10 < tau < 0.90
    assert curve[-1]["tau"] > 1.0  # "게이트 무시" 지점 포함


def test_perm_nll_prefers_correct_logits():
    rank = [4, 2, 1, 3]
    good = perm_nll(_one_hot_logits(rank), torch.tensor([rank]))
    bad = perm_nll(_one_hot_logits([1, 3, 2, 4]), torch.tensor([rank]))
    assert good < bad


def test_identity_index_matches():
    assert ALL_PERMUTATIONS[IDENTITY_IDX] == IDENTITY
