"""24-순열 전수 스코어링 디코더 + 손실 + UNORDERABLE 게이트 τ 캘리브레이션.

EM은 순열 단위 지표이므로 순열 단위로 스코어링·argmax 한다:
  score(π) = Σ_i log_softmax(rank_logits)[i, π_i]
디코딩 규칙 (docs/findings.md: orderable 샘플에 identity 0건):
  P(no_ordering) > τ  →  [1,2,3,4]
  아니면              →  비-identity 23개 중 score 최대 순열
τ는 val EM을 직접 최대화하는 값으로 스윕 (스윕엔 "게이트 무시"에 해당하는 τ>1 포함).
"""

import numpy as np
import torch
import torch.nn.functional as F

from src.utils.permutation import ALL_PERMUTATIONS, IDENTITY

_PERM_TENSOR = torch.tensor(ALL_PERMUTATIONS, dtype=torch.long) - 1   # (24, 4), 0-indexed
IDENTITY_IDX = ALL_PERMUTATIONS.index(IDENTITY)
_PERM_INDEX = {tuple(p): i for i, p in enumerate(ALL_PERMUTATIONS)}


def perm_scores(rank_logits):
    """(B,4,4) rank 로짓 -> (B,24) 순열 로그확률 스코어."""
    lp = F.log_softmax(rank_logits, dim=-1)                           # (B,4,4)
    idx = _PERM_TENSOR.to(lp.device)                                  # (24,4)
    # scores[b,p] = Σ_i lp[b, i, idx[p,i]]
    gathered = lp.unsqueeze(1).expand(-1, idx.size(0), -1, -1).gather(
        3, idx.unsqueeze(0).unsqueeze(-1).expand(lp.size(0), -1, -1, 1)
    )
    return gathered.squeeze(-1).sum(-1)


def perm_index(rank):
    return _PERM_INDEX[tuple(rank)]


def perm_nll(rank_logits, target_ranks, label_smoothing=0.05):
    """순열 단위 CE. target_ranks: (B,4) 1-indexed rank 텐서."""
    scores = perm_scores(rank_logits)
    target = torch.tensor(
        [perm_index([int(v) for v in t]) for t in target_ranks], device=scores.device
    )
    return F.cross_entropy(scores, target, label_smoothing=label_smoothing)


def decode(rank_logits, gate_probs, tau, ban_identity=True):
    """배치 디코딩 -> rank 리스트(1-indexed)의 리스트."""
    scores = perm_scores(rank_logits)
    if ban_identity:
        scores[:, IDENTITY_IDX] = float("-inf")
    best = scores.argmax(dim=1)
    out = []
    for b in range(scores.size(0)):
        if float(gate_probs[b]) > tau:
            out.append(list(IDENTITY))
        else:
            out.append(list(ALL_PERMUTATIONS[int(best[b])]))
    return out


def calibrate_tau(rank_logits, gate_probs, true_ranks, taus=None, ban_identity=True):
    """val 전체에 대해 EM(τ)을 스윕해 최적 τ와 곡선을 반환.

    true_ranks: (B,4) 1-indexed (no_ordering 샘플은 identity가 정답이므로 그대로 비교).
    ban_identity=False가 실측 우위 (decode-ablation, findings.md §1 시사점 2 반증 참조).
    """
    taus = taus if taus is not None else [round(t, 2) for t in np.arange(0.0, 1.02, 0.02)] + [1.1]
    scores = perm_scores(rank_logits)
    if ban_identity:
        scores[:, IDENTITY_IDX] = float("-inf")
    best = scores.argmax(dim=1).cpu().numpy()
    gate = np.asarray(gate_probs, dtype=np.float64)
    truth = np.asarray([[int(v) for v in t] for t in true_ranks])
    is_identity_truth = (truth == np.array(IDENTITY)).all(axis=1)
    perm_correct = np.array(
        [list(ALL_PERMUTATIONS[int(b)]) == list(t) for b, t in zip(best, truth)]
    )

    curve = []
    for tau in taus:
        pred_identity = gate > tau
        em = np.where(pred_identity, is_identity_truth, perm_correct).mean()
        curve.append({"tau": float(tau), "em": float(em)})
    best_pt = max(curve, key=lambda c: c["em"])
    return best_pt["tau"], best_pt["em"], curve
