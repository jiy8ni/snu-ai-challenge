"""융합 결정: 합의도 게이트·LL 재순위·ban_identity·폴백 검증."""

import pytest

from src.infer.fuse import consensus_of, fuse_all
from src.infer.ll_score import UNORD_KEY
from src.train.targets import format_answer
from src.utils.permutation import ALL_PERMUTATIONS, IDENTITY


def _ll_view(best_key, best_lp=-1.0, other_lp=-50.0, unord_lp=-60.0, n_tok=10):
    ll = {format_answer(r): [other_lp, n_tok] for r in ALL_PERMUTATIONS}
    ll[best_key] = [best_lp, n_tok]
    ll[UNORD_KEY] = [unord_lp, n_tok]
    return ll


def _votes(*ranks):
    return [("orderable", list(r)) for r in ranks]


R_A = [2, 1, 4, 3]
R_B = [3, 1, 4, 2]
R_C = [4, 3, 2, 1]


def test_consensus_of():
    assert consensus_of(_votes(R_A, R_A, R_A, R_B)) == 3
    assert consensus_of([("fail", None), ("unorderable", None)]) == 0


def test_h1_high_consensus_keeps_vote_even_if_ll_disagrees():
    votes = {"a": _votes(R_A, R_A, R_A, R_B)}
    ll = {"a": [_ll_view(format_answer(R_C))]}  # LL은 R_C를 선호
    assert fuse_all(votes, ll, "h1", k=3)["a"] == R_A


def test_h1_low_consensus_uses_ll():
    votes = {"a": _votes(R_A, R_B, R_C, [1, 3, 2, 4])}  # 합의도 1 (전부 분산)
    ll = {"a": [_ll_view(format_answer(R_C))]}
    assert fuse_all(votes, ll, "h1", k=3)["a"] == R_C


def test_h1_low_consensus_replaces_disperse_gate_identity():
    """합의 전무 샘플: 투표 단독은 disperse_gate로 identity, 융합은 LL 판정."""
    votes = {"a": _votes(R_A, R_B, R_C, [1, 3, 2, 4])}
    ll = {"a": [_ll_view(format_answer(R_B))]}
    assert fuse_all(votes, ll, "vote")["a"] == list(IDENTITY)
    assert fuse_all(votes, ll, "h1", k=2)["a"] == R_B


def test_h2_restricts_to_voted_candidates():
    """LL 전체 1위(R_C)가 투표에 없으면, 투표된 후보 중 LL 최고(R_B)를 채택."""
    votes = {"a": _votes(R_A, R_B)}  # 합의도 1 < k
    view = _ll_view(format_answer(R_C), best_lp=-1.0)
    view[format_answer(R_B)] = [-2.0, 10]
    view[format_answer(R_A)] = [-3.0, 10]
    ll = {"a": [view]}
    assert fuse_all(votes, ll, "h1", k=3)["a"] == R_C
    assert fuse_all(votes, ll, "h2", k=3)["a"] == R_B


def test_h2_no_orderable_votes_falls_back_to_full_ll():
    votes = {"a": [("fail", None), ("unorderable", None)]}
    ll = {"a": [_ll_view(format_answer(R_C))]}
    assert fuse_all(votes, ll, "h2", k=3)["a"] == R_C


def test_margin_gate_in_ll_path_yields_identity():
    votes = {"a": _votes(R_A, R_B)}  # 저합의 -> LL 경로
    view = _ll_view(format_answer(R_A), best_lp=-10.0, unord_lp=-5.0)  # 평균 차 0.5
    ll = {"a": [view]}
    assert fuse_all(votes, ll, "h1", k=3, margin=0.4)["a"] == list(IDENTITY)
    assert fuse_all(votes, ll, "h1", k=3, margin=0.6)["a"] == R_A


def test_ban_identity_in_ll_path():
    votes = {"a": _votes(R_A, R_B)}
    view = _ll_view(format_answer(IDENTITY), best_lp=-1.0)
    view[format_answer(R_B)] = [-2.0, 10]
    ll = {"a": [view]}
    assert fuse_all(votes, ll, "h1", k=3)["a"] == list(IDENTITY)
    assert fuse_all(votes, ll, "h1", k=3, ban_identity=True)["a"] == R_B


def test_missing_ll_falls_back_to_vote():
    votes = {"a": _votes(R_A, R_A, R_B, R_C)}
    assert fuse_all(votes, {}, "h1", k=3)["a"] == R_A  # 합의도 2 < 3이지만 LL 없음


def test_vgate_keeps_vote_rank_when_p_identity_low():
    """LL이 순서를 반대해도 vgate는 투표의 순서를 그대로 쓴다 (identity만 관여)."""
    votes = {"a": _votes(R_A, R_A, R_B)}
    ll = {"a": [_ll_view(format_answer(R_C))]}  # identity 사후확률 ~0
    assert fuse_all(votes, ll, "vgate", p_identity_threshold=0.05)["a"] == R_A


def test_vgate_returns_identity_when_p_identity_high():
    """4/4 만장일치라도 LL 사후확률이 문턱을 넘으면 identity로 뒤집는다."""
    votes = {"a": _votes(R_A, R_A, R_A, R_A)}
    ll = {"a": [_ll_view(format_answer(IDENTITY), best_lp=0.0)]}
    assert fuse_all(votes, ll, "vgate", p_identity_threshold=0.05)["a"] == IDENTITY


def test_vgate_keeps_vote_identity_below_threshold():
    """합집합 규칙: 사후확률이 낮아도 투표가 identity면 유지 (투표 게이트 P 0.9496)."""
    votes = {"a": _votes(IDENTITY, IDENTITY, IDENTITY)}
    ll = {"a": [_ll_view(format_answer(R_C))]}  # identity 사후확률 ~0
    assert fuse_all(votes, ll, "vgate", p_identity_threshold=0.05)["a"] == IDENTITY


def test_vgate_falls_back_to_vote_without_ll():
    votes = {"a": _votes(R_A, R_A, R_B)}
    assert fuse_all(votes, {}, "vgate", p_identity_threshold=0.05)["a"] == R_A


def test_vgate_without_threshold_fails_loudly():
    """문턱 없는 vgate는 알기 어려운 TypeError 대신 명시적으로 막는다."""
    votes = {"a": _votes(R_A)}
    ll = {"a": [_ll_view(format_answer(R_A))]}
    with pytest.raises(AssertionError, match="p_identity_threshold"):
        fuse_all(votes, ll, "vgate")


def test_disperse_top_plumbed_through():
    """fuse.py:63이 kwargs 없이 호출해 도달 불가였던 손잡이 — 회귀 잠금."""
    votes = {"a": _votes(R_A, R_B)}  # 최빈 표수 1
    assert fuse_all(votes, {}, "vote", disperse_top=1)["a"] == IDENTITY
    assert fuse_all(votes, {}, "vote", disperse_gate=False)["a"] == R_A


def test_identity_quota_plumbed_through():
    votes = {"a": _votes(R_A, R_A, IDENTITY)}
    assert fuse_all(votes, {}, "vote")["a"] == R_A                      # quota off
    assert fuse_all(votes, {}, "vote", identity_quota=1)["a"] == IDENTITY


def test_vote_policy_matches_aggregate_default():
    votes = {"a": _votes(R_A, R_A, R_B, R_C)}
    assert fuse_all(votes, {}, "vote")["a"] == R_A
