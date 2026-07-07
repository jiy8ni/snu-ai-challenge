"""융합 결정: 합의도 게이트·LL 재순위·ban_identity·폴백 검증."""

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


def test_vote_policy_matches_aggregate_default():
    votes = {"a": _votes(R_A, R_A, R_B, R_C)}
    assert fuse_all(votes, {}, "vote")["a"] == R_A
