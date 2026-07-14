"""LL 스코어링: 후보 생성(셔플 좌표 정합)·합산 판정·margin·tie-break 검증."""

from src.infer.ll_score import UNORD_KEY, candidate_texts, decide_all
from src.train.targets import format_answer
from src.utils.permutation import ALL_PERMUTATIONS, IDENTITY, shuffle_rank_label

ID_PERM = [0, 1, 2, 3]


def test_candidate_texts_identity_perm():
    cands = dict(candidate_texts(ID_PERM))
    assert len(cands) == 25
    # identity 뷰에서는 원 좌표 rank가 그대로 텍스트에 등장
    for rank in ALL_PERMUTATIONS:
        assert format_answer(rank) in cands[format_answer(rank)]
    assert "UNORDERABLE" in cands[UNORD_KEY]
    assert format_answer(IDENTITY) in cands[UNORD_KEY]


def test_candidate_texts_shuffled_view_uses_shuffled_label():
    """프레임 재배치 뷰의 후보 텍스트는 셔플 좌표 라벨을 담아야 한다 (증강 규칙과 동일)."""
    perm = [2, 0, 3, 1]
    cands = dict(candidate_texts(perm))
    for rank in ALL_PERMUTATIONS:
        expected = format_answer(shuffle_rank_label(rank, perm))
        assert expected in cands[format_answer(rank)]
    # UNORDERABLE 후보는 뷰와 무관하게 동일
    assert cands[UNORD_KEY] == dict(candidate_texts(ID_PERM))[UNORD_KEY]


def _views(scores_by_key, n_tok=10):
    """{키: logprob 합} -> 뷰 1개짜리 ll 리스트."""
    return [{k: [lp, n_tok] for k, lp in scores_by_key.items()}]


def _full_ll(best_key, best_lp=-1.0, other_lp=-50.0, unord_lp=-60.0, n_tok=10):
    ll = {format_answer(r): [other_lp, n_tok] for r in ALL_PERMUTATIONS}
    ll[best_key] = [best_lp, n_tok]
    ll[UNORD_KEY] = [unord_lp, n_tok]
    return ll


def test_decide_picks_max_ll_rank():
    best = format_answer([2, 1, 4, 3])
    preds = decide_all({"a": [_full_ll(best)]}, margin=0.0)
    assert preds["a"] == [2, 1, 4, 3]


def test_decide_sums_over_views():
    """두 뷰 합산: 단일 뷰에서 밀려도 합에서 이기는 후보가 선택된다."""
    k1, k2 = format_answer([2, 1, 4, 3]), format_answer([3, 1, 4, 2])
    v1 = _full_ll(k1, best_lp=-1.0)
    v1[k2] = [-2.0, 10]
    v2 = _full_ll(k2, best_lp=-1.0)
    v2[k1] = [-10.0, 10]
    # 합: k1 = -11, k2 = -3 -> k2 승
    preds = decide_all({"a": [v1, v2]}, margin=0.0)
    assert preds["a"] == [3, 1, 4, 2]


def test_unorderable_margin_gate():
    best = format_answer([2, 1, 4, 3])
    ll = _full_ll(best, best_lp=-10.0, unord_lp=-5.0)  # 평균 lp: best -1.0, unord -0.5
    # margin 0.4 < 차이 0.5 -> UNORDERABLE 판정 (identity)
    assert decide_all({"a": [ll]}, margin=0.4)["a"] == IDENTITY
    # margin 0.6 > 차이 0.5 -> orderable 유지
    assert decide_all({"a": [ll]}, margin=0.6)["a"] == [2, 1, 4, 3]


def test_length_norm_matters_for_unorderable():
    """UNORDERABLE 후보만 길이가 다르므로 norm 여부가 판정을 바꿀 수 있다."""
    best = format_answer([2, 1, 4, 3])
    ll = _full_ll(best, best_lp=-10.0, unord_lp=-8.0)
    ll[UNORD_KEY] = [-8.0, 5]  # 합 -8 > -10 (unord 승) / 평균 -1.6 < -1.0 (orderable 승)
    assert decide_all({"a": [ll]}, margin=0.0, length_norm=False)["a"] == IDENTITY
    assert decide_all({"a": [ll]}, margin=0.0, length_norm=True)["a"] == [2, 1, 4, 3]


def test_ban_identity_excludes_identity_candidate():
    ll = _full_ll(format_answer(IDENTITY), best_lp=-1.0)
    ll[format_answer([2, 1, 3, 4])] = [-2.0, 10]
    assert decide_all({"a": [ll]}, margin=0.0)["a"] == IDENTITY
    assert decide_all({"a": [ll]}, margin=0.0, ban_identity=True)["a"] == [2, 1, 3, 4]


def test_tie_break_deterministic_lexicographic():
    ll = {format_answer(r): [-5.0, 10] for r in ALL_PERMUTATIONS}
    ll[UNORD_KEY] = [-50.0, 10]
    preds = decide_all({"a": [ll]}, margin=0.0)
    # 전 후보 동률 -> 사전순 첫 후보 ("[1, 2, 3, 4]")로 결정적
    assert preds["a"] == IDENTITY
    assert decide_all({"a": [ll]}, margin=0.0)["a"] == preds["a"]
