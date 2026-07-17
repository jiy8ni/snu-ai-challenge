"""LL 스코어링: 후보 생성(셔플 좌표 정합)·합산 판정·margin·tie-break·p_identity 검증."""

import math

import pytest

from src.infer.ll_score import (
    UNORD_KEY,
    candidate_texts,
    decide_all,
    decide_one,
    identity_posterior,
)
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


@pytest.mark.parametrize("perm", [ID_PERM, [2, 0, 3, 1]])
def test_candidate_texts_plain_has_24_and_no_unord(perm):
    """plain은 UNORDERABLE 개념이 없으므로 후보가 24개 (src/train/targets.py 참조)."""
    cands = dict(candidate_texts(perm, style="plain"))
    assert len(cands) == 24
    assert UNORD_KEY not in cands
    for rank in ALL_PERMUTATIONS:
        text = cands[format_answer(rank)]
        assert format_answer(shuffle_rank_label(rank, perm)) in text
        assert "UNORDERABLE" not in text
        assert "Conclusion" not in text


def test_plain_scores_decide_without_unord_key():
    """decide_one은 UNORD_KEY가 없어도 동작해야 한다 (plain 경로)."""
    scores = {format_answer(r): 0.0 for r in ALL_PERMUTATIONS}
    scores[format_answer([3, 1, 2, 4])] = 1.0
    assert decide_one(scores, margin=0.0) == [3, 1, 2, 4]
    # identity가 최고여도 그대로 채택 (plain에선 "이미 시간순"이라는 정상 답)
    scores2 = {format_answer(r): 0.0 for r in ALL_PERMUTATIONS}
    scores2[format_answer(IDENTITY)] = 1.0
    assert decide_one(scores2, margin=0.0) == list(IDENTITY)


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


# --- identity_posterior (no_ordering 검출 신호, val 953 AUC 0.8844) ---


def test_identity_posterior_uniform_is_one_over_24():
    ll = {format_answer(r): [-5.0, 39] for r in ALL_PERMUTATIONS}
    ll[UNORD_KEY] = [-1.0, 22]
    assert identity_posterior([ll]) == pytest.approx(1 / 24)


def test_identity_posterior_ignores_unorderable_key():
    """UNORD 후보는 길이(22 vs 39)도 사건도 달라 24-way 분포에서 제외된다."""
    base = _full_ll(format_answer(IDENTITY), best_lp=-1.0, n_tok=39)
    p1 = identity_posterior([base])
    swung = dict(base, **{UNORD_KEY: [+100.0, 22]})
    assert identity_posterior([swung]) == pytest.approx(p1)


def test_identity_posterior_uses_raw_sums_not_length_norm():
    """토큰 수를 바꿔도 사후확률은 불변 — length-norm(온도 1/39)이 신호를 뭉개는 것을 차단.

    실측: length_norm 적용 시 AUC 0.8844 -> 0.8028, p 범위가 (0.034, 0.056)로 붕괴.
    """
    a = _full_ll(format_answer(IDENTITY), best_lp=-1.0, other_lp=-5.0, n_tok=39)
    b = _full_ll(format_answer(IDENTITY), best_lp=-1.0, other_lp=-5.0, n_tok=7)
    assert identity_posterior([a]) == pytest.approx(identity_posterior([b]))


def test_identity_posterior_averages_posteriors_across_views():
    """뷰 간 사후확률 평균 (logprob 합산이 아님) -> 문턱이 --tta 수에 불변."""
    hi = _full_ll(format_answer(IDENTITY), best_lp=0.0, other_lp=-50.0, n_tok=39)
    lo = _full_ll(format_answer([2, 1, 4, 3]), best_lp=0.0, other_lp=-50.0, n_tok=39)
    assert identity_posterior([hi]) == pytest.approx(1.0, abs=1e-6)
    assert identity_posterior([lo]) == pytest.approx(0.0, abs=1e-6)
    assert identity_posterior([hi, lo]) == pytest.approx(0.5, abs=1e-6)
    # 뷰를 복제해도 값이 그대로여야 한다 (합산이면 예리해져 달라진다)
    assert identity_posterior([hi, lo, hi, lo]) == pytest.approx(0.5, abs=1e-6)


def test_identity_posterior_rejects_partial_candidates():
    """score 중단본(24후보 미만)이 조용히 틀린 사후확률을 내지 않도록 방어."""
    ll = {format_answer(r): [-5.0, 39] for r in ALL_PERMUTATIONS[:23]}
    ll[UNORD_KEY] = [-1.0, 22]
    with pytest.raises(AssertionError):
        identity_posterior([ll])


def test_p_identity_gate_returns_identity_above_threshold():
    scores = {format_answer(r): -5.0 for r in ALL_PERMUTATIONS}
    scores[format_answer([2, 1, 4, 3])] = -1.0  # LL argmax는 비-identity
    scores[UNORD_KEY] = -50.0
    assert decide_one(scores, p_identity=0.9, p_identity_threshold=0.05) == IDENTITY


def test_p_identity_gate_bans_identity_below_threshold():
    """게이트가 orderable로 판정하면 identity는 정의상 오답 -> 후보에서 제외."""
    scores = {format_answer(r): -50.0 for r in ALL_PERMUTATIONS}
    scores[format_answer(IDENTITY)] = -1.0   # identity가 LL 최고점
    scores[format_answer([2, 1, 4, 3])] = -2.0
    scores[UNORD_KEY] = -99.0
    assert decide_one(scores, p_identity=0.001, p_identity_threshold=0.05) == [2, 1, 4, 3]


def test_p_identity_threshold_none_preserves_legacy_path():
    ll = _full_ll(format_answer(IDENTITY), best_lp=-1.0)
    ll[format_answer([2, 1, 3, 4])] = [-2.0, 10]
    scores = {k: lp for k, (lp, _) in ll.items()}
    assert decide_one(scores) == decide_one(scores, p_identity=0.9)  # 문턱 없으면 미작동
    assert decide_one(scores, p_identity_threshold=0.05) == decide_one(scores)


def test_margin_none_disables_unorderable_gate():
    best = format_answer([2, 1, 4, 3])
    ll = _full_ll(best, best_lp=-10.0, unord_lp=-5.0)  # unord가 이겨 identity가 될 상황
    scores = {k: lp / n for k, (lp, n) in ll.items()}
    assert decide_one(scores, margin=0.4) == IDENTITY
    assert decide_one(scores, margin=None) == [2, 1, 4, 3]
