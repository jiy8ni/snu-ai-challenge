"""targets.py: 4종 타깃 포맷·프롬프트·증강 규칙 검증.

cot/mid/short는 UNORDERABLE 개념이 있는 구 스타일, plain은 없는 신 스타일이므로
스타일 전체를 도는 파라미터라이즈는 UNORD_STYLES / PLAIN_STYLES로 갈라서 쓴다.
"""

import random
import re

import pytest

from src.train.cot_target import build_cot
from src.train.targets import (
    IDENTITY_PRIOR,
    PLAIN_STYLES,
    STYLES,
    augment_perm_and_rank,
    augment_perm_truthful,
    build_instruction,
    build_target,
    chronological_perm,
    format_answer,
    is_unorderable_output,
)
from src.utils.permutation import (
    ALL_PERMUTATIONS,
    IDENTITY,
    order_to_rank,
    parse_permutation,
    shuffle_rank_label,
)

EVENTS = ["a man opens a door", "he walks into the room", "he sits down"]

# UNORDERABLE 분기를 가진 구 스타일 (plain 제외)
UNORD_STYLES = [s for s in STYLES if s not in PLAIN_STYLES]

_ORDER_LINE = re.compile(r"^Order: Frame (\d) -> Frame (\d) -> Frame (\d) -> Frame (\d)$")


def test_mid_orderable_format():
    target = build_target("mid", EVENTS, [2, 3, 1, 4], no_ordering=False)
    lines = target.split("\n")
    assert lines[0] == "Order: Frame 3 -> Frame 1 -> Frame 2 -> Frame 4"
    assert lines[1] == "Conclusion: ORDERABLE"
    assert lines[2] == "Answer: [2, 3, 1, 4]"
    assert parse_permutation(target) == [2, 3, 1, 4]
    assert not is_unorderable_output(target)


@pytest.mark.parametrize("rank", [p for p in ALL_PERMUTATIONS if p != IDENTITY])
def test_mid_order_answer_consistent(rank):
    """Order 줄과 Answer 줄이 서로 역변환 관계여야 한다 (자기모순 방지)."""
    target = build_target("mid", EVENTS, rank, no_ordering=False)
    m = _ORDER_LINE.match(target.split("\n")[0])
    assert m, target
    order = [int(g) for g in m.groups()]
    assert order_to_rank(order) == parse_permutation(target) == rank


@pytest.mark.parametrize("style", UNORD_STYLES)
def test_unorderable_target(style):
    """구 스타일만: no_ordering=True면 Order 줄 없이 UNORDERABLE + identity.

    plain은 이 분기 자체가 없다 (test_plain_ignores_no_ordering_flag 참조).
    """
    target = build_target(style, EVENTS, None, no_ordering=True)
    assert "Order:" not in target
    assert is_unorderable_output(target)
    assert parse_permutation(target) == IDENTITY


@pytest.mark.parametrize("style", ["mid", "short"])
@pytest.mark.parametrize("rank", ALL_PERMUTATIONS)
def test_parse_roundtrip(style, rank):
    if rank == IDENTITY:
        return  # orderable + identity 조합은 데이터에 존재하지 않음
    target = build_target(style, EVENTS, rank, no_ordering=False)
    assert parse_permutation(target) == rank


def test_cot_delegates_to_build_cot():
    for rank, no_ord in ([3, 1, 2, 4], False), (IDENTITY, True):
        assert build_target("cot", EVENTS, rank, no_ord) == build_cot(EVENTS, rank, no_ord)


def test_short_has_no_order_line():
    target = build_target("short", EVENTS, [4, 3, 2, 1], no_ordering=False)
    assert target == "Conclusion: ORDERABLE\nAnswer: [4, 3, 2, 1]"


@pytest.mark.parametrize("style", UNORD_STYLES)
def test_instruction_mentions_format(style):
    inst = build_instruction(style, 'A man walks, then sits.')
    assert "A man walks, then sits." in inst
    assert "UNORDERABLE" in inst
    assert "Answer: [r1, r2, r3, r4]" in inst
    if style == "mid":
        assert "Order: Frame a" in inst


def test_augment_orderable_never_identity():
    rng = random.Random(0)
    for rank in (r for r in ALL_PERMUTATIONS if r != IDENTITY):
        for _ in range(30):
            perm, new_rank = augment_perm_and_rank(rank, no_ordering=False, rng=rng)
            assert sorted(perm) == [0, 1, 2, 3]
            assert new_rank != IDENTITY
            assert shuffle_rank_label(rank, perm) == new_rank  # 라벨-이미지 재배치 일관성


def test_augment_no_ordering_rank_fixed():
    rng = random.Random(1)
    seen_perms = set()
    for _ in range(50):
        perm, new_rank = augment_perm_and_rank(None, no_ordering=True, rng=rng)
        assert new_rank == IDENTITY
        assert sorted(perm) == [0, 1, 2, 3]
        seen_perms.add(tuple(perm))
    assert len(seen_perms) > 1  # 이미지 배치는 실제로 섞인다


def test_format_answer_matches_csv_style():
    assert format_answer([3, 1, 2, 4]) == "[3, 1, 2, 4]"


# ---------------------------------------------------------------------------
# plain 스타일 + 진실 라벨 증강 (_0716 No_ordering 재해석 레시피)
# 근거: src/train/targets.py docstring / reports/preprocessing.md
# ---------------------------------------------------------------------------


def test_plain_target_format():
    """Order/Answer 2줄. Conclusion·UNORDERABLE 없음 (그 개념이 없는 스타일)."""
    target = build_target("plain", EVENTS, [2, 3, 1, 4], no_ordering=False)
    assert target == "Order: Frame 3 -> Frame 1 -> Frame 2 -> Frame 4\nAnswer: [2, 3, 1, 4]"
    assert "Conclusion" not in target
    assert not is_unorderable_output(target)


def test_plain_ignores_no_ordering_flag():
    """plain은 no_ordering 인자를 무시하고 rank만 본다.

    진실 레시피에서 no_ordering 레코드는 rank=[1,2,3,4](진실)로 들어오므로
    자연히 identity 타깃이 나온다 — 특례 분기가 필요 없다.
    """
    on = build_target("plain", EVENTS, [2, 3, 1, 4], no_ordering=True)
    off = build_target("plain", EVENTS, [2, 3, 1, 4], no_ordering=False)
    assert on == off
    identity_target = build_target("plain", EVENTS, list(IDENTITY), no_ordering=True)
    assert identity_target == "Order: Frame 1 -> Frame 2 -> Frame 3 -> Frame 4\nAnswer: [1, 2, 3, 4]"


@pytest.mark.parametrize("rank", ALL_PERMUTATIONS)
def test_plain_order_answer_consistent(rank):
    """identity 포함 24개 전부: Order 줄과 Answer 줄이 역변환 관계."""
    target = build_target("plain", EVENTS, rank, no_ordering=False)
    m = _ORDER_LINE.match(target.split("\n")[0])
    assert m, target
    order = [int(g) for g in m.groups()]
    assert order_to_rank(order) == parse_permutation(target) == rank


def test_plain_instruction_has_no_unorderable():
    inst = build_instruction("plain", "A man walks, then sits.")
    assert "A man walks, then sits." in inst
    assert "UNORDERABLE" not in inst  # 존재하지 않는 개념을 가르치지 않는다
    assert "Answer: [r1, r2, r3, r4]" in inst
    assert "Order: Frame a" in inst
    assert "[1, 2, 3, 4]" in inst  # 사전확률 안내 (이미 시간순이면 identity)
    assert "POSSIBLY SHUFFLED" in inst  # 미셔플 15.5%에 거짓이 되지 않도록


@pytest.mark.parametrize("rank", ALL_PERMUTATIONS)
def test_chronological_perm_yields_identity(rank):
    perm = chronological_perm(rank)
    assert sorted(perm) == [0, 1, 2, 3]
    assert shuffle_rank_label(rank, perm) == IDENTITY


@pytest.mark.parametrize("rank", ALL_PERMUTATIONS)
def test_augment_truthful_label_always_true(rank):
    """핵심 불변식: 라벨이 항상 실제 재배치와 일치한다 (identity rank 입력 포함).

    구 augment_perm_and_rank의 no_ordering 분기가 깨뜨리던 바로 그 성질.
    """
    rng = random.Random(0)
    for _ in range(50):
        perm, new_rank = augment_perm_truthful(rank, rng)
        assert sorted(perm) == [0, 1, 2, 3]
        assert shuffle_rank_label(rank, perm) == new_rank


def test_augment_truthful_prior_boundaries():
    rng = random.Random(0)
    for rank in ALL_PERMUTATIONS:
        perm, new_rank = augment_perm_truthful(rank, rng, identity_prior=1.0)
        assert new_rank == IDENTITY
        assert perm == chronological_perm(rank)
        _, new_rank = augment_perm_truthful(rank, rng, identity_prior=0.0)
        assert new_rank != IDENTITY


def test_augment_truthful_identity_rate_matches_prior():
    """P(라벨=identity) == identity_prior. 균등(1/24=0.042)이 아니어야 한다."""
    rng = random.Random(42)
    n = 3000
    hits = sum(
        augment_perm_truthful(rng.choice(ALL_PERMUTATIONS), rng)[1] == IDENTITY
        for _ in range(n)
    )
    assert abs(hits / n - IDENTITY_PRIOR) < 0.02, f"identity rate {hits / n}"


def test_augment_truthful_rejects_bad_prior():
    with pytest.raises(AssertionError):
        augment_perm_truthful([1, 2, 3, 4], random.Random(0), identity_prior=1.5)
