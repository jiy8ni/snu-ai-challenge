"""targets.py: 3종 타깃 포맷·프롬프트·증강 규칙 검증."""

import random
import re

import pytest

from src.train.cot_target import build_cot
from src.train.targets import (
    STYLES,
    augment_perm_and_rank,
    build_instruction,
    build_target,
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


@pytest.mark.parametrize("style", STYLES)
def test_unorderable_target(style):
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


@pytest.mark.parametrize("style", STYLES)
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
