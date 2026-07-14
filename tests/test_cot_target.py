import random

from src.train.cot_target import build_cot
from src.utils.permutation import (
    ALL_PERMUTATIONS,
    parse_permutation,
    shuffle_rank_label,
)

EVENTS_2 = ["a man runs", "he jumps into the pool"]
EVENTS_4 = ["intro screen", "a man runs", "he jumps", "he lands in the pool"]


def test_orderable_answer_is_rank():
    rank = [2, 3, 1, 4]
    cot = build_cot(EVENTS_2, rank, no_ordering=False)
    assert "Conclusion: ORDERABLE" in cot
    assert parse_permutation(cot) == rank


def test_orderable_timeline_uses_order():
    # rank [2,3,1,4] -> 시간순 프레임 나열 3,1,2,4
    cot = build_cot(EVENTS_2, [2, 3, 1, 4], no_ordering=False)
    assert "1st in time: Frame 3" in cot
    assert "2nd in time: Frame 1" in cot
    assert "4th in time: Frame 4" in cot


def test_event_annotation_only_when_four_events():
    cot4 = build_cot(EVENTS_4, [1, 2, 3, 4], no_ordering=False)
    assert "(E1)" in cot4
    cot2 = build_cot(EVENTS_2, [1, 2, 3, 4], no_ordering=False)
    assert "(E1)" not in cot2


def test_unorderable():
    cot = build_cot(EVENTS_2, [1, 2, 3, 4], no_ordering=True)
    assert "Conclusion: UNORDERABLE" in cot
    assert parse_permutation(cot) == [1, 2, 3, 4]
    assert "Matching frames" not in cot


def test_all_24_permutations_roundtrip():
    for rank in ALL_PERMUTATIONS:
        cot = build_cot(EVENTS_2, rank, no_ordering=False)
        assert parse_permutation(cot) == rank


def test_augmentation_regeneration_consistency():
    # 증강 시나리오: 입력을 perm으로 재배치 -> 라벨 변환 -> CoT 재생성
    # 재생성된 CoT의 Answer는 변환된 라벨과 일치해야 한다.
    rng = random.Random(7)
    for _ in range(50):
        rank = list(rng.choice(ALL_PERMUTATIONS))
        perm = list(range(4))
        rng.shuffle(perm)
        new_rank = shuffle_rank_label(rank, perm)
        cot = build_cot(EVENTS_2, new_rank, no_ordering=False)
        assert parse_permutation(cot) == new_rank
