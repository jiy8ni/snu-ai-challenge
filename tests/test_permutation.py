import itertools
import random

import pytest

from src.utils.permutation import (
    ALL_PERMUTATIONS,
    IDENTITY,
    order_to_rank,
    parse_permutation,
    rank_to_order,
    shuffle_rank_label,
    unshuffle_rank_label,
)


def test_baseline_example():
    # baseline 노트북의 예시: 모델 출력 [4,2,1,3] -> 제출 [3,2,4,1]
    assert order_to_rank([4, 2, 1, 3]) == [3, 2, 4, 1]
    assert rank_to_order([3, 2, 4, 1]) == [4, 2, 1, 3]


def test_order_rank_inverse_all_24():
    for p in ALL_PERMUTATIONS:
        assert rank_to_order(order_to_rank(p)) == p
        assert order_to_rank(rank_to_order(p)) == p


def test_identity_fixed_point():
    assert order_to_rank(IDENTITY) == IDENTITY
    assert rank_to_order(IDENTITY) == IDENTITY


def test_shuffle_semantics():
    # 원본 시간순: 프레임 3, 1, 4, 2 (order) -> rank [2, 4, 1, 3]
    rank = [2, 4, 1, 3]
    # 입력을 역순으로 재배치: new_inputs[j] = old_inputs[3-j]
    perm = [3, 2, 1, 0]
    # 재배치된 프레임들의 rank는 그대로 따라간다
    assert shuffle_rank_label(rank, perm) == [3, 1, 4, 2]


def test_shuffle_unshuffle_roundtrip():
    rng = random.Random(0)
    for _ in range(200):
        rank = list(rng.choice(ALL_PERMUTATIONS))
        perm = list(range(4))
        rng.shuffle(perm)
        shuffled = shuffle_rank_label(rank, perm)
        assert unshuffle_rank_label(shuffled, perm) == rank


def test_shuffle_consistency_with_order():
    # 재배치 후에도 "시간순으로 프레임을 나열한 실제 내용"은 불변이어야 한다.
    # old 입력의 내용을 그 인덱스로 표현하면, 시간순 내용 나열은
    # perm[rank_to_order(...)-1] 로 복원된다.
    rng = random.Random(1)
    for _ in range(100):
        rank = list(rng.choice(ALL_PERMUTATIONS))
        perm = list(range(4))
        rng.shuffle(perm)
        shuffled_rank = shuffle_rank_label(rank, perm)
        old_content_in_time = [k - 1 for k in rank_to_order(rank)]
        new_content_in_time = [perm[k - 1] for k in rank_to_order(shuffled_rank)]
        assert new_content_in_time == old_content_in_time


@pytest.mark.parametrize(
    "text,expected",
    [
        ("[1, 2, 3, 4]", [1, 2, 3, 4]),
        ("The answer is [4, 2, 1, 3].", [4, 2, 1, 3]),
        # CoT: 마지막 유효 순열을 취한다
        ("candidates [1,2,3,4] ... final answer: [2,1,4,3]", [2, 1, 4, 3]),
        # 마지막 리스트가 무효면 그 앞의 유효한 것으로 fallback
        ("[3,1,2,4] then scores [0.9, 0.1]", [3, 1, 2, 4]),
        ("[1, 2, 3]", None),
        ("[1, 2, 3, 3]", None),
        ("no list here", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_permutation(text, expected):
    assert parse_permutation(text) == expected
