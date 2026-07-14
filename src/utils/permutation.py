"""순열 표현 변환 유틸.

두 가지 표현을 오간다 (docs/label_encoding.md 참조):

- rank  (제출 형식): rank[i]  = 입력 프레임 i(0-indexed 위치)의 시간 순위 (1~4)
- order (시간순 나열): order[k] = 시간순 k번째인 입력 프레임 번호 (1~4)

예: order [4, 2, 1, 3] <-> rank [3, 2, 4, 1]

모든 순열 변환은 반드시 이 모듈의 함수만 사용한다.
"""

import ast
import itertools
import re

N_FRAMES = 4
IDENTITY = [1, 2, 3, 4]
ALL_PERMUTATIONS = [list(p) for p in itertools.permutations(range(1, N_FRAMES + 1))]


def is_valid_permutation(p):
    return isinstance(p, (list, tuple)) and sorted(p) == IDENTITY


def order_to_rank(order):
    """시간순 나열 -> 제출 rank. order_to_rank([4,2,1,3]) == [3,2,4,1]"""
    assert is_valid_permutation(order), f"invalid permutation: {order}"
    rank = [0] * N_FRAMES
    for k, frame_num in enumerate(order):
        rank[frame_num - 1] = k + 1
    return rank


def rank_to_order(rank):
    """제출 rank -> 시간순 나열. order_to_rank의 역함수."""
    assert is_valid_permutation(rank), f"invalid permutation: {rank}"
    order = [0] * N_FRAMES
    for i, r in enumerate(rank):
        order[r - 1] = i + 1
    return order


def shuffle_rank_label(rank, perm):
    """순열 증강: 입력 프레임을 재배치했을 때의 rank 라벨.

    perm: 0-indexed 재배치. new_inputs[j] = old_inputs[perm[j]].
    rank는 프레임에 붙는 속성이므로 라벨도 같은 방식으로 따라간다.
    """
    assert sorted(perm) == list(range(N_FRAMES)), f"invalid perm: {perm}"
    assert is_valid_permutation(rank), f"invalid permutation: {rank}"
    return [rank[perm[j]] for j in range(N_FRAMES)]


def unshuffle_rank_label(rank_shuffled, perm):
    """TTA 역변환: 셔플된 입력에 대한 예측 rank를 원 입력 좌표로 되돌린다.

    shuffle_rank_label(rank, perm) == rank_shuffled 를 만족하는 rank를 반환.
    """
    assert sorted(perm) == list(range(N_FRAMES)), f"invalid perm: {perm}"
    assert is_valid_permutation(rank_shuffled), f"invalid permutation: {rank_shuffled}"
    rank = [0] * N_FRAMES
    for j in range(N_FRAMES):
        rank[perm[j]] = rank_shuffled[j]
    return rank


_LIST_RE = re.compile(r"\[[^\[\]]*\]")


def parse_permutation(text):
    """모델 출력 텍스트에서 {1,2,3,4} 순열 리스트를 추출. 실패 시 None.

    여러 리스트가 있으면 마지막에 등장하는 유효한 순열을 취한다
    (CoT 출력에서 최종 답이 마지막에 오므로).
    """
    if not text:
        return None
    for m in reversed(_LIST_RE.findall(text)):
        try:
            cand = ast.literal_eval(m)
        except (ValueError, SyntaxError):
            continue
        if is_valid_permutation(cand):
            return list(cand)
    return None


def parse_answer_column(s):
    """train.csv Answer 컬럼("[3, 1, 2, 4]" 형태) 파싱."""
    p = ast.literal_eval(s)
    assert is_valid_permutation(p), f"invalid Answer: {s}"
    return list(p)
