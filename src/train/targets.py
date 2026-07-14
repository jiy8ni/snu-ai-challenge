"""SFT 타깃 포맷 통합 빌더 (Track B): cot / mid / short 3종 + 프롬프트 + 증강 규칙.

학습(vl_dataset)·추론(infer/predict)·테스트가 전부 이 모듈 하나를 통해
같은 텍스트를 생성하도록 단일 창구로 둔다.

- cot   : 기존 4-스텝 CoT (src/train/cot_target.build_cot 위임, ~130-230 tok)
- mid   : Order -> Conclusion -> Answer (~35 tok, 기본값)
- short : Conclusion -> Answer (~20 tok)

모든 스타일은 마지막 두 줄이 "Conclusion: ORDERABLE|UNORDERABLE" / "Answer: [r1, r2, r3, r4]"
로 끝나므로, 파싱은 permutation.parse_permutation + "UNORDERABLE" 문자열 검사로 통일된다.
"""

import random

from src.train.cot_target import build_cot
from src.utils.permutation import (
    IDENTITY,
    N_FRAMES,
    is_valid_permutation,
    rank_to_order,
    shuffle_rank_label,
)

STYLES = ("cot", "mid", "short")

UNORDERABLE_TOKEN = "UNORDERABLE"


def format_answer(rank):
    """train.csv Answer 컬럼과 동일한 문자열 포맷: "[3, 1, 2, 4]"."""
    return "[" + ", ".join(str(r) for r in rank) + "]"


def build_target(style, events, rank, no_ordering):
    """스타일별 SFT 타깃 텍스트. 순수 함수 — 증강 후에는 반드시 재호출한다."""
    assert style in STYLES, f"unknown style: {style}"
    if style == "cot":
        return build_cot(events, rank, no_ordering)

    if no_ordering:
        return f"Conclusion: {UNORDERABLE_TOKEN}\nAnswer: {format_answer(IDENTITY)}"

    assert is_valid_permutation(rank), f"invalid rank: {rank}"
    lines = []
    if style == "mid":
        order = rank_to_order(rank)
        lines.append("Order: " + " -> ".join(f"Frame {k}" for k in order))
    lines.append("Conclusion: ORDERABLE")
    lines.append(f"Answer: {format_answer(rank)}")
    return "\n".join(lines)


_FORMAT_SPEC = {
    "cot": (
        "Respond in exactly this format:\n"
        "Step 1 - Events described in the caption, in temporal order:\n"
        "  E1. <event>\n  ...\n"
        "Step 2 - Matching frames to the timeline (or why the frames cannot be uniquely ordered):\n"
        "  1st in time: Frame <n>\n  ...\n"
        "Step 3 - Temporal position of each frame: Frame 1 -> <ordinal>, ...\n"
        "Conclusion: ORDERABLE or UNORDERABLE\n"
        "Answer: [r1, r2, r3, r4]"
    ),
    "mid": (
        "Respond in exactly this format (omit the Order line if UNORDERABLE):\n"
        "Order: Frame a -> Frame b -> Frame c -> Frame d\n"
        "Conclusion: ORDERABLE or UNORDERABLE\n"
        "Answer: [r1, r2, r3, r4]"
    ),
    "short": (
        "Respond in exactly this format:\n"
        "Conclusion: ORDERABLE or UNORDERABLE\n"
        "Answer: [r1, r2, r3, r4]"
    ),
}


def build_instruction(style, caption):
    """4개 이미지 뒤에 붙는 사용자 지시문. 학습·추론 공용."""
    assert style in STYLES, f"unknown style: {style}"
    return (
        "The four images above are frames from a single video, shown in SHUFFLED order "
        "and labeled Frame 1, Frame 2, Frame 3, Frame 4.\n"
        f'Caption describing the video in temporal order: "{caption}"\n'
        "Task: use the caption and visual cues to recover the chronological order of the frames. "
        "r_i in the Answer is the temporal position (1-4) of Frame i.\n"
        "If the frames cannot be uniquely ordered from the caption (e.g., several frames depict "
        f"the same described event), conclude {UNORDERABLE_TOKEN} and answer {format_answer(IDENTITY)}.\n"
        + _FORMAT_SPEC[style]
    )


def is_unorderable_output(text):
    """모델 출력에서 UNORDERABLE 결론 여부 (Conclusion 줄 기준)."""
    return bool(text) and UNORDERABLE_TOKEN in text


def augment_perm_and_rank(rank, no_ordering, rng=None):
    """순열 증강용 (perm, 증강 후 타깃 rank) 샘플링.

    - perm: 0-indexed 재배치 (new_inputs[j] = old_inputs[perm[j]]). 이미지 재배열에 사용.
    - orderable 샘플은 증강 결과 rank가 identity가 되는 perm을 재추출로 배제한다.
      (데이터 불변식 "orderable -> 비identity"가 test에도 성립한다고 보고 분포를 보존)
    - no_ordering 샘플은 프레임 배치와 무관하게 타깃 rank = [1,2,3,4] 고정.
    """
    rng = rng or random
    perm = list(range(N_FRAMES))
    if no_ordering:
        rng.shuffle(perm)
        return perm, list(IDENTITY)
    assert is_valid_permutation(rank), f"invalid rank: {rank}"
    while True:
        rng.shuffle(perm)
        new_rank = shuffle_rank_label(rank, perm)
        if new_rank != IDENTITY:
            return perm, new_rank
