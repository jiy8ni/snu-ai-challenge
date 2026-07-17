"""SFT 타깃 포맷 통합 빌더 (Track B): cot / mid / short / plain 4종 + 프롬프트 + 증강 규칙.

학습(vl_dataset)·추론(infer/predict)·테스트가 전부 이 모듈 하나를 통해
같은 텍스트를 생성하도록 단일 창구로 둔다.

- cot   : 기존 4-스텝 CoT (src/train/cot_target.build_cot 위임, ~130-230 tok)
- mid   : Order -> Conclusion -> Answer (~35 tok, 구 라운드 기본값)
- short : Conclusion -> Answer (~20 tok)
- plain : Order -> Answer (~30 tok). **_0716 재해석 레시피** — 아래 참조

cot/mid/short는 파싱이 permutation.parse_permutation + "UNORDERABLE" 문자열 검사로 통일된다.

## plain 스타일과 No_ordering 재해석 (2026-07-16)

공식 정의는 **"No_ordering=True일 경우 이미지는 셔플링되지 않았으며, 정답은 [1,2,3,4]로
고정"** — 즉 *생성 과정*(안 섞음)에 대한 서술이지 *콘텐츠 속성*(정렬 불가)이 아니다.
구 스타일(cot/mid/short)은 이를 "정렬 불가(UNORDERABLE)"로 오독한 위에 서 있다.

실측 근거 (reports/preprocessing.md):
  - em_no_ordering 0.596 > em_orderable 0.5025 — TTA는 no_ordering 샘플도 셔플하는데
    모델이 원순서를 59.6% 복원한다(무작위 4%). 정렬 불가면 불가능한 수치.
  - 노이즈 지표(검은 프레임·근접중복)가 두 클래스에 직교 (0.97x)
  - orderable 8,057건 중 identity 정답 0건, 등장 순열 23종 — 출제자가 identity를
    미셔플 전용으로 예약했다 (제출 형식이 순열만 받아 센티넬 자리가 없기 때문)

따라서 plain은 UNORDERABLE 개념 자체를 없애고 **순수 24-way 순서 맞추기**로 둔다.
모델이 [1,2,3,4]를 답하면 그것이 곧 "이미 시간순" = No_ordering이고 그대로 제출하면 정답.

구 스타일은 동결된 구 체크포인트·라운드 재현을 위해 **의도적으로 그대로 둔다**.
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

STYLES = ("cot", "mid", "short", "plain")

# UNORDERABLE 개념이 없는 스타일 (No_ordering 재해석 이후 레시피). 모듈 docstring 참조.
PLAIN_STYLES = ("plain",)

UNORDERABLE_TOKEN = "UNORDERABLE"

# test 실측 사전확률: No_ordering(=미셔플) 비율. train 9,535건 중 1,478건 = 0.1550.
# 진실 증강에서 "시간순 배치"를 뽑을 확률로 쓴다 — 균등(1/24=0.042)으로 두면 test와
# 어긋나고, 특정 불가능한 샘플에서 최적 베팅(identity)을 잃는다.
IDENTITY_PRIOR = 0.155


def format_answer(rank):
    """train.csv Answer 컬럼과 동일한 문자열 포맷: "[3, 1, 2, 4]"."""
    return "[" + ", ".join(str(r) for r in rank) + "]"


def build_target(style, events, rank, no_ordering):
    """스타일별 SFT 타깃 텍스트. 순수 함수 — 증강 후에는 반드시 재호출한다.

    plain은 no_ordering 인자를 **무시**한다 (그 개념이 없는 스타일). rank가 진실이면
    no_ordering 레코드는 rank=[1,2,3,4]이므로 자연히 identity 타깃이 된다.
    """
    assert style in STYLES, f"unknown style: {style}"
    if style == "cot":
        return build_cot(events, rank, no_ordering)

    if style == "plain":
        assert is_valid_permutation(rank), f"invalid rank: {rank}"
        order = rank_to_order(rank)
        return (
            "Order: " + " -> ".join(f"Frame {k}" for k in order) + "\n"
            f"Answer: {format_answer(rank)}"
        )

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
    "plain": (
        "Respond in exactly this format:\n"
        "Order: Frame a -> Frame b -> Frame c -> Frame d\n"
        "Answer: [r1, r2, r3, r4]"
    ),
}


def build_instruction(style, caption):
    """4개 이미지 뒤에 붙는 사용자 지시문. 학습·추론 공용.

    plain은 별도 지시문을 쓴다 (모듈 docstring의 재해석 참조):
      - "shown in SHUFFLED order"는 미셔플 15.5%에 **거짓**이라 "possibly shuffled"로
      - UNORDERABLE 문장 제거 (존재하지 않는 개념을 가르치던 자리)
      - 대신 "이미 시간순이면 [1,2,3,4]"를 명시 — 사전확률 15.5%를 프롬프트로 알린다
    """
    assert style in STYLES, f"unknown style: {style}"
    if style == "plain":
        return (
            "The four images above are frames from a single video, shown in a POSSIBLY "
            "SHUFFLED order and labeled Frame 1, Frame 2, Frame 3, Frame 4.\n"
            f'Caption describing the video in temporal order: "{caption}"\n'
            "Task: use the caption and visual cues to recover the chronological order of the frames. "
            "r_i in the Answer is the temporal position (1-4) of Frame i.\n"
            "The frames were not necessarily shuffled: if they are already in chronological "
            f"order, the answer is {format_answer(IDENTITY)}.\n"
            + _FORMAT_SPEC[style]
        )
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
    """순열 증강용 (perm, 증강 후 타깃 rank) 샘플링. **구 스타일(cot/mid/short) 전용.**

    - perm: 0-indexed 재배치 (new_inputs[j] = old_inputs[perm[j]]). 이미지 재배열에 사용.
    - orderable 샘플은 증강 결과 rank가 identity가 되는 perm을 재추출로 배제한다.
      (데이터 불변식 "orderable -> 비identity"가 test에도 성립한다고 보고 분포를 보존)
    - no_ordering 샘플은 프레임 배치와 무관하게 타깃 rank = [1,2,3,4] 고정.

    ⚠️ 마지막 규칙은 "No_ordering = 정렬 불가" 오독의 산물이다 (모듈 docstring 참조):
    프레임을 섞어놓고 라벨을 identity로 고정하므로 **거짓 라벨을 가르친다**. 신규
    레시피는 augment_perm_truthful을 쓴다. 이 함수는 구 체크포인트 재현용으로만 남긴다.
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


def chronological_perm(rank):
    """현재 배치를 시간순으로 되돌리는 perm. new_inputs[j] = old_inputs[perm[j]].

    rank[i] = 입력 프레임 i의 시간 순위이므로, 시간순 k번째 자리(j=k-1)에 와야 할
    입력 프레임은 rank가 k인 프레임 = rank.index(k). 즉 rank_to_order의 0-indexed 판.
    이 perm으로 재배치하면 타깃은 정의상 IDENTITY가 된다.
    """
    assert is_valid_permutation(rank), f"invalid rank: {rank}"
    return [rank.index(k + 1) for k in range(N_FRAMES)]


def augment_perm_truthful(rank, rng=None, identity_prior=IDENTITY_PRIOR):
    """순열 증강 (perm, 타깃 rank) — **라벨이 항상 진실**. plain 레시피 전용.

    augment_perm_and_rank와의 차이는 no_ordering 특례가 **없다**는 것이다.
    "No_ordering"은 샘플의 속성이 아니라 (샘플, 배치)의 속성 — 지금 배치가 우연히
    시간순이면 그 뷰가 no_ordering인 것이다 (모듈 docstring의 재해석 참조).
    따라서 두 클래스를 구분하지 않고 동일 규칙으로 처리한다. no_ordering 레코드는
    rank=[1,2,3,4](진실)로 들어오므로 자연히 올바른 라벨이 나온다.

    분포 (test 정합):
      - 확률 identity_prior: 시간순 배치 -> 라벨 IDENTITY
      - 나머지: 라벨이 IDENTITY가 아닌 균등 perm -> 라벨 shuffle_rank_label(rank, perm)
      즉 P(라벨=identity) = identity_prior, 나머지 23개 라벨 각 (1-prior)/23.
      균등(1/24=0.042)으로 두면 test의 15.5%와 어긋나 "특정 불가 시 identity 베팅"
      (최빈 라벨이라 최적)을 잃는다. IDENTITY_PRIOR 주석 참조.

    불변식: 반환된 (perm, new_rank)는 항상 shuffle_rank_label(rank, perm) == new_rank.
    """
    rng = rng or random
    assert is_valid_permutation(rank), f"invalid rank: {rank}"
    assert 0.0 <= identity_prior <= 1.0, f"identity_prior must be in [0, 1]: {identity_prior}"

    if rng.random() < identity_prior:
        perm = chronological_perm(rank)
        return perm, list(IDENTITY)

    perm = list(range(N_FRAMES))
    while True:
        rng.shuffle(perm)
        new_rank = shuffle_rank_label(rank, perm)
        if new_rank != IDENTITY:
            return perm, new_rank
