"""Rule-based caption augmentation for temporal connectives.

This module is the rule-based caption augmenter and the only caption
augmentation path in use (external-LLM paraphrase augmentation was withdrawn
per competition rules).  Every rewrite here must round-trip through
``caption_events.split_events`` unchanged.

연결어 풀은 파서(``caption_events._SEQ_SPLIT`` / ``_MID_AFTER`` /
``_LEADING_AFTER``)가 인식하는 형태만 렌더한다: ``then`` / ``followed by`` /
``finally`` / ``next,`` / ``ending with`` / ``afterwards`` / ``subsequently`` /
``;`` / ``before`` (서술 순서 = 시간 순서), 그리고 ``after`` (인접쌍 역전).
후자 두 종류의 "after" 템플릿은 서술 순서를 시간 순서와 어긋나게(비단조) 만들되
round-trip 상으로는 원래 이벤트 순서를 복원하도록 구성해, 파서가 역전을
정확히 되돌리는지까지 데이터 다양성으로 커버한다.

``caption_variants``는 결정적 템플릿만 사용한다 — 추론 캡션 TTA 재현성 계약.
반면 ``_mixed_chain``은 위치별 연결어를 무작위로 섞는 **학습 전용** 변형이라
``caption_variants``에는 넣지 않고 ``augment_caption``에서만 후보로 얹는다.
"""

import random

from src.preprocess.caption_events import split_events


def _finish(text):
    text = text.strip()
    return text if text.endswith((".", "!", "?")) else text + "."


def _then_chain(events):
    return _finish(", then ".join(events))


def _followed_by_chain(events):
    return _finish(", followed by ".join(events))


def _finally_chain(events):
    if len(events) == 2:
        return _finish(f"{events[0]}, finally {events[1]}")
    return _finish(", then ".join(events[:-1]) + f", finally {events[-1]}")


def _before_first_pair(events):
    head = f"{events[0]} before {events[1]}"
    if len(events) == 2:
        return _finish(head)
    return _finish(head + ", then " + ", then ".join(events[2:]))


def _after_first_pair(events):
    head = f"{events[1]} after {events[0]}"
    if len(events) == 2:
        return _finish(head)
    return _finish(head + ", then " + ", then ".join(events[2:]))


def _ending_with(events):
    if len(events) == 2:
        return _finish(f"{events[0]}, ending with {events[1]}")
    return _finish(", then ".join(events[:-1]) + f", ending with {events[-1]}")


def _semicolon_chain(events):
    # "A; B; C." — _SEQ_SPLIT의 ';\s*' 분기.
    return _finish("; ".join(events))


def _afterwards_chain(events):
    # "A, afterwards B, afterwards C." — 정규식이 afterwards 뒤 공백을 요구.
    return _finish(", afterwards ".join(events))


def _subsequently_chain(events):
    # "A, subsequently B." — _SEQ_SPLIT의 subsequently 분기.
    return _finish(", subsequently ".join(events))


def _next_chain(events):
    # "A, next, B." — _SEQ_SPLIT의 next 분기는 next 뒤 콤마+공백을 요구한다.
    return _finish(", next, ".join(events))


def _leading_after(events):
    # 문두 "After A, B" (+ 3개 이상이면 ", then C ...") — _LEADING_AFTER가 처리.
    # A(=events[0])에 콤마가 있으면 non-greedy 그룹이 잘려 round-trip이 자동 기각한다
    # (별도 방어 불필요).
    head = f"After {events[0]}, {events[1]}"
    if len(events) == 2:
        return _finish(head)
    return _finish(head + ", then " + ", then ".join(events[2:]))


def _after_last_pair(events):
    # 마지막 인접쌍만 "Y after X"로 역전 — 시간 순서는 유지.
    #   4이벤트: "A, then B, then D after C."
    # len==2면 "B after A."로 _after_first_pair와 동일해지지만 별도 가드 없이
    # 그냥 렌더하고 caption_variants의 seen 중복 제거에 맡긴다.
    tail = f"{events[-1]} after {events[-2]}"
    prefix = events[:-2]
    if not prefix:
        return _finish(tail)
    return _finish(", then ".join(prefix) + ", then " + tail)


def _after_mid_pair(events):
    # 두 번째·세 번째 이벤트만 "third after second"로 역전 — 시간 순서 유지.
    #   4이벤트: "A, then C after B, then D."  /  3이벤트: "A, then C after B."
    if len(events) < 3:  # len>=3 가드
        return ""
    parts = [events[0], f"{events[2]} after {events[1]}"] + list(events[3:])
    return _finish(", then ".join(parts))


# 순서·인덱스 불변: 추론 캡션 TTA는 이 튜플의 위치(인덱스)로 변형을 재현하므로
# 신규 템플릿은 반드시 **끝에만** append 한다 (기존 6종의 순서를 바꾸지 말 것).
_TEMPLATES = (
    _then_chain,
    _followed_by_chain,
    _finally_chain,
    _before_first_pair,
    _after_first_pair,
    _ending_with,
    _semicolon_chain,
    _afterwards_chain,
    _subsequently_chain,
    _next_chain,
    _leading_after,
    _after_last_pair,
    _after_mid_pair,
)


# _mixed_chain 전용 연결어 풀 (모두 파서가 인식하는 순차 분할 지점).
_MIXED_CONNECTIVES = (
    ", then ",
    ", followed by ",
    "; ",
    ", subsequently ",
    ", afterwards ",
)


def _mixed_chain(events, rng):
    """위치별로 순차 연결어를 무작위로 골라 조인한 **학습 전용** 변형.

    결정성 계약 때문에 ``caption_variants``에는 넣지 않는다. 시간 순서는
    그대로 유지하되 연결어만 섞어 서술 표면을 다양화한다.
    """
    out = events[0]
    for ev in events[1:]:
        out += rng.choice(_MIXED_CONNECTIVES) + ev
    return _finish(out)


def caption_variants(caption):
    """Return valid temporal-connective rewrites of ``caption``.

    A variant is valid only when splitting it recovers exactly the same event
    list as the original caption.  This keeps the augmentation conservative:
    captions with one event, unusual punctuation, or parser-ambiguous text
    simply produce no variants.

    결정적 템플릿만 사용하므로 같은 입력에 대해 항상 같은 리스트를 돌려준다
    (추론 캡션 TTA 재현성 계약). 기존 6종이 유효하면 리스트 앞부분에 원래
    순서 그대로 놓인다.
    """
    events = split_events(caption)
    if len(events) < 2:
        return []

    variants = []
    seen = {caption.strip()}
    for render in _TEMPLATES:
        candidate = render(events)
        if candidate in seen:
            continue
        if split_events(candidate) == events:
            variants.append(candidate)
            seen.add(candidate)
    return variants


def augment_caption(caption, rng=None):
    """Sample one valid caption variant, or return the original caption.

    학습 시엔 결정적 변형 풀(``caption_variants``)에 더해, 이벤트가 2개 이상이면
    위치별 연결어를 무작위로 섞은 ``_mixed_chain`` 후보(학습 전용)를 한 개
    생성해 round-trip·중복 검증을 통과할 때만 후보 풀에 얹은 뒤 함께 뽑는다.
    """
    rng = rng or random
    events = split_events(caption)
    variants = caption_variants(caption)
    if len(events) >= 2:
        mixed = _mixed_chain(events, rng)
        if (
            mixed != caption.strip()
            and mixed not in variants
            and split_events(mixed) == events
        ):
            variants = variants + [mixed]
    if not variants:
        return caption
    return rng.choice(variants)
