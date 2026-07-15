import random

from src.preprocess.caption_augment import (
    _TEMPLATES,
    _after_last_pair,
    _after_mid_pair,
    _afterwards_chain,
    _leading_after,
    _mixed_chain,
    _next_chain,
    _semicolon_chain,
    _subsequently_chain,
    _then_chain,
    augment_caption,
    caption_variants,
)
from src.preprocess.caption_events import split_events


def test_caption_variants_roundtrip_temporal_events():
    caption = "A man opens the door, then he walks inside, finally he sits down."
    events = split_events(caption)

    variants = caption_variants(caption)

    assert variants
    assert any("after" in v for v in variants)
    assert any("before" in v for v in variants)
    for variant in variants:
        assert variant != caption
        assert split_events(variant) == events


def test_caption_variants_leave_single_event_unaugmented():
    caption = "A skier moves forward as the camera zooms in."
    assert caption_variants(caption) == []
    assert augment_caption(caption, random.Random(0)) == caption


def test_augment_caption_is_seeded_and_valid():
    caption = "A girl starts dancing, followed by a spin, ending with a bow."

    augmented = augment_caption(caption, random.Random(7))

    # augment_caption은 결정적 변형(caption_variants) 또는 학습 전용 _mixed_chain을
    # 반환할 수 있다. 어느 쪽이든 round-trip(이벤트 순서 보존)은 반드시 성립한다.
    assert split_events(augmented) == split_events(caption)
    # 시드 고정 재현성: 같은 시드는 항상 같은 출력.
    assert augment_caption(caption, random.Random(7)) == augmented


# --- 신규 템플릿 검증 ---------------------------------------------------------

_NEW_TEMPLATES = (
    _semicolon_chain,
    _afterwards_chain,
    _subsequently_chain,
    _next_chain,
    _leading_after,
    _after_last_pair,
    _after_mid_pair,
)


def test_new_templates_roundtrip_and_reachable():
    # 이벤트 절은 콤마·연결어가 없어야 파서가 깔끔히 복원한다.
    event_lists = (
        ["a man waves", "he turns around"],
        ["a man waves", "he turns around", "he walks away"],
        ["a man waves", "he turns around", "he walks away", "he sits down"],
    )
    for events in event_lists:
        caption = _then_chain(events)  # 결정적·round-trip 보장 기준 캡션
        assert split_events(caption) == events
        variants = caption_variants(caption)
        for fn in _NEW_TEMPLATES:
            candidate = fn(events)
            if not candidate:
                # _after_mid_pair는 len<3에서 의도적으로 빈 문자열을 낸다.
                assert fn is _after_mid_pair and len(events) < 3
                continue
            # round-trip 성립 (파서가 원래 이벤트 순서를 정확히 복원)
            assert split_events(candidate) == events
            # caption_variants에 포함되거나, 포함 안 되면 그 이유가 seen 중복이다.
            idx = _TEMPLATES.index(fn)
            deduped = candidate == caption.strip() or any(
                _TEMPLATES[j](events) == candidate for j in range(idx)
            )
            assert candidate in variants or deduped


def test_caption_variants_deterministic_and_index_stable():
    caption = "a chef chops onions, then he heats the pan, finally he plates the dish."

    first = caption_variants(caption)
    second = caption_variants(caption)
    assert first == second  # 같은 입력 두 번 호출 결과 동일 (결정성)

    events = split_events(caption)
    # 기존 6종만 돌린 결과가 전체 리스트 앞부분에 순서 그대로 놓여야 한다 (인덱스 안정성).
    prefix = _variants_for(events, caption, _TEMPLATES[:6])
    assert first[: len(prefix)] == prefix
    assert len(first) > len(prefix)  # 신규 템플릿이 실제로 변형을 더한다


def _variants_for(events, caption, templates):
    """caption_variants 로직을 템플릿 부분집합으로 재현하는 테스트 헬퍼."""
    variants = []
    seen = {caption.strip()}
    for render in templates:
        candidate = render(events)
        if candidate in seen:
            continue
        if split_events(candidate) == events:
            variants.append(candidate)
            seen.add(candidate)
    return variants


def test_mixed_chain_reproducible_and_valid():
    events = ["a dog runs", "it jumps a fence", "it catches a ball", "it returns"]

    first = _mixed_chain(events, random.Random(11))
    second = _mixed_chain(events, random.Random(11))
    assert first == second  # 같은 시드 재현성

    assert split_events(first) == events  # round-trip 성립 케이스 존재


def test_mixed_chain_excluded_from_caption_variants():
    # 결정성 계약: mixed_chain은 결정적 템플릿 튜플에 포함되지 않는다.
    assert _mixed_chain not in _TEMPLATES

    # 결정적 변형과 표면이 다른 mixed 조합은 caption_variants에 절대 나타나지 않는다.
    caption = "a dog runs, then it jumps a fence, then it catches a ball."
    events = split_events(caption)
    variants = caption_variants(caption)
    novel = [
        m
        for m in (_mixed_chain(events, random.Random(s)) for s in range(50))
        if m not in variants
    ]
    assert novel  # 최소 한 개는 결정적 변형과 겹치지 않는 표면이어야 유의미
    for mixed in novel:
        assert mixed not in caption_variants(caption)


def test_augment_caption_single_event_returns_original():
    caption = "A lone tree stands in a field as the wind blows."
    assert len(split_events(caption)) == 1
    assert augment_caption(caption, random.Random(5)) == caption
