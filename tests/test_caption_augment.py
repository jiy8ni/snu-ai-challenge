import random

from src.preprocess.caption_augment import augment_caption, caption_variants
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
    rng = random.Random(7)

    augmented = augment_caption(caption, rng)

    assert augmented in caption_variants(caption)
    assert split_events(augmented) == split_events(caption)
