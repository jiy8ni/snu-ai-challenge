"""Rule-based caption augmentation for temporal connectives.

This module is the rule-based caption augmenter.  LLM paraphrases are generated
offline by ``src.preprocess.llm_caption_augment`` and then consumed by the
training dataset through the ``caption_llm_variants`` field.
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


_TEMPLATES = (
    _then_chain,
    _followed_by_chain,
    _finally_chain,
    _before_first_pair,
    _after_first_pair,
    _ending_with,
)


def caption_variants(caption):
    """Return valid temporal-connective rewrites of ``caption``.

    A variant is valid only when splitting it recovers exactly the same event
    list as the original caption.  This keeps the augmentation conservative:
    captions with one event, unusual punctuation, or parser-ambiguous text
    simply produce no variants.
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
    """Sample one valid caption variant, or return the original caption."""
    variants = caption_variants(caption)
    if not variants:
        return caption
    rng = rng or random
    return rng.choice(variants)
