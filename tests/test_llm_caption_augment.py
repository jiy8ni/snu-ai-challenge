from src.preprocess.hard_cases import _score_case
from src.preprocess.llm_caption_augment import (
    hard_score,
    is_valid_variant,
    parse_variants,
    target_variant_count,
)


def test_hard_score_from_em_and_numeric_score():
    assert hard_score({"em": "True"}) == 0.0
    assert hard_score({"em": "False"}) == 1.0
    assert hard_score({"hard_score": "0.7"}) == 0.7


def test_target_variant_count_spends_more_on_hard_cases():
    easy = target_variant_count(0.0, base_variants=1, hard_extra_variants=3, max_variants=4)
    hard = target_variant_count(1.0, base_variants=1, hard_extra_variants=3, max_variants=4)
    assert easy == 1
    assert hard == 4


def test_parse_variants_accepts_json_wrapper():
    text = 'Sure:\n{"variants": ["A man opens a door, then walks inside."]}'
    assert parse_variants(text) == ["A man opens a door, then walks inside."]


def test_variant_filter_rejects_answers_and_low_overlap():
    original = "A man opens the door, then he walks inside."
    events = ["A man opens the door", "he walks inside"]
    assert is_valid_variant(
        "A man opens the door before walking inside.",
        original,
        events,
    )
    assert not is_valid_variant("Answer: [1, 2, 3, 4]", original, events)
    assert not is_valid_variant("A dog swims through a pool.", original, events)


def test_score_case_penalizes_wrong_identity_on_orderable():
    em, tau, score = _score_case([1, 2, 3, 4], [2, 1, 3, 4], no_ordering=False)
    assert not em
    assert tau < 1.0
    assert score >= 0.75
