import argparse

import pytest

from src.preprocess.hard_cases import _score_case
from src.preprocess.hard_weights import prob_from_score, repeats_from_score
from src.preprocess import llm_caption_augment as lca
from src.preprocess.llm_caption_augment import (
    accept_variants,
    annotate_hard_fields,
    augment_records,
    events_in_order,
    hard_score,
    is_valid_variant,
    parse_variants,
    revalidate_records,
    target_variant_count,
)


def _args(**overrides):
    """CLI argparse 기본값과 동일한 네임스페이스 (테스트용)."""
    ns = argparse.Namespace(
        out="unused.jsonl",
        output_field="caption_llm_variants",
        base_variants=1,
        hard_extra_variants=3,
        max_variants_per_record=4,
        max_repeats=5,
        base_caption_aug_prob=0.3,
        hard_caption_aug_prob=0.9,
        min_token_overlap=0.45,
        require_event_count=False,
        order_check=True,
        min_event_overlap=0.34,
        dup_jaccard=0.85,
        no_priority_order=False,
        easy_base_variants=None,
        easy_score_threshold=0.0,
        omit_easy_fields=False,
        log_file=None,
        model="test-model",
        temperature=0.7,
        max_output_tokens=350,
        max_cost_krw=25000.0,
        krw_per_usd=1400.0,
        price_input_usd_per_1m=0.10,
        price_output_usd_per_1m=0.625,
        max_records=None,
        save_every=0,
        dry_run=True,
        sleep=0,
        retries=1,
        retry_sleep=0,
    )
    for key, value in overrides.items():
        setattr(ns, key, value)
    return ns


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


# --- (a) 이벤트 순서 검증 -----------------------------------------------------

_ORIG = "A man opens the door, then he walks inside."
_EVENTS = ["A man opens the door", "he walks inside"]


def test_order_check_rejects_flipped_events():
    flipped = "He walks inside, then a man opens the door."
    assert not events_in_order(flipped, _EVENTS)
    assert not is_valid_variant(flipped, _ORIG, _EVENTS)


def test_order_check_rejects_merged_events():
    # 두 이벤트를 연결어 없는 한 절로 병합 — 두 원본이 같은 후보에 매핑돼 기각.
    merged = "A man opens the door and walks inside."
    assert not events_in_order(merged, _EVENTS)
    assert not is_valid_variant(merged, _ORIG, _EVENTS)


def test_order_check_rejects_dropped_event():
    dropped = "A man opens the door slowly and carefully with both hands."
    assert not events_in_order(dropped, _EVENTS)


def test_order_check_accepts_wording_change_same_order():
    paraphrase = "A man pushes the door open, then he steps inside."
    assert events_in_order(paraphrase, _EVENTS)
    assert is_valid_variant(paraphrase, _ORIG, _EVENTS)


def test_order_check_passes_single_event():
    assert events_in_order("A man strolls inside.", ["A man walks inside"])
    assert is_valid_variant("A man strolls inside now.", "A man walks inside.", ["A man walks inside"])


def test_order_check_can_be_disabled():
    flipped = "He walks inside, then a man opens the door."
    assert is_valid_variant(flipped, _ORIG, _EVENTS, check_event_order=False)


def test_revalidate_records_purges_contaminated_variants():
    good = "A man pushes the door open, then he steps inside."
    flipped = "He walks inside, then a man opens the door."
    records = [{"Id": "r1", "caption": _ORIG, "caption_llm_variants": [good, flipped]}]

    out, stats = revalidate_records(records, _args())

    assert out[0]["caption_llm_variants"] == [good]
    assert stats["variants_before"] == 2
    assert stats["variants_after"] == 1
    assert stats["removed"] == 1
    assert stats["records_touched"] == 1


# --- (c) 자카드 근사중복 필터 --------------------------------------------------


def test_accept_variants_skips_near_duplicates():
    v1 = "A man pushes the door open, then he steps inside."
    near_dup = "A man pushes the door open, then he steps inside now."
    accepted = accept_variants([v1, near_dup], _ORIG, _EVENTS, _args())
    assert accepted == [v1]


def test_accept_variants_keeps_distinct_surfaces():
    v1 = "A man pushes the door open, then he steps inside."
    v2 = "A man opens the door before walking inside."
    accepted = accept_variants([v1, v2], _ORIG, _EVENTS, _args())
    assert accepted == [v1, v2]


def test_accept_variants_dedups_against_existing():
    v1 = "A man pushes the door open, then he steps inside."
    near_dup = "A man pushes the door open, then he steps inside now."
    accepted = accept_variants([near_dup], _ORIG, _EVENTS, _args(), existing=[v1])
    assert accepted == []


# --- (b) 예산 hard 우선순위 ----------------------------------------------------

_EASY_CAP = "A dog runs across the yard, then it fetches a ball."
_HARD_CAP = "A chef chops onions, then he heats the pan."


def _two_records():
    records = [
        {"Id": "e1", "caption": _EASY_CAP},
        {"Id": "h1", "caption": _HARD_CAP},
    ]
    hard_by_id = {"h1": {"em": "False"}}  # score 1.0; e1은 테이블에 없어 score 0.0
    return records, hard_by_id


def _fake_generate(calls):
    def fake(args, caption, events, n, existing=None):
        calls.append(caption)
        return ([f"variant of {caption}"], {"prompt_tokens": 10, "completion_tokens": 10}, "p", "r")
    return fake


def test_priority_order_spends_on_hard_first(monkeypatch, tmp_path):
    records, hard_by_id = _two_records()
    calls = []
    monkeypatch.setattr(lca, "generate_variants", _fake_generate(calls))
    args = _args(dry_run=False, out=str(tmp_path / "out.jsonl"))

    out, stats = augment_records(records, hard_by_id, args)

    assert calls == [_HARD_CAP, _EASY_CAP]  # hard 먼저 지출
    assert [r["Id"] for r in out] == ["e1", "h1"]  # 출력은 원본 순서 유지
    assert stats["actual_calls"] == 2


def test_no_priority_order_restores_file_order(monkeypatch, tmp_path):
    records, hard_by_id = _two_records()
    calls = []
    monkeypatch.setattr(lca, "generate_variants", _fake_generate(calls))
    args = _args(dry_run=False, no_priority_order=True, out=str(tmp_path / "out.jsonl"))

    augment_records(records, hard_by_id, args)

    assert calls == [_EASY_CAP, _HARD_CAP]


def test_easy_base_variants_zero_skips_easy_calls(monkeypatch, tmp_path):
    records, hard_by_id = _two_records()
    calls = []
    monkeypatch.setattr(lca, "generate_variants", _fake_generate(calls))
    args = _args(dry_run=False, easy_base_variants=0, out=str(tmp_path / "out.jsonl"))

    out, stats = augment_records(records, hard_by_id, args)

    assert calls == [_HARD_CAP]  # easy 콜 생략, hard에 전량 집중
    assert stats["planned_calls"] == 1


def test_budget_exhaustion_reports_skips():
    records, hard_by_id = _two_records()
    args = _args(max_cost_krw=0.0)  # dry-run + 예산 0 — 전부 스킵

    _, stats = augment_records(records, hard_by_id, args)

    assert stats["skipped_budget"] == 2
    assert stats["actual_calls"] == 0


# --- (f) 수식 단일화 + omit-easy + 호출 로그 -----------------------------------


def test_hard_weight_formulas_boundaries():
    assert repeats_from_score(0.0, 5) == 1
    assert repeats_from_score(1.0, 5) == 5
    assert repeats_from_score(0.5, 5) == 3
    assert repeats_from_score(2.0, 5) == 5  # clamp
    assert prob_from_score(0.0, 0.3, 0.9) == pytest.approx(0.3)
    assert prob_from_score(1.0, 0.3, 0.9) == pytest.approx(0.9)


def test_hard_cases_and_llm_augment_share_formula():
    # hard_cases.py의 구식 --max-extra-repeats N == max_repeats N+1 매핑 검증.
    for score in (0.0, 0.25, 0.5, 0.75, 1.0):
        rec = annotate_hard_fields({}, {}, score, 5, 0.3, 0.9)
        assert rec["caption_aug_repeats"] == repeats_from_score(score, 4 + 1)
        assert rec["caption_aug_prob"] == round(prob_from_score(score, 0.3, 0.9), 6)


def test_omit_easy_fields_leaves_record_clean():
    easy = annotate_hard_fields({"Id": "x"}, {}, 0.0, 5, 0.3, 0.9, omit_easy_fields=True)
    assert "caption_aug_prob" not in easy
    assert "caption_aug_repeats" not in easy

    hard = annotate_hard_fields({"Id": "y"}, {}, 1.0, 5, 0.3, 0.9, omit_easy_fields=True)
    assert hard["caption_aug_prob"] == 0.9
    assert hard["caption_aug_repeats"] == 5


def test_call_log_written(monkeypatch, tmp_path):
    import json as _json

    records, hard_by_id = _two_records()
    calls = []
    monkeypatch.setattr(lca, "generate_variants", _fake_generate(calls))
    log_path = tmp_path / "calls.jsonl"
    args = _args(dry_run=False, out=str(tmp_path / "out.jsonl"), log_file=str(log_path))

    augment_records(records, hard_by_id, args)

    entries = [_json.loads(l) for l in log_path.read_text(encoding="utf-8").splitlines()]
    assert len(entries) == 2
    assert {e["Id"] for e in entries} == {"e1", "h1"}
    assert all(e["prompt"] and e["response"] and "cost_usd" in e for e in entries)
