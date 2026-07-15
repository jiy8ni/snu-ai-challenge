"""Generate LLM caption paraphrases with a hard-case-aware budget.

Example:
  export OPENAI_API_KEY=...
  python -m src.preprocess.llm_caption_augment \
    --input /workspace/snuai/outputs/sft_train.jsonl \
    --hard-cases /workspace/snuai/outputs/hard_train_cases.csv \
    --out /workspace/snuai/outputs/sft_train_llm_aug.jsonl \
    --max-cost-krw 25000

The script writes the original records plus:
  - caption_llm_variants: list[str]
  - caption_aug_repeats: total copies requested by VLSFTDataset
  - caption_aug_prob: per-record caption augmentation probability

It uses only stdlib HTTP so RunPod does not need an extra OpenAI package.

품질·예산 제어(기본 on):
  - 이벤트 순서 검증(events_in_order): 변형이 원본 이벤트를 같은 순서로
    서술하지 않으면 기각 — 순열 라벨과 모순되는 캡션 학습 차단.
    끄려면 --no-order-check, 임계는 --min-event-overlap.
  - 자카드 근사중복 필터: 수락분·원본과 --dup-jaccard(0.85) 이상 유사한
    후보는 스킵해 변형 슬롯 낭비 방지.
  - 부스팅식 예산: hard_score 내림차순으로 API를 지출해 예산 소진 시에도
    hard 케이스가 먼저 채워진다(--no-priority-order로 해제).
    --easy-base-variants 0이면 easy 콜을 생략하고 hard에 전량 집중.
  - --revalidate-only: 기존 증강 jsonl의 variants를 API 호출 없이
    신규 검증기로 재필터링(이미 지출한 예산의 산출물 정화).
  - API 호출 로그가 <out>.calls.jsonl에 보존된다(규정: 비용·프롬프트·로그).
  - 주의: 여기서 기록하는 per-record caption_aug_prob는 vl_dataset의 config
    전역값을 덮어쓴다. easy 레코드까지 base(0.3)로 하향시키고 싶지 않으면
    --omit-easy-fields로 score 0 레코드의 필드 기록을 생략할 것.
"""

import argparse
import csv
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from copy import copy

from src.preprocess.caption_events import split_events
from src.preprocess.hard_weights import clamp01, prob_from_score, repeats_from_score

DEFAULT_MODEL = "gpt-5.4-nano"
DEFAULT_INPUT_USD_PER_1M = 0.10
DEFAULT_OUTPUT_USD_PER_1M = 0.625
DEFAULT_KRW_PER_USD = 1400.0

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from", "he", "her",
    "his", "in", "into", "is", "it", "its", "of", "on", "or", "she", "the",
    "their", "then", "they", "to", "with",
}


def load_jsonl(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(path, records):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def _read_table(path):
    if not path:
        return []
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jsonl", ".ndjson"):
        return load_jsonl(path)
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def load_hard_cases(path):
    rows = _read_table(path)
    return {str(row.get("Id") or row.get("id")): row for row in rows if row.get("Id") or row.get("id")}


def normalise_variants(value):
    if not value:
        return []
    if isinstance(value, list):
        return [str(v).strip() for v in value if str(v).strip()]
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            parsed = json.loads(text)
        except json.JSONDecodeError:
            parsed = [part.strip() for part in text.split("|||")]
        if isinstance(parsed, list):
            return [str(v).strip() for v in parsed if str(v).strip()]
        return [text]
    return [str(value).strip()]


def _truthy(value):
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


_clamp01 = clamp01  # 하위호환 별칭 (수식 본체는 hard_weights.py)


def hard_score(row):
    if not row:
        return 0.0
    if row.get("hard_score") not in (None, ""):
        return _clamp01(row["hard_score"])
    if row.get("em") not in (None, ""):
        return 0.0 if _truthy(row["em"]) else 1.0
    if row.get("correct") not in (None, ""):
        return 0.0 if _truthy(row["correct"]) else 1.0
    return 0.0


def target_variant_count(score, base_variants, hard_extra_variants, max_variants):
    n = int(base_variants) + int(math.ceil(score * hard_extra_variants))
    return max(0, min(int(max_variants), n))


def annotate_hard_fields(
    rec,
    hard_row,
    score,
    max_repeats,
    base_caption_aug_prob,
    hard_caption_aug_prob,
    omit_easy_fields=False,
):
    """레코드에 caption_aug_repeats/prob/hard_score를 기록.

    주의: 여기서 기록한 per-record ``caption_aug_prob``는 학습 시
    vl_dataset의 config 전역값(예: 0.45)을 **덮어쓴다**. easy 레코드에
    base(0.3)를 기록하면 전역값을 하향시키는 셈이라, ``omit_easy_fields``가
    켜지면 score 0(명시 override 없음) 레코드는 필드를 아예 기록하지 않아
    config 폴백이 적용되게 한다.
    """
    out = copy(rec)
    explicit_repeats = bool(hard_row) and hard_row.get("caption_aug_repeats") not in (None, "")
    explicit_prob = bool(hard_row) and hard_row.get("caption_aug_prob") not in (None, "")
    if omit_easy_fields and score <= 0.0 and not explicit_repeats and not explicit_prob:
        return out

    if explicit_repeats:
        repeats = int(float(hard_row["caption_aug_repeats"]))
    else:
        repeats = repeats_from_score(score, max_repeats)
    out["caption_aug_repeats"] = max(1, min(int(max_repeats), repeats))

    if explicit_prob:
        prob = float(hard_row["caption_aug_prob"])
    else:
        prob = prob_from_score(score, base_caption_aug_prob, hard_caption_aug_prob)
    out["caption_aug_prob"] = round(_clamp01(prob), 6)
    out["hard_score"] = round(score, 6)
    return out


def estimate_tokens(text):
    return max(1, int(math.ceil(len(text) / 4)))


def estimate_cost_usd(input_tokens, output_tokens, input_usd_per_1m, output_usd_per_1m):
    return (input_tokens * input_usd_per_1m + output_tokens * output_usd_per_1m) / 1_000_000.0


def _prompt(caption, events, n):
    event_text = "\n".join(f"{i + 1}. {ev}" for i, ev in enumerate(events))
    system = (
        "You create concise, safe English caption paraphrases for vision-language "
        "training. Return only JSON."
    )
    user = (
        f"Original caption:\n{caption}\n\n"
        f"Temporal events that must stay in the same order:\n{event_text}\n\n"
        f"Write {n} paraphrases as one-sentence captions. Preserve all concrete "
        "people, objects, actions, and the chronological order. Do not add visual "
        "details, frame numbers, answers, labels, or explanations. Return exactly "
        '{"variants": ["...", "..."]}.'
    )
    return system, user


def _extract_content(message):
    content = message.get("content", "")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(part.get("text", "") if isinstance(part, dict) else str(part) for part in content)
    return str(content)


def parse_variants(text):
    text = text.strip()
    candidates = [text]
    m = re.search(r"\{.*\}", text, flags=re.S)
    if m:
        candidates.append(m.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            variants = parsed.get("variants", [])
        else:
            variants = parsed
        if isinstance(variants, list):
            return [str(v).strip() for v in variants if str(v).strip()]
    return []


def _content_tokens(text):
    tokens = re.findall(r"[A-Za-z0-9]+", text.lower())
    return {t for t in tokens if len(t) > 2 and t not in STOPWORDS}


def _jaccard(tokens_a, tokens_b):
    if not tokens_a and not tokens_b:
        return 1.0
    return len(tokens_a & tokens_b) / max(1, len(tokens_a | tokens_b))


def events_in_order(candidate, events, min_event_overlap=0.34):
    """후보 캡션이 원본 이벤트를 같은 시간 순서로 서술하는지 검사.

    LLM 패러프레이즈는 이벤트 문구 자체를 바꾸므로 규칙 기반처럼
    split_events 완전 라운드트립을 강제할 수 없다. 대신 각 원본 이벤트를
    후보 이벤트 중 콘텐츠 토큰 겹침이 최대인 것에 매핑하고, 그 매핑
    인덱스가 엄격 단조증가인지 본다 — 순서 뒤집힘(비단조)과 이벤트
    병합(두 원본이 같은 후보에 매핑) 모두 여기서 걸린다. 최대 겹침이
    min_event_overlap 미만이면 이벤트 누락으로 기각.
    """
    if len(events) < 2:
        return True
    cand_events = split_events(candidate)
    if not cand_events:
        return False
    cand_tokens = [_content_tokens(ev) for ev in cand_events]
    prev_j = -1
    for ev in events:
        ev_tokens = _content_tokens(ev)
        if not ev_tokens:  # 전부 stopword/2자 이하 — 매핑 근거가 없어 스킵
            continue
        best_j, best_overlap = -1, -1.0
        for j, ct in enumerate(cand_tokens):
            overlap = len(ev_tokens & ct) / len(ev_tokens)
            if overlap > best_overlap:  # 동률은 최소 j 유지
                best_j, best_overlap = j, overlap
        if best_overlap < min_event_overlap:
            return False
        if best_j <= prev_j:
            return False
        prev_j = best_j
    return True


def is_valid_variant(
    candidate,
    original,
    events,
    min_token_overlap=0.45,
    require_event_count=False,
    check_event_order=True,
    min_event_overlap=0.34,
):
    text = candidate.strip()
    if not text or text.lower() == original.strip().lower():
        return False
    if len(text) > max(240, int(len(original) * 1.8)):
        return False
    if re.search(r"\b(frame|answer|conclusion|orderable|unorderable)\b", text, flags=re.I):
        return False

    original_tokens = _content_tokens(original)
    if original_tokens:
        overlap = len(original_tokens & _content_tokens(text)) / len(original_tokens)
        if overlap < min_token_overlap:
            return False

    if require_event_count:
        parsed = split_events(text)
        if len(parsed) != len(events):
            return False

    if check_event_order and not events_in_order(text, events, min_event_overlap):
        return False
    return True


def accept_variants(candidates, caption, events, args, existing=None):
    """유효성(순서 검증 포함)+자카드 근사중복을 통과한 신규 후보만 순서대로 수집.

    existing(이미 수락된 변형)과 원본 캡션 모두를 중복 기준으로 삼는다.
    반환은 신규 수락분만 — 호출측이 existing 뒤에 이어붙인다.
    """
    kept_tokens = [_content_tokens(caption)]
    seen = {caption.strip()}
    for v in existing or []:
        seen.add(v)
        kept_tokens.append(_content_tokens(v))

    accepted = []
    for candidate in candidates:
        candidate = candidate.strip()
        if not candidate or candidate in seen:
            continue
        if not is_valid_variant(
            candidate,
            caption,
            events,
            min_token_overlap=args.min_token_overlap,
            require_event_count=args.require_event_count,
            check_event_order=args.order_check,
            min_event_overlap=args.min_event_overlap,
        ):
            continue
        tokens = _content_tokens(candidate)
        if any(_jaccard(tokens, kt) >= args.dup_jaccard for kt in kept_tokens):
            continue
        accepted.append(candidate)
        seen.add(candidate)
        kept_tokens.append(tokens)
    return accepted


def _api_url(url):
    url = url.rstrip("/")
    if url.endswith("/chat/completions"):
        return url
    return url + "/chat/completions"


def call_chat_completion(args, system, user):
    api_key = os.environ.get(args.api_key_env, "")
    url = _api_url(args.api_url)
    if not api_key and not re.search(r"//(localhost|127\.0\.0\.1|0\.0\.0\.0|\[::1\])", url):
        raise RuntimeError(f"{args.api_key_env} is not set")

    payload = {
        "model": args.model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": args.temperature,
        "max_tokens": args.max_output_tokens,
    }
    if args.json_mode:
        payload["response_format"] = {"type": "json_object"}

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=args.timeout) as resp:
        body = json.loads(resp.read().decode("utf-8"))
    message = body["choices"][0]["message"]
    return _extract_content(message), body.get("usage", {})


def generate_variants(args, caption, events, n, existing=None):
    system, user = _prompt(caption, events, n)
    for attempt in range(args.retries):
        try:
            text, usage = call_chat_completion(args, system, user)
            raw = parse_variants(text)
            variants = accept_variants(raw, caption, events, args, existing=existing)
            return variants, usage, system + "\n" + user, text
        except (urllib.error.URLError, TimeoutError, RuntimeError, KeyError, IndexError, json.JSONDecodeError) as exc:
            if attempt + 1 >= args.retries:
                raise
            wait = args.retry_sleep * (2 ** attempt)
            print(f"[retry {attempt + 1}/{args.retries}] {exc}; sleeping {wait:.1f}s")
            time.sleep(wait)
    return [], {}, system + "\n" + user, ""


def revalidate_records(records, args):
    """API 호출 없이 기존 caption_llm_variants를 신규 검증기로 재필터링.

    팀원이 이미 지출한 예산의 산출물을 무료로 정화하는 경로. 제거된
    변형 수와 기각률을 통계로 돌려준다 (기각률이 높으면
    --min-event-overlap 하향을 검토할 것).
    """
    out = []
    before = after = touched = 0
    for rec in records:
        rec_out = copy(rec)
        existing = normalise_variants(rec_out.get(args.output_field, []))
        if existing:
            caption = rec.get("caption", "")
            events = rec.get("events") or split_events(caption) or [caption]
            kept = accept_variants(existing, caption, events, args)
            before += len(existing)
            after += len(kept)
            if len(kept) != len(existing):
                touched += 1
            rec_out[args.output_field] = kept
        out.append(rec_out)
    stats = {
        "records": len(records),
        "variants_before": before,
        "variants_after": after,
        "removed": before - after,
        "reject_rate": round((before - after) / before, 4) if before else 0.0,
        "records_touched": touched,
    }
    return out, stats


def _append_call_log(args, rec_id, prompt, response, usage, cost_usd):
    """규정(docs/rules.md §외부 API)의 비용·프롬프트·생성 로그 보존 의무 이행."""
    if not getattr(args, "log_file", None):
        return
    os.makedirs(os.path.dirname(args.log_file) or ".", exist_ok=True)
    entry = {
        "Id": rec_id,
        "model": args.model,
        "prompt": prompt,
        "response": response,
        "usage": usage,
        "cost_usd": round(cost_usd, 8),
    }
    with open(args.log_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")


def augment_records(records, hard_by_id, args):
    n_records = len(records)
    hard_rows = [hard_by_id.get(str(rec.get("Id", "")), {}) for rec in records]
    scores = [hard_score(row) for row in hard_rows]

    # 부스팅식 지출: hard_score 내림차순(동률은 원본 순서)으로 처리해 예산이
    # 소진되더라도 hard 케이스가 먼저 채워진다. 출력 순서는 원본을 유지한다.
    if args.no_priority_order:
        order = list(range(n_records))
    else:
        order = sorted(range(n_records), key=lambda i: (-scores[i], i))

    out = [None] * n_records
    spent_usd = 0.0
    called = skipped_budget = 0
    planned_calls = planned_variants = 0
    limit = args.max_records if args.max_records and args.max_records > 0 else None

    for step, idx in enumerate(order):
        rec = records[idx]
        hard_row = hard_rows[idx]
        score = scores[idx]
        rec_out = annotate_hard_fields(
            rec,
            hard_row,
            score,
            args.max_repeats,
            args.base_caption_aug_prob,
            args.hard_caption_aug_prob,
            omit_easy_fields=args.omit_easy_fields,
        )

        existing = normalise_variants(rec_out.get(args.output_field, []))
        base_variants = args.base_variants
        if args.easy_base_variants is not None and score <= args.easy_score_threshold:
            base_variants = args.easy_base_variants
        target_n = target_variant_count(
            score, base_variants, args.hard_extra_variants, args.max_variants_per_record
        )
        needed = max(0, target_n - len(existing))
        if limit is not None and called >= limit:
            needed = 0

        if needed > 0:
            planned_calls += 1
            planned_variants += needed
            events = rec.get("events") or split_events(rec["caption"]) or [rec["caption"]]
            system, user = _prompt(rec["caption"], events, needed)
            est_in = estimate_tokens(system + "\n" + user)
            est_out = args.max_output_tokens
            est_usd = estimate_cost_usd(
                est_in,
                est_out,
                args.price_input_usd_per_1m,
                args.price_output_usd_per_1m,
            )
            if (spent_usd + est_usd) * args.krw_per_usd > args.max_cost_krw:
                skipped_budget += 1
            elif not args.dry_run:
                variants, usage, prompt_text, response_text = generate_variants(
                    args, rec["caption"], events, needed, existing=existing
                )
                existing.extend(variants)
                called += 1
                input_tokens = int(usage.get("prompt_tokens") or est_in)
                output_tokens = int(usage.get("completion_tokens") or estimate_tokens(" ".join(variants)))
                call_usd = estimate_cost_usd(
                    input_tokens,
                    output_tokens,
                    args.price_input_usd_per_1m,
                    args.price_output_usd_per_1m,
                )
                spent_usd += call_usd
                _append_call_log(args, rec.get("Id"), prompt_text, response_text, usage, call_usd)
                if args.sleep:
                    time.sleep(args.sleep)
            else:
                spent_usd += est_usd

        rec_out[args.output_field] = existing[: args.max_variants_per_record]
        out[idx] = rec_out
        if args.save_every and not args.dry_run and (step + 1) % args.save_every == 0:
            # 미처리 인덱스는 원본 레코드로 채워 재개 시 손실이 없게 한다.
            write_jsonl(args.out, [out[i] if out[i] is not None else records[i] for i in range(n_records)])

    stats = {
        "records": n_records,
        "planned_calls": planned_calls,
        "planned_variants": planned_variants,
        "actual_calls": called,
        "skipped_budget": skipped_budget,
        "estimated_or_actual_cost_krw": round(spent_usd * args.krw_per_usd, 2),
    }
    return out, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="input SFT JSONL")
    ap.add_argument("--out", required=True, help="output augmented SFT JSONL")
    ap.add_argument("--hard-cases", default=None, help="CSV/JSONL with Id, hard_score, repeats, prob")
    ap.add_argument("--output-field", default="caption_llm_variants")
    ap.add_argument("--base-variants", type=int, default=1)
    ap.add_argument("--hard-extra-variants", type=int, default=3)
    ap.add_argument("--max-variants-per-record", type=int, default=4)
    ap.add_argument("--max-repeats", type=int, default=5)
    ap.add_argument("--base-caption-aug-prob", type=float, default=0.3)
    ap.add_argument("--hard-caption-aug-prob", type=float, default=0.9)
    ap.add_argument("--min-token-overlap", type=float, default=0.45)
    ap.add_argument("--require-event-count", action="store_true")
    ap.add_argument("--no-order-check", dest="order_check", action="store_false",
                    help="이벤트 순서 보존 검증(기본 on)을 끈다 — 구 동작 복원용")
    ap.set_defaults(order_check=True)
    ap.add_argument("--min-event-overlap", type=float, default=0.34,
                    help="이벤트별 매핑 최소 토큰 겹침 (미만이면 이벤트 누락으로 기각)")
    ap.add_argument("--dup-jaccard", type=float, default=0.85,
                    help="수락된 변형·원본과의 자카드 유사도가 이 값 이상이면 근사중복으로 스킵 (>1이면 사실상 off)")
    ap.add_argument("--revalidate-only", action="store_true",
                    help="API 호출 없이 기존 caption_llm_variants를 신규 검증기로 재필터링만 수행")
    ap.add_argument("--no-priority-order", action="store_true",
                    help="hard_score 내림차순 지출(기본)을 끄고 파일 순서로 처리")
    ap.add_argument("--easy-base-variants", type=int, default=None,
                    help="score<=--easy-score-threshold 레코드의 base 변형 수 오버라이드 (0이면 easy 콜 생략)")
    ap.add_argument("--easy-score-threshold", type=float, default=0.0)
    ap.add_argument("--omit-easy-fields", action="store_true",
                    help="score 0 레코드에 caption_aug_prob/repeats 필드를 기록하지 않음 (config 전역값 폴백 유도)")
    ap.add_argument("--log-file", default=None,
                    help="API 호출 로그 jsonl (기본: <out>.calls.jsonl — 규정상 비용·프롬프트·응답 보존)")

    ap.add_argument("--api-url", default=os.environ.get("SNUAI_LLM_AUG_API_URL", "https://api.openai.com/v1"))
    ap.add_argument("--api-key-env", default="OPENAI_API_KEY")
    ap.add_argument("--model", default=os.environ.get("SNUAI_LLM_AUG_MODEL", DEFAULT_MODEL))
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--max-output-tokens", type=int, default=350)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--retries", type=int, default=3)
    ap.add_argument("--retry-sleep", type=float, default=2.0)
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--no-json-mode", dest="json_mode", action="store_false")
    ap.set_defaults(json_mode=True)

    ap.add_argument("--max-cost-krw", type=float, default=25000.0)
    ap.add_argument("--krw-per-usd", type=float, default=DEFAULT_KRW_PER_USD)
    ap.add_argument("--price-input-usd-per-1m", type=float, default=DEFAULT_INPUT_USD_PER_1M)
    ap.add_argument("--price-output-usd-per-1m", type=float, default=DEFAULT_OUTPUT_USD_PER_1M)
    ap.add_argument("--max-records", type=int, default=None, help="API-call limit for smoke tests")
    ap.add_argument("--save-every", type=int, default=25)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    records = load_jsonl(args.input)

    if args.revalidate_only:  # 기존 산출물 정화 — API 호출·예산 소비 없음
        out, stats = revalidate_records(records, args)
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        write_jsonl(args.out, out)
        print(f"saved: {args.out}")
        return

    if args.log_file is None and not args.dry_run:
        args.log_file = args.out + ".calls.jsonl"

    hard_by_id = load_hard_cases(args.hard_cases)
    out, stats = augment_records(records, hard_by_id, args)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    if not args.dry_run:
        write_jsonl(args.out, out)
        print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
