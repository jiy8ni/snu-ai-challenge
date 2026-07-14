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


def _clamp01(value):
    return min(1.0, max(0.0, float(value)))


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
):
    out = copy(rec)
    if hard_row and hard_row.get("caption_aug_repeats") not in (None, ""):
        repeats = int(float(hard_row["caption_aug_repeats"]))
    else:
        repeats = 1 + int(math.ceil(score * (max_repeats - 1)))
    out["caption_aug_repeats"] = max(1, min(int(max_repeats), repeats))

    if hard_row and hard_row.get("caption_aug_prob") not in (None, ""):
        prob = float(hard_row["caption_aug_prob"])
    else:
        prob = base_caption_aug_prob + (hard_caption_aug_prob - base_caption_aug_prob) * score
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


def is_valid_variant(candidate, original, events, min_token_overlap=0.45, require_event_count=False):
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
    return True


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


def generate_variants(args, caption, events, n):
    system, user = _prompt(caption, events, n)
    for attempt in range(args.retries):
        try:
            text, usage = call_chat_completion(args, system, user)
            raw = parse_variants(text)
            variants = []
            seen = {caption.strip()}
            for candidate in raw:
                if candidate in seen:
                    continue
                if is_valid_variant(
                    candidate,
                    caption,
                    events,
                    min_token_overlap=args.min_token_overlap,
                    require_event_count=args.require_event_count,
                ):
                    variants.append(candidate)
                    seen.add(candidate)
            return variants, usage, system + "\n" + user, text
        except (urllib.error.URLError, TimeoutError, RuntimeError, KeyError, IndexError, json.JSONDecodeError) as exc:
            if attempt + 1 >= args.retries:
                raise
            wait = args.retry_sleep * (2 ** attempt)
            print(f"[retry {attempt + 1}/{args.retries}] {exc}; sleeping {wait:.1f}s")
            time.sleep(wait)
    return [], {}, system + "\n" + user, ""


def augment_records(records, hard_by_id, args):
    out = []
    spent_usd = 0.0
    called = skipped_budget = 0
    planned_calls = planned_variants = 0
    limit = args.max_records if args.max_records and args.max_records > 0 else None

    for idx, rec in enumerate(records):
        hard_row = hard_by_id.get(str(rec.get("Id", "")), {})
        score = hard_score(hard_row)
        rec_out = annotate_hard_fields(
            rec,
            hard_row,
            score,
            args.max_repeats,
            args.base_caption_aug_prob,
            args.hard_caption_aug_prob,
        )

        existing = normalise_variants(rec_out.get(args.output_field, []))
        target_n = target_variant_count(
            score, args.base_variants, args.hard_extra_variants, args.max_variants_per_record
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
                variants, usage, _, _ = generate_variants(args, rec["caption"], events, needed)
                existing.extend(variants)
                called += 1
                input_tokens = int(usage.get("prompt_tokens") or est_in)
                output_tokens = int(usage.get("completion_tokens") or estimate_tokens(" ".join(variants)))
                spent_usd += estimate_cost_usd(
                    input_tokens,
                    output_tokens,
                    args.price_input_usd_per_1m,
                    args.price_output_usd_per_1m,
                )
                if args.sleep:
                    time.sleep(args.sleep)
            else:
                spent_usd += est_usd

        rec_out[args.output_field] = existing[: args.max_variants_per_record]
        out.append(rec_out)
        if args.save_every and not args.dry_run and (idx + 1) % args.save_every == 0:
            write_jsonl(args.out, out + records[idx + 1 :])

    stats = {
        "records": len(records),
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
    hard_by_id = load_hard_cases(args.hard_cases)
    out, stats = augment_records(records, hard_by_id, args)
    print(json.dumps(stats, indent=2, ensure_ascii=False))
    if not args.dry_run:
        write_jsonl(args.out, out)
        print(f"saved: {args.out}")


if __name__ == "__main__":
    main()
