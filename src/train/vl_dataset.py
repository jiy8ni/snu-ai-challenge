"""sft jsonl -> VLM SFT 레코드 (Track B, Unsloth/TRL vision 포맷) + 순열 증강.

증강 규칙 (targets.augment_perm_and_rank):
  - 매 __getitem__마다 프레임 재배치 perm을 샘플링해 이미지 순서와 rank 라벨을 함께 변환
  - 타깃 텍스트는 build_target으로 **재생성** (jsonl의 정적 cot 필드는 사용 금지)
  - no_ordering 샘플은 배치와 무관하게 타깃 [1,2,3,4] 고정
  - orderable 샘플은 증강 후에도 identity rank가 되지 않도록 재추출
  - 선택적으로 캡션의 시간 연결어를 규칙 기반으로 바꾸되 이벤트 순서는 보존

재현성: rng가 호출 순서대로 전진한다 -> dataloader_num_workers=0 권장.
이미지 경로는 jsonl에 Windows 역슬래시로 저장돼 있으므로 정규화한다.
"""

import json
import os
import random
from copy import copy

from PIL import Image

from src.preprocess.caption_augment import augment_caption, caption_variants
from src.preprocess.frame_quality import crop_letterbox
from src.train.targets import augment_perm_and_rank, build_instruction, build_target


def load_records(jsonl_path):
    with open(jsonl_path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def resolve_image_paths(record, data_dir):
    """jsonl의 상대경로(윈도우 역슬래시 포함 가능) -> 절대경로."""
    return [os.path.join(data_dir, p.replace("\\", "/")) for p in record["images"]]


def build_messages(images, caption, style, target=None):
    """Unsloth/TRL vision 포맷 messages. target=None이면 추론용(user 턴만)."""
    content = [{"type": "image", "image": im} for im in images]
    content.append({"type": "text", "text": build_instruction(style, caption)})
    messages = [{"role": "user", "content": content}]
    if target is not None:
        messages.append(
            {"role": "assistant", "content": [{"type": "text", "text": target}]}
        )
    return messages


def _clamp_prob(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        value = 0.0
    return min(1.0, max(0.0, value))


def _load_hard_case_overrides(path):
    if not path:
        return {}
    if not os.path.exists(path):
        raise FileNotFoundError(f"hard_cases_path not found: {path}")

    overrides = {}
    ext = os.path.splitext(path)[1].lower()
    if ext in (".jsonl", ".ndjson"):
        rows = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    else:
        import csv

        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

    allowed = {
        "caption_aug_repeats",
        "caption_aug_prob",
        "hard_score",
        "caption_llm_variants",
    }
    for row in rows:
        sid = row.get("Id") or row.get("id")
        if not sid:
            continue
        overrides[str(sid)] = {
            k: row[k] for k in allowed if k in row and row[k] not in ("", None)
        }
    return overrides


def _normalise_variants(value):
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
    return []


def _expand_by_caption_repeats(records, field, max_repeats):
    expanded = []
    for rec in records:
        try:
            repeats = int(float(rec.get(field, 1)))
        except (TypeError, ValueError):
            repeats = 1
        repeats = max(1, repeats)
        if max_repeats:
            repeats = min(repeats, int(max_repeats))
        expanded.extend([rec] * repeats)
    return expanded


class VLSFTDataset:
    """torch.utils.data.Dataset 프로토콜 (map-style). 반환: {"messages": [...]}"""

    def __init__(
        self, jsonl_path, data_dir, style="mid", augment=True, crop=True, seed=42, limit=None,
        oversample_no_ordering=1, caption_aug_prob=0.0,
        caption_aug_source="rule", llm_caption_field="caption_llm_variants",
        hard_cases_path=None, hard_aug_repeats_field="caption_aug_repeats",
        hard_aug_max_repeats=5,
    ):
        """oversample_no_ordering: no_ordering 레코드를 n배로 복제 (1=off).
        매 __getitem__마다 perm 증강이 새로 뽑히므로 복제본도 서로 다른 뷰가 된다.
        UNORDERABLE recall(_0705 실측 22%) 보강용 — 제공 데이터 증강이라 규정 합법."""
        assert 0.0 <= caption_aug_prob <= 1.0, "caption_aug_prob must be in [0, 1]"
        assert caption_aug_source in ("rule", "llm", "mix"), "caption_aug_source must be rule|llm|mix"
        self.records = load_records(jsonl_path)[: limit or None]
        overrides = _load_hard_case_overrides(hard_cases_path)
        if overrides:
            merged = []
            for rec in self.records:
                sid = str(rec.get("Id", ""))
                if sid in overrides:
                    rec = copy(rec)
                    rec.update(overrides[sid])
                merged.append(rec)
            self.records = merged
        self.records = _expand_by_caption_repeats(
            self.records, hard_aug_repeats_field, hard_aug_max_repeats
        )
        if oversample_no_ordering > 1:
            extra = [r for r in self.records if r["no_ordering"]]
            self.records = self.records + extra * (oversample_no_ordering - 1)
        self.data_dir = data_dir
        self.style = style
        self.augment = augment
        self.crop = crop
        self.rng = random.Random(seed)
        self.caption_rng = random.Random(seed + 1000003)
        self.caption_aug_prob = caption_aug_prob
        self.caption_aug_source = caption_aug_source
        self.llm_caption_field = llm_caption_field

    def __len__(self):
        return len(self.records)

    def load_images(self, record, perm):
        paths = resolve_image_paths(record, self.data_dir)
        paths = [paths[perm[j]] for j in range(4)]
        images = [Image.open(p).convert("RGB") for p in paths]
        return [crop_letterbox(im) for im in images] if self.crop else images

    def __getitem__(self, i):
        rec = self.records[i]
        if self.augment:
            perm, rank = augment_perm_and_rank(rec["rank"], rec["no_ordering"], self.rng)
        else:
            perm, rank = [0, 1, 2, 3], rec["rank"]
        images = self.load_images(rec, perm)
        caption = rec["caption"]
        prob = _clamp_prob(rec.get("caption_aug_prob", self.caption_aug_prob))
        if prob > 0 and self.caption_rng.random() < prob:
            caption = self._sample_caption_variant(rec, caption)
        target = build_target(self.style, rec["events"], rank, rec["no_ordering"])
        return {"messages": build_messages(images, caption, self.style, target)}

    def _sample_caption_variant(self, rec, caption):
        variants = []
        if self.caption_aug_source in ("llm", "mix"):
            variants.extend(_normalise_variants(rec.get(self.llm_caption_field)))
        if self.caption_aug_source in ("rule", "mix"):
            variants.extend(caption_variants(caption))

        clean = []
        seen = {caption.strip()}
        for variant in variants:
            variant = variant.strip()
            if variant and variant not in seen:
                clean.append(variant)
                seen.add(variant)
        if clean:
            return self.caption_rng.choice(clean)
        if self.caption_aug_source == "rule":
            return augment_caption(caption, self.caption_rng)
        return caption
