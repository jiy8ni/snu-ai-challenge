"""sft jsonl -> VLM SFT 레코드 (Track B, Unsloth/TRL vision 포맷) + 순열 증강.

증강 규칙 (targets.augment_perm_and_rank):
  - 매 __getitem__마다 프레임 재배치 perm을 샘플링해 이미지 순서와 rank 라벨을 함께 변환
  - 타깃 텍스트는 build_target으로 **재생성** (jsonl의 정적 cot 필드는 사용 금지)
  - no_ordering 샘플은 배치와 무관하게 타깃 [1,2,3,4] 고정
  - orderable 샘플은 증강 후에도 identity rank가 되지 않도록 재추출

재현성: 공유 rng가 호출 순서대로 전진한다 -> dataloader_num_workers=0 권장.
이미지 경로는 jsonl에 Windows 역슬래시로 저장돼 있으므로 정규화한다.
"""

import json
import os
import random

from PIL import Image

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


class VLSFTDataset:
    """torch.utils.data.Dataset 프로토콜 (map-style). 반환: {"messages": [...]}"""

    def __init__(
        self, jsonl_path, data_dir, style="mid", augment=True, crop=True, seed=42, limit=None,
        oversample_no_ordering=1,
    ):
        """oversample_no_ordering: no_ordering 레코드를 n배로 복제 (1=off).
        매 __getitem__마다 perm 증강이 새로 뽑히므로 복제본도 서로 다른 뷰가 된다.
        UNORDERABLE recall(_0705 실측 22%) 보강용 — 제공 데이터 증강이라 규정 합법."""
        self.records = load_records(jsonl_path)[: limit or None]
        if oversample_no_ordering > 1:
            extra = [r for r in self.records if r["no_ordering"]]
            self.records = self.records + extra * (oversample_no_ordering - 1)
        self.data_dir = data_dir
        self.style = style
        self.augment = augment
        self.crop = crop
        self.rng = random.Random(seed)

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
        target = build_target(self.style, rec["events"], rank, rec["no_ordering"])
        return {"messages": build_messages(images, rec["caption"], self.style, target)}
