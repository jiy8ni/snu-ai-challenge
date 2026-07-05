"""SigLIP2 동결 임베딩 캐시 추출 (Track A 입력).

train/test 전 프레임 + 캡션 + 이벤트 절을 1회 임베딩해 npz로 저장한다.
레터박스 크롭 on/off 두 벌을 지원한다 (전처리 ablation의 기반, reports/preprocessing.md 행 1).

CPU에서 수 시간이 걸리므로 청크 단위로 저장하고, 재실행 시 완료된 청크는 건너뛴다.

사용:
  python -m src.embed.siglip_embed --split test --crop
  python -m src.embed.siglip_embed --split train --crop
  python -m src.embed.siglip_embed --split train            # nocrop 변형 (ablation용)
  python -m src.embed.siglip_embed --split test --crop --limit 8   # 스모크

산출: outputs/emb/{model_tag}_{split}_{crop|nocrop}.npz
  ids (N,) | img (N,4,D) fp16 L2정규화 | caption (N,D) fp16
  events (N,MAX_EVENTS,D) fp16 | event_mask (N,MAX_EVENTS) bool | n_events (N,)
"""

import argparse
import glob
import os

import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from src.data.loader import frame_paths, load_paths, load_split
from src.preprocess.caption_events import split_events
from src.preprocess.frame_quality import crop_letterbox

MODEL_NAME = "google/siglip2-base-patch16-384"
MODEL_TAG = "siglip2b384"
MAX_EVENTS = 6          # 이벤트 절 상한 (5개 초과는 0.2% 미만 → 잘라냄)
CHUNK_SAMPLES = 250     # 청크당 샘플 수 (이미지 1,000장)
IMG_BATCH = 16
TXT_BATCH = 64


def local_model_dir():
    """models/ 아래에 스냅샷이 있으면 그 경로를, 없으면 HF id를 반환."""
    paths = load_paths()
    cand = os.path.join(paths["models_dir"], MODEL_NAME.split("/")[-1])
    return cand if os.path.exists(os.path.join(cand, "config.json")) else MODEL_NAME


class SiglipEmbedder:
    def __init__(self, model_name=None, device="cpu"):
        from transformers import AutoModel, AutoProcessor

        torch.set_num_threads(os.cpu_count() or 8)
        self.device = device
        name = model_name or local_model_dir()
        self.model = AutoModel.from_pretrained(name).to(device).eval()
        self.processor = AutoProcessor.from_pretrained(name)

    @staticmethod
    def _to_tensor(out):
        # transformers v4는 텐서, v5는 ModelOutput(pooler_output=투영 임베딩)
        return out if isinstance(out, torch.Tensor) else out.pooler_output

    @torch.inference_mode()
    def embed_images(self, images):
        """PIL 이미지 리스트 -> (N, D) float32, L2 정규화."""
        chunks = []
        for i in range(0, len(images), IMG_BATCH):
            batch = images[i : i + IMG_BATCH]
            inputs = self.processor(images=batch, return_tensors="pt").to(self.device)
            emb = self._to_tensor(self.model.get_image_features(**inputs))
            chunks.append(torch.nn.functional.normalize(emb, dim=-1).cpu())
        return torch.cat(chunks).numpy()

    @torch.inference_mode()
    def embed_texts(self, texts):
        """문자열 리스트 -> (N, D) float32, L2 정규화. SigLIP은 max_length 패딩으로 학습됨."""
        chunks = []
        for i in range(0, len(texts), TXT_BATCH):
            batch = texts[i : i + TXT_BATCH]
            inputs = self.processor(
                text=batch, return_tensors="pt",
                padding="max_length", max_length=64, truncation=True,
            ).to(self.device)
            emb = self._to_tensor(self.model.get_text_features(**inputs))
            chunks.append(torch.nn.functional.normalize(emb, dim=-1).cpu())
        return torch.cat(chunks).numpy()


def load_frame(path, crop):
    img = Image.open(path).convert("RGB")
    return crop_letterbox(img) if crop else img


def process_chunk(embedder, rows, crop):
    """샘플 rows(list of Series) -> 청크 배열 dict."""
    n = len(rows)
    images = []
    for r in rows:
        images.extend(load_frame(p, crop) for p in frame_paths(r))
    img_emb = embedder.embed_images(images)              # (n*4, D)
    dim = img_emb.shape[1]
    img_emb = img_emb.reshape(n, 4, dim)

    captions = [r["Sentence"] for r in rows]
    events_per = [(split_events(s) or [s])[:MAX_EVENTS] for s in captions]
    flat_events = [e for evs in events_per for e in evs]
    txt_emb = embedder.embed_texts(captions + flat_events)
    cap_emb, ev_emb = txt_emb[:n], txt_emb[n:]

    events = np.zeros((n, MAX_EVENTS, dim), dtype=np.float16)
    mask = np.zeros((n, MAX_EVENTS), dtype=bool)
    pos = 0
    for i, evs in enumerate(events_per):
        k = len(evs)
        events[i, :k] = ev_emb[pos : pos + k].astype(np.float16)
        mask[i, :k] = True
        pos += k

    return {
        "ids": np.array([r["Id"] for r in rows]),
        "img": img_emb.astype(np.float16),
        "caption": cap_emb.astype(np.float16),
        "events": events,
        "event_mask": mask,
        "n_events": np.array([len(e) for e in events_per], dtype=np.int16),
    }


def merge_chunks(tmp_dir, out_path):
    files = sorted(glob.glob(os.path.join(tmp_dir, "chunk_*.npz")))
    parts = [dict(np.load(f, allow_pickle=False)) for f in files]
    merged = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    np.savez_compressed(out_path, **merged)
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    ap.add_argument("--crop", action="store_true", help="레터박스 크롭 적용 변형")
    ap.add_argument("--limit", type=int, default=None, help="샘플 수 제한 (스모크)")
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    paths = load_paths()
    variant = "crop" if args.crop else "nocrop"
    tag = f"{MODEL_TAG}_{args.split}_{variant}"
    emb_dir = os.path.join(paths["outputs_dir"], "emb")
    # limit(스모크) 실행은 별도 tmp에 격리 — 본 실행의 청크 재개 로직 오염 방지
    tmp_suffix = f"_limit{args.limit}" if args.limit else ""
    tmp_dir = os.path.join(emb_dir, f"tmp_{tag}{tmp_suffix}")
    out_path = os.path.join(emb_dir, f"{tag}.npz")
    os.makedirs(tmp_dir, exist_ok=True)

    if os.path.exists(out_path) and args.limit is None:
        print(f"already done: {out_path}")
        return

    df = load_split(args.split)
    if args.limit:
        df = df.head(args.limit)
    rows = [r for _, r in df.iterrows()]

    embedder = SiglipEmbedder(model_name=args.model)
    starts = list(range(0, len(rows), CHUNK_SAMPLES))
    for start in tqdm(starts, desc=tag, unit="chunk"):
        chunk_path = os.path.join(tmp_dir, f"chunk_{start:06d}.npz")
        if os.path.exists(chunk_path):
            continue
        data = process_chunk(embedder, rows[start : start + CHUNK_SAMPLES], args.crop)
        tmp = chunk_path + ".part.npz"
        np.savez(tmp, **data)
        os.replace(tmp, chunk_path)

    if args.limit is None:
        merged = merge_chunks(tmp_dir, out_path)
        assert len(merged["ids"]) == len(df), (len(merged["ids"]), len(df))
        assert list(merged["ids"]) == list(df["Id"]), "청크 병합 순서가 CSV와 불일치"
        print(f"saved: {out_path} (N={len(merged['ids'])}, D={merged['img'].shape[-1]})")
    else:
        print(f"smoke ok: {len(rows)} samples -> chunks in {tmp_dir}")


if __name__ == "__main__":
    main()
