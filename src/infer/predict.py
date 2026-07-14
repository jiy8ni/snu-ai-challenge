"""VLM permutation-TTA 추론 -> raw jsonl (aggregate.py 입력).

Kaggle GPU(학습된 모델)와 본선 3090에서 동일하게 쓰는 플레인 transformers 경로.
로컬 CPU에서는 models/Qwen2-VL-2B-Instruct + --limit 로 파이프라인 스모크가 가능하다.

TTA: 모든 샘플이 동일한 고정 perm 목록(identity + 시드 셔플 N-1개)을 공유한다.
프레임 재배치 -> 모델은 셔플 좌표로 답변 -> aggregate.py가 unshuffle 후 투표.

사용:
  python -m src.infer.predict --model models/Qwen2-VL-2B-Instruct --split test \
      --style mid --tta 4 --out outputs/raw_test.jsonl [--limit 5] [--fold val]
"""

import argparse
import json
import os
import random

import torch
from PIL import Image
from tqdm import tqdm

from src.data.loader import frame_paths, load_paths, load_split
from src.preprocess.frame_quality import crop_letterbox
from src.train.vl_dataset import build_messages
from src.utils.permutation import N_FRAMES
from src.utils.runtime import configure_disk_cache

MAX_NEW_TOKENS = {"short": 32, "mid": 64, "cot": 300}

# 학습 설정(configs/sft_qwen.yaml pixels)과 동일하게 유지할 것.
# 병합 모델의 preprocessor_config에 캡이 유실될 수 있어(2026-07-06 실측: test 대형
# 이미지가 무제한 해상도로 처리돼 ll_score OOM) 추론 코드에서 강제한다.
PIXELS_MIN = 25088   # 32 * 28 * 28
PIXELS_MAX = 250880  # 320 * 28 * 28


def cap_pixels(img, max_pixels=PIXELS_MAX):
    """총 픽셀 수가 예산을 넘는 이미지를 비율 유지 다운스케일 (PIL 수준 보증).

    processor 캡은 transformers 버전에 따라 조용히 무시될 수 있으므로
    (구 min/max_pixels 속성 vs 신 size dict), 입력 이미지 자체를 캡 이내로
    맞춰 토큰 예산 초과를 원천 차단한다. 업스케일은 하지 않는다.
    """
    n = img.width * img.height
    if n <= max_pixels:
        return img
    scale = (max_pixels / n) ** 0.5
    w = max(1, int(img.width * scale))
    h = max(1, int(img.height * scale))
    return img.resize((w, h), Image.LANCZOS)


def apply_pixel_caps(processor, min_pixels=PIXELS_MIN, max_pixels=PIXELS_MAX):
    """train_unsloth.build_model과 동일한 규칙으로 이미지 토큰 예산을 캡.

    구형(min/max_pixels 속성)과 신형(size dict) 어느 쪽이 유효한지 버전마다
    다르고, 무시되는 쪽도 대입은 예외 없이 "성공"하므로 둘 다 설정한다.
    (2026-07-06: except AttributeError 분기는 실제로 도달하지 않는 silent
    no-op였다 — test 대형 이미지 OOM 재발의 원인.)
    """
    ip = getattr(processor, "image_processor", None)
    if ip is None:
        return
    for name, val in (("min_pixels", min_pixels), ("max_pixels", max_pixels)):
        try:
            setattr(ip, name, val)
        except AttributeError:
            pass
    try:
        ip.size = {"shortest_edge": min_pixels, "longest_edge": max_pixels}
    except AttributeError:
        pass


def tta_perms(n, seed=42):
    """identity + 서로 다른 셔플 perm (n-1)개. 전 샘플 공유 (재현성)."""
    perms = [list(range(N_FRAMES))]
    rng = random.Random(seed)
    seen = {tuple(perms[0])}
    while len(perms) < n:
        p = list(range(N_FRAMES))
        rng.shuffle(p)
        if tuple(p) not in seen:
            seen.add(tuple(p))
            perms.append(p)
    return perms


def is_lora_adapter(model_path):
    return os.path.exists(os.path.join(model_path, "adapter_config.json"))


def load_model(model_path, device):
    configure_disk_cache()
    if is_lora_adapter(model_path):
        from unsloth import FastVisionModel

        model, processor = FastVisionModel.from_pretrained(model_path, load_in_4bit=True)
        FastVisionModel.for_inference(model)
        model.eval()
        processor.tokenizer.padding_side = "left"
        apply_pixel_caps(processor)
        print(f"loaded LoRA adapter in 4bit mode: {model_path}")
        return model, processor

    from transformers import AutoProcessor

    try:
        from transformers import AutoModelForImageTextToText as AutoVLM
    except ImportError:  # 구버전 호환
        from transformers import AutoModelForVision2Seq as AutoVLM

    # Qwen2.5-VL은 fp16에서 vision tower 활성값 오버플로로 출력이 깨질 수 있다
    # → bf16 지원 GPU(A100/L4/3090)는 bf16, T4만 fp16 폴백
    if device == "cuda":
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    else:
        dtype = torch.float32
    model = AutoVLM.from_pretrained(model_path, dtype=dtype).to(device).eval()
    processor = AutoProcessor.from_pretrained(model_path)
    processor.tokenizer.padding_side = "left"  # 배치 생성 필수 (right-pad는 생성 위치가 어긋남)
    apply_pixel_caps(processor)
    return model, processor


def load_frames(row, crop):
    images = [Image.open(p).convert("RGB") for p in frame_paths(row)]
    if crop:
        images = [crop_letterbox(im) for im in images]
    return [cap_pixels(im) for im in images]


@torch.inference_mode()
def generate_batch(model, processor, batch_messages, max_new_tokens, device):
    texts = [
        processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True)
        for m in batch_messages
    ]
    image_lists = [
        [c["image"] for c in m[0]["content"] if c["type"] == "image"] for m in batch_messages
    ]
    inputs = processor(text=texts, images=image_lists, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    out = model.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
    trimmed = out[:, inputs["input_ids"].shape[1]:]
    return processor.batch_decode(trimmed, skip_special_tokens=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--fold", default=None, help="train일 때 split.csv fold 필터 (예: val)")
    ap.add_argument("--style", default="mid", choices=["short", "mid", "cot"])
    ap.add_argument("--tta", type=int, default=4)
    ap.add_argument("--batch", type=int, default=4, help="TTA 뷰 단위 배치 크기")
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-crop", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-new-tokens", type=int, default=None)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = load_model(args.model, device)

    df = load_split(args.split)
    if args.fold:
        import pandas as pd

        split_df = pd.read_csv(os.path.join(load_paths()["outputs_dir"], "split.csv"))
        df = df.merge(split_df[["Id", "fold"]], on="Id")
        df = df[df["fold"] == args.fold]
    if args.limit:
        df = df.head(args.limit)

    perms = tta_perms(args.tta, args.seed)
    max_new = args.max_new_tokens or MAX_NEW_TOKENS[args.style]

    done = set()
    if os.path.exists(args.out):  # 재개: 이미 생성된 (Id, perm) 건너뜀
        with open(args.out, encoding="utf-8") as f:
            done = {(r["Id"], tuple(r["perm"])) for r in map(json.loads, f)}

    jobs = []
    for _, row in df.iterrows():
        for perm in perms:
            if (row["Id"], tuple(perm)) not in done:
                jobs.append((row, perm))

    with open(args.out, "a", encoding="utf-8") as fout:
        for s in tqdm(range(0, len(jobs), args.batch), desc=f"predict:{args.split}"):
            chunk = jobs[s : s + args.batch]
            ok, messages = [], []
            for row, perm in chunk:
                try:
                    frames = load_frames(row, crop=not args.no_crop)
                except OSError as e:
                    # 손상 이미지 1장이 장시간 실행 전체를 죽이지 않게 건너뜀.
                    # 기록은 남기지 않으므로 데이터 복구 후 같은 명령으로 재개하면 재시도된다.
                    print(f"[손상 이미지 스킵] Id={row['Id']} perm={perm}: {e}")
                    continue
                frames = [frames[perm[j]] for j in range(N_FRAMES)]
                ok.append((row, perm))
                messages.append(build_messages(frames, row["Sentence"], args.style))
            if not messages:
                continue
            texts = generate_batch(model, processor, messages, max_new, device)
            for (row, perm), text in zip(ok, texts):
                fout.write(json.dumps(
                    {"Id": row["Id"], "perm": perm, "text": text}, ensure_ascii=False
                ) + "\n")
            fout.flush()
    print(f"saved: {args.out} (+{len(jobs)} generations)")


if __name__ == "__main__":
    main()
