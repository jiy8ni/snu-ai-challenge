"""본선 3090 24h 추론 예산 실측 (docs/rules.md §3.3 — 유일한 미검증 규정 항목).

소량 샘플로 초/샘플을 실측하고 test 전체(819)로 외삽한다. A100에서 실행 후
--gpu-factor(기본 3.0, 보수적)를 곱해 3090 예상 시간을 문서화한다.
워밍업 1샘플은 측정에서 제외한다 (컴파일·캐시 워밍 편향 방지).

사용 (Colab A100):
  python -m src.infer.budget_check --model MERGED --n 20 --tta 8 --ll-tta 0 --batch 16
  python -m src.infer.budget_check --model MERGED --n 20 --tta 0 --ll-tta 4 --chunk 25
3090 리허설(본선 환경)에서는 --gpu-factor 1.0 으로 실측값 그대로 판독.
"""

import argparse
import json
import time

import torch

from src.data.loader import load_split
from src.infer.ll_score import candidate_texts, score_view
from src.infer.predict import generate_batch, load_frames, load_model, tta_perms
from src.train.vl_dataset import build_messages
from src.utils.permutation import N_FRAMES

N_TEST = 819


def measure(fn, rows):
    """rows 각각에 fn을 적용, 첫 샘플(워밍업) 제외 초/샘플."""
    times = []
    for i, row in enumerate(rows):
        t0 = time.perf_counter()
        fn(row)
        if i > 0:
            times.append(time.perf_counter() - t0)
    return sum(times) / max(len(times), 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="test", choices=["train", "test"])
    ap.add_argument("--n", type=int, default=20, help="측정 샘플 수 (워밍업 1 포함)")
    ap.add_argument("--tta", type=int, default=8, help="생성 TTA 뷰 수 (0=생성 측정 생략)")
    ap.add_argument("--ll-tta", type=int, default=1, help="LL 뷰 수 (0=LL 측정 생략)")
    ap.add_argument("--batch", type=int, default=16, help="생성 배치 (3090은 4~8 권장)")
    ap.add_argument("--chunk", type=int, default=25, help="LL forward당 후보 수 (3090은 4 권장)")
    ap.add_argument("--style", default="mid")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--gpu-factor", type=float, default=3.0,
                    help="A100 -> 3090 보수 환산 계수 (3090에서 실측 시 1.0)")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = load_model(args.model, device)
    rows = [row for _, row in load_split(args.split).head(args.n).iterrows()]
    report = {"n_measured": args.n - 1, "device": torch.cuda.get_device_name(0) if device == "cuda" else "cpu"}
    total_sec = 0.0

    if args.tta > 0:
        perms = tta_perms(args.tta, args.seed)

        def run_gen(row):
            frames = load_frames(row, crop=True)
            msgs = [
                build_messages([frames[p[j]] for j in range(N_FRAMES)], row["Sentence"], args.style)
                for p in perms
            ]
            for s in range(0, len(msgs), args.batch):
                generate_batch(model, processor, msgs[s:s + args.batch],
                               args.max_new_tokens, device)

        sec = measure(run_gen, rows)
        report["gen"] = {"tta": args.tta, "batch": args.batch, "sec_per_sample": round(sec, 2),
                         "test_hours": round(sec * N_TEST / 3600, 2)}
        total_sec += sec * N_TEST

    if args.ll_tta > 0:
        ll_perms = tta_perms(args.ll_tta, args.seed)
        processor.tokenizer.padding_side = "right"  # score_view 요건 (cmd_score와 동일)

        def run_ll(row):
            frames = load_frames(row, crop=True)
            for perm in ll_perms:
                view = [frames[perm[j]] for j in range(N_FRAMES)]
                cands = candidate_texts(perm)
                score_view(model, processor, view, row["Sentence"],
                           [t for _, t in cands], device, args.chunk)

        sec = measure(run_ll, rows)
        report["ll"] = {"tta": args.ll_tta, "chunk": args.chunk, "sec_per_sample": round(sec, 2),
                        "test_hours": round(sec * N_TEST / 3600, 2)}
        total_sec += sec * N_TEST

    measured_h = total_sec / 3600
    report["total_hours_measured_gpu"] = round(measured_h, 2)
    report["gpu_factor"] = args.gpu_factor
    report["total_hours_3090_est"] = round(measured_h * args.gpu_factor, 2)
    report["verdict"] = (
        "OK (<= 20h, 버퍼 4h)" if measured_h * args.gpu_factor <= 20
        else "위험: TTA 축소 또는 batch/chunk 조정 필요"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
