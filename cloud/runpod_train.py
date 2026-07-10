"""RunPod-friendly wrapper around cloud.train_unsloth.run."""

import argparse
import os

from cloud.train_unsloth import run
from src.data.loader import load_paths
from src.utils.runtime import configure_disk_cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sft_qwen_runpod.yaml")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--smoke", action="store_true", help="train only 32 records")
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--per-device-batch", type=int, default=None)
    ap.add_argument("--grad-accum", type=int, default=None)
    args = ap.parse_args()

    configure_disk_cache()
    paths = load_paths()
    sft = os.path.join(paths["outputs_dir"], "sft_train.jsonl")
    if not os.path.exists(sft):
        raise FileNotFoundError(
            f"Missing {sft}. Run `python -m src.data.split` and "
            "`python -m src.train.cot_target` first."
        )

    limit = 32 if args.smoke and args.limit is None else args.limit
    output_dir = args.output_dir
    if output_dir is None:
        name = "qwen25vl7b_smoke" if args.smoke else "qwen25vl7b"
        output_dir = os.path.join(paths["outputs_dir"], name)

    overrides = {}
    if args.per_device_batch is not None:
        overrides["per_device_batch"] = args.per_device_batch
    if args.grad_accum is not None:
        overrides["grad_accum"] = args.grad_accum

    lora_dir = run(
        args.config,
        sft,
        paths["data_dir"],
        resume=args.resume,
        limit=limit,
        output_dir=output_dir,
        train_overrides=overrides or None,
    )
    print(f"LoRA saved: {lora_dir}")


if __name__ == "__main__":
    main()
