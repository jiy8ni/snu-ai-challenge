"""Merge a RunPod LoRA checkpoint into a plain transformers model."""

import argparse
import os

from src.data.loader import load_paths
from src.utils.runtime import configure_disk_cache


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", default=None, help="LoRA dir; default outputs/qwen25vl7b/lora")
    ap.add_argument("--out", default=None, help="merged model dir; default models/qwen25vl7b_merged")
    args = ap.parse_args()

    configure_disk_cache()
    paths = load_paths()
    lora_dir = args.lora or os.path.join(paths["outputs_dir"], "qwen25vl7b", "lora")
    out_dir = args.out or os.path.join(paths["models_dir"], "qwen25vl7b_merged")
    os.makedirs(os.path.dirname(out_dir), exist_ok=True)

    from unsloth import FastVisionModel

    model, processor = FastVisionModel.from_pretrained(lora_dir, load_in_4bit=False)
    merged = model.merge_and_unload()
    merged.save_pretrained(out_dir)
    processor.save_pretrained(out_dir)
    print(f"merged saved: {out_dir}")


if __name__ == "__main__":
    main()
