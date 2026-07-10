"""Prepare paths.yaml for RunPod.

Expected data layout:
  /workspace/snuai/data/train.csv
  /workspace/snuai/data/test.csv
  /workspace/snuai/data/sample_submission.csv
  /workspace/snuai/data/train/<Id>/...
  /workspace/snuai/data/test/<Id>/...

You can override the data root:
  python -m cloud.runpod_prepare --data-root /workspace/path/to/data
"""

import argparse
import os
import shutil

from src.utils.runtime import configure_disk_cache


def _has_competition_files(path):
    return all(
        os.path.exists(os.path.join(path, name))
        for name in ("train.csv", "test.csv", "sample_submission.csv", "train", "test")
    )


def find_data_root(search_root):
    preferred = os.path.join(search_root, "data")
    if _has_competition_files(preferred):
        return preferred
    for base, dirs, files in os.walk(search_root):
        if {"train.csv", "test.csv", "sample_submission.csv"}.issubset(files):
            if "train" in dirs and "test" in dirs:
                return base
    raise FileNotFoundError(
        f"Could not find competition data under {search_root}. "
        "Pass --data-root explicitly."
    )


def write_paths_yaml(base_dir, data_root):
    outputs_dir = os.path.join(base_dir, "outputs")
    models_dir = os.path.join(base_dir, "models")
    reports_dir = os.path.join(base_dir, "reports")
    for path in (outputs_dir, models_dir, reports_dir):
        os.makedirs(path, exist_ok=True)

    path_yaml = os.path.join(base_dir, "paths_runpod.yaml")
    text = "\n".join(
        [
            f"data_dir: {data_root}",
            f"train_csv: {os.path.join(data_root, 'train.csv')}",
            f"test_csv: {os.path.join(data_root, 'test.csv')}",
            f"sample_submission: {os.path.join(data_root, 'sample_submission.csv')}",
            f"train_image_dir: {os.path.join(data_root, 'train')}",
            f"test_image_dir: {os.path.join(data_root, 'test')}",
            f"models_dir: {models_dir}",
            f"outputs_dir: {outputs_dir}",
            f"reports_dir: {reports_dir}",
            "",
        ]
    )
    with open(path_yaml, "w", encoding="utf-8") as f:
        f.write(text)
    return path_yaml, outputs_dir


def maybe_copy_split(repo_root, outputs_dir):
    src = os.path.join(repo_root, "outputs", "split.csv")
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(outputs_dir, "split.csv"))
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-dir", default=os.environ.get("SNUAI_BASE_DIR", "/workspace/snuai"))
    ap.add_argument("--data-root", default=os.environ.get("SNUAI_DATA_ROOT"))
    ap.add_argument("--search-root", default="/workspace")
    args = ap.parse_args()

    os.environ["SNUAI_BASE_DIR"] = args.base_dir
    configure_disk_cache(args.base_dir)

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_root = args.data_root or find_data_root(args.search_root)
    if not _has_competition_files(data_root):
        raise FileNotFoundError(f"Invalid data root: {data_root}")

    path_yaml, outputs_dir = write_paths_yaml(args.base_dir, data_root)
    copied = maybe_copy_split(repo_root, outputs_dir)
    print(f"DATA_ROOT={data_root}")
    print(f"SNUAI_PATHS_CONFIG={path_yaml}")
    print(f"outputs_dir={outputs_dir}")
    if copied:
        print("copied existing outputs/split.csv")
    else:
        print("split.csv not found in repo; run: python -m src.data.split")
    print("\nRun this in the shell:")
    print(f"export SNUAI_BASE_DIR={args.base_dir}")
    print(f"export SNUAI_DATA_ROOT={data_root}")
    print(f"export SNUAI_PATHS_CONFIG={path_yaml}")


if __name__ == "__main__":
    main()
