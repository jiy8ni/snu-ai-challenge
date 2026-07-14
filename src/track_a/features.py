"""프레임 단위 이미지 스칼라 피처 캐시 (Track A 보조 입력).

임베딩으로는 못 보는 저수준 신호 3종을 프레임별로 저장한다:
  low_info (검은/단색), letterbox_frac (검은 띠 비율), phash 쌍거리(6쌍).
frame_quality.py의 함수를 재사용한다. 임베딩 캐시와 별도 파일로 두어
스칼라 on/off ablation을 임베딩 재계산 없이 수행한다.

사용:
  python -m src.track_a.features --split train
산출: outputs/emb/frame_stats_{split}.csv
"""

import argparse
import os

import pandas as pd
from tqdm import tqdm

from src.data.loader import frame_paths, load_paths, load_split
from src.preprocess.frame_quality import (
    detect_letterbox,
    is_low_info,
    load_gray,
    pairwise_phash_distances,
    phash,
)

PAIRS = [(i, j) for i in range(4) for j in range(i + 1, 4)]


def sample_stats(paths4):
    row = {}
    hashes = []
    for i, p in enumerate(paths4, 1):
        gray = load_gray(p)
        top, bottom = detect_letterbox(gray)
        row[f"low_info_{i}"] = bool(is_low_info(gray))
        row[f"letterbox_frac_{i}"] = (top + bottom) / gray.shape[0]
        hashes.append(phash(p))
    for (i, j), d in zip(PAIRS, pairwise_phash_distances(hashes)):
        row[f"phash_d{i}{j}"] = int(d)
    return row


def build(split):
    df = load_split(split)
    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc=f"frame_stats:{split}"):
        row = {"Id": r["Id"]}
        row.update(sample_stats(frame_paths(r)))
        rows.append(row)
    return pd.DataFrame(rows)


def stats_path(split, paths=None):
    paths = paths or load_paths()
    return os.path.join(paths["outputs_dir"], "emb", f"frame_stats_{split}.csv")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=["train", "test"])
    args = ap.parse_args()

    out = stats_path(args.split)
    if os.path.exists(out):
        print(f"already done: {out}")
        return
    os.makedirs(os.path.dirname(out), exist_ok=True)
    stats = build(args.split)
    stats.to_csv(out, index=False)
    print(f"saved: {out} ({len(stats)} rows)")


if __name__ == "__main__":
    main()
