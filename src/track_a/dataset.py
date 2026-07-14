"""임베딩 npz 캐시 + 라벨 -> Track A 텐서 번들.

스칼라 8종 (전부 프레임별 대칭 함수 — 모델 등변성을 깨지 않음):
  0-2: 다른 프레임들과의 코사인 max / mean / min
  3:   최고 이벤트 코사인 (유효 이벤트 한정)
  4:   이벤트 마진 top1-top2 (이벤트 < 2면 0)
  5:   low_info 플래그              (frame_stats CSV, 없으면 0)
  6:   letterbox_frac               (frame_stats CSV, 없으면 0)
  7:   프레임이 속한 쌍의 최소 phash 거리 / 64 (frame_stats CSV, 없으면 0)
"""

import os

import numpy as np
import pandas as pd
import torch

from src.data.loader import load_paths, load_split
from src.track_a.features import PAIRS, stats_path
from src.utils.permutation import parse_answer_column

N_SCALARS = 8


def _emb_path(split, variant, paths):
    return os.path.join(paths["outputs_dir"], "emb", f"siglip2b384_{split}_{variant}.npz")


def _embedding_scalars(img, events, event_mask):
    """임베딩에서 유도되는 스칼라 5종. img (N,4,D), events (N,E,D) — L2 정규화 가정."""
    ff = torch.einsum("nid,njd->nij", img, img)                     # (N,4,4)
    off_diag = ff.masked_fill(torch.eye(4, dtype=torch.bool).unsqueeze(0), float("nan"))
    ff_max = off_diag.nan_to_num(float("-inf")).max(dim=2).values
    ff_mean = off_diag.nanmean(dim=2)
    ff_min = off_diag.nan_to_num(float("inf")).min(dim=2).values

    fe = torch.einsum("nfd,ned->nfe", img, events)                  # (N,4,E)
    fe = fe.masked_fill(~event_mask.unsqueeze(1), float("-inf"))
    top2 = fe.topk(2, dim=2).values                                 # (N,4,2)
    ev_best = top2[..., 0]
    n_valid = event_mask.sum(dim=1)                                 # (N,)
    margin = torch.where(
        (n_valid >= 2).unsqueeze(1), top2[..., 0] - top2[..., 1], torch.zeros_like(ev_best)
    )
    return torch.stack([ff_max, ff_mean, ff_min, ev_best, margin], dim=2)  # (N,4,5)


def _stats_scalars(ids, split, paths):
    """frame_stats CSV -> (N,4,3) [low_info, letterbox_frac, phash_min/64]. 없으면 0 + False."""
    path = stats_path(split, paths)
    n = len(ids)
    if not os.path.exists(path):
        return torch.zeros(n, 4, 3), np.zeros(n, dtype=bool), False
    stats = pd.read_csv(path).set_index("Id").reindex(ids)
    assert not stats.isna().any().any(), f"frame_stats에 누락 Id 존재: {path}"

    out = np.zeros((n, 4, 3), dtype=np.float32)
    for i in range(4):
        out[:, i, 0] = stats[f"low_info_{i + 1}"].astype(float)
        out[:, i, 1] = stats[f"letterbox_frac_{i + 1}"].astype(float)
        pair_cols = [f"phash_d{a}{b}" for (a, b) in PAIRS if i in (a, b)]
        out[:, i, 2] = stats[pair_cols].min(axis=1) / 64.0
    noise_flag = out[:, :, 0].sum(axis=1) > 0                       # 샘플에 저정보 프레임 존재
    return torch.from_numpy(out), noise_flag, True


class TrackAData:
    """전 샘플 텐서를 메모리에 보유 (9.5k 샘플 x ~50KB — 문제 없음)."""

    def __init__(self, ids, img, scalars, events, event_mask, caption, meta):
        self.ids = ids
        self.img = img
        self.scalars = scalars
        self.events = events
        self.event_mask = event_mask
        self.caption = caption
        self.meta = meta  # DataFrame: rank/no_ordering/fold/caption_group/noise_flag (train만)

    def __len__(self):
        return len(self.ids)

    def tensors(self, idx):
        """인덱스 배열 -> 모델 forward 인자 튜플."""
        return (
            self.img[idx], self.scalars[idx], self.events[idx],
            self.event_mask[idx], self.caption[idx],
        )


def load_track_a(split, variant="crop", paths=None):
    paths = paths or load_paths()
    npz_file = _emb_path(split, variant, paths)
    assert os.path.exists(npz_file), f"임베딩 캐시 없음: {npz_file} (siglip_embed 먼저 실행)"
    z = np.load(npz_file, allow_pickle=False)

    ids = [str(x) for x in z["ids"]]
    img = torch.from_numpy(z["img"].astype(np.float32))
    events = torch.from_numpy(z["events"].astype(np.float32))
    event_mask = torch.from_numpy(z["event_mask"])
    caption = torch.from_numpy(z["caption"].astype(np.float32))

    emb_sc = _embedding_scalars(img, events, event_mask)
    stat_sc, noise_flag, has_stats = _stats_scalars(ids, split, paths)
    scalars = torch.cat([emb_sc, stat_sc], dim=2)
    assert scalars.shape[1:] == (4, N_SCALARS)

    df = load_split(split, paths).set_index("Id").reindex(ids).reset_index()
    meta = pd.DataFrame({"Id": ids})
    meta["noise_flag"] = noise_flag
    meta["has_stats"] = has_stats
    if split == "train":
        meta["rank"] = [parse_answer_column(a) for a in df["Answer"]]
        meta["no_ordering"] = df["No_ordering"].astype(bool).values
        split_df = pd.read_csv(os.path.join(paths["outputs_dir"], "split.csv"))
        merged = meta.merge(split_df[["Id", "fold", "caption_group"]], on="Id", how="left")
        assert not merged["fold"].isna().any(), "split.csv에 없는 Id 존재"
        meta = merged

    return TrackAData(ids, img, scalars, events, event_mask, caption, meta)
