"""CSV + 이미지 경로 로더. 경로는 configs/paths.yaml 단일 정의를 따른다."""

import os

import pandas as pd
import yaml

_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))


def load_paths(config_path=None):
    """우선순위: 인자 > 환경변수 SNUAI_PATHS_CONFIG (Kaggle 등 외부 환경) > 기본 yaml."""
    path = config_path or os.environ.get("SNUAI_PATHS_CONFIG") or os.path.join(
        _ROOT, "configs", "paths.yaml"
    )
    with open(path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return {k: os.path.join(_ROOT, v) if isinstance(v, str) else v for k, v in cfg.items()}


def load_split(split, paths=None):
    """split: 'train' | 'test'. 프레임 절대 경로 컬럼 Path_1~4를 추가해 반환."""
    assert split in ("train", "test")
    paths = paths or load_paths()
    df = pd.read_csv(paths[f"{split}_csv"])
    image_dir = paths[f"{split}_image_dir"]
    for i in range(1, 5):
        df[f"Path_{i}"] = df.apply(
            lambda r, i=i: os.path.join(image_dir, r["Id"], r[f"Input_{i}"]), axis=1
        )
    return df


def frame_paths(row):
    """한 샘플(row)의 4개 프레임 절대 경로 리스트."""
    return [row[f"Path_{i}"] for i in range(1, 5)]
