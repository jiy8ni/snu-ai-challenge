"""프레임 단위 품질 신호: phash, 저정보(검은/단색) 탐지, 레터박스 검출·크롭.

산출 피처는 No_ordering 판별(no_ordering.py)과 학습 데이터 정제에 쓰인다.
"""

import numpy as np
from PIL import Image

# 저정보 프레임: grayscale 표준편차가 이 값 미만이면 플래그
LOW_STD_THRESHOLD = 10.0
# 근접 중복: phash(64bit) 해밍거리가 이 값 이하면 중복 쌍
PHASH_DUP_THRESHOLD = 6
# 레터박스: 행 평균 밝기가 이 값 미만이면 검은 띠로 간주
LETTERBOX_DARK_ROW = 8.0


def load_gray(path):
    return np.asarray(Image.open(path).convert("L"), dtype=np.float32)


def phash(path_or_img):
    import imagehash  # 지연 임포트: crop_letterbox만 쓰는 클라우드 환경의 의존성 최소화

    img = Image.open(path_or_img) if isinstance(path_or_img, str) else path_or_img
    return imagehash.phash(img)


def pixel_std(gray):
    return float(gray.std())


def entropy(gray, bins=64):
    hist, _ = np.histogram(gray, bins=bins, range=(0, 255))
    p = hist / max(hist.sum(), 1)
    p = p[p > 0]
    return float(-(p * np.log2(p)).sum())


def is_low_info(gray):
    """검은 프레임/단색 프레임 여부."""
    return pixel_std(gray) < LOW_STD_THRESHOLD


def detect_letterbox(gray):
    """상하 검은 띠 (top, bottom) 픽셀 수. 없으면 (0, 0)."""
    row_mean = gray.mean(axis=1)
    h = len(row_mean)
    top = 0
    while top < h // 3 and row_mean[top] < LETTERBOX_DARK_ROW:
        top += 1
    bottom = 0
    while bottom < h // 3 and row_mean[h - 1 - bottom] < LETTERBOX_DARK_ROW:
        bottom += 1
    return top, bottom


def crop_letterbox(img):
    """PIL 이미지에서 상하 검은 띠를 제거해 반환 (없으면 원본 그대로)."""
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    top, bottom = detect_letterbox(gray)
    if top == 0 and bottom == 0:
        return img
    h = img.height
    return img.crop((0, top, img.width, h - bottom))


def frame_features(path):
    """단일 프레임의 품질 피처 dict."""
    img = Image.open(path)
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    top, bottom = detect_letterbox(gray)
    return {
        "phash": phash(img),
        "std": pixel_std(gray),
        "entropy": entropy(gray),
        "low_info": is_low_info(gray),
        "letterbox_px": top + bottom,
        "width": img.width,
        "height": img.height,
    }


def pairwise_phash_distances(hashes):
    """4개 프레임 phash -> 6개 쌍별 해밍거리 (i<j 순서)."""
    n = len(hashes)
    return [hashes[i] - hashes[j] for i in range(n) for j in range(i + 1, n)]


def sample_features(paths):
    """한 샘플(프레임 경로 4개)의 품질 피처."""
    frames = [frame_features(p) for p in paths]
    dists = pairwise_phash_distances([f["phash"] for f in frames])
    return {
        "phash_dists": dists,
        "min_phash_dist": min(dists),
        "n_dup_pairs": sum(d <= PHASH_DUP_THRESHOLD for d in dists),
        "n_low_info": sum(f["low_info"] for f in frames),
        "min_std": min(f["std"] for f in frames),
        "min_entropy": min(f["entropy"] for f in frames),
        "any_letterbox": any(f["letterbox_px"] > 0 for f in frames),
    }
