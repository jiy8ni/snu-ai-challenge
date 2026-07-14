import numpy as np
from PIL import Image

from src.preprocess.frame_quality import (
    crop_letterbox,
    detect_letterbox,
    is_low_info,
    pairwise_phash_distances,
    phash,
)


def _img(arr):
    return Image.fromarray(arr.astype(np.uint8))


def _noise(seed, h=120, w=160):
    rng = np.random.default_rng(seed)
    return rng.integers(0, 255, size=(h, w, 3))


def test_black_frame_is_low_info():
    black = np.zeros((120, 160), dtype=np.float32)
    assert is_low_info(black)


def test_noise_frame_is_not_low_info():
    gray = np.asarray(_img(_noise(0)).convert("L"), dtype=np.float32)
    assert not is_low_info(gray)


def test_letterbox_detection_and_crop():
    arr = _noise(1, h=120, w=160)
    arr[:20] = 0   # 상단 띠
    arr[-15:] = 0  # 하단 띠
    img = _img(arr)
    gray = np.asarray(img.convert("L"), dtype=np.float32)
    top, bottom = detect_letterbox(gray)
    assert top == 20 and bottom == 15
    cropped = crop_letterbox(img)
    assert cropped.height == 120 - 35


def test_no_letterbox_returns_original():
    img = _img(_noise(2))
    assert crop_letterbox(img).size == img.size


def test_phash_duplicate_vs_distinct():
    arr = _noise(3)
    a = _img(arr)
    b = _img(np.clip(arr + 5, 0, 255))  # 밝기만 살짝 다른 근접 중복
    c = _img(_noise(4))                 # 전혀 다른 프레임
    ha, hb, hc = phash(a), phash(b), phash(c)
    assert ha - hb <= 6
    assert ha - hc > 6


def test_pairwise_distances_count():
    hashes = [phash(_img(_noise(s))) for s in range(4)]
    assert len(pairwise_phash_distances(hashes)) == 6
