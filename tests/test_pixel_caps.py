"""픽셀 캡: PIL 강제 리사이즈 + processor 캡 이중 설정 검증 (2026-07-06 OOM 회귀 방지)."""

from PIL import Image

from src.infer.predict import PIXELS_MAX, PIXELS_MIN, apply_pixel_caps, cap_pixels


def test_cap_pixels_downscales_large_image():
    img = Image.new("RGB", (1920, 1080))
    out = cap_pixels(img)
    assert out.width * out.height <= PIXELS_MAX
    # 비율 유지 (반올림 오차 허용)
    assert abs(out.width / out.height - 1920 / 1080) < 0.02


def test_cap_pixels_keeps_small_image_unchanged():
    img = Image.new("RGB", (640, 360))  # 230,400 <= 250,880
    assert cap_pixels(img) is img


def test_cap_pixels_never_upscales():
    img = Image.new("RGB", (100, 100))  # PIXELS_MIN 미만이어도 그대로
    assert cap_pixels(img) is img


class _WritableIP:
    pass


class _SizeOnlyIP:
    """min/max_pixels 대입은 받되 무시하는 신형 유사 케이스 — size가 유효 경로."""

    @property
    def read_only(self):
        return None


class _P:
    def __init__(self, ip):
        self.image_processor = ip


def test_apply_pixel_caps_sets_both_attrs_and_size():
    ip = _WritableIP()
    apply_pixel_caps(_P(ip))
    assert ip.min_pixels == PIXELS_MIN
    assert ip.max_pixels == PIXELS_MAX
    assert ip.size == {"shortest_edge": PIXELS_MIN, "longest_edge": PIXELS_MAX}


def test_apply_pixel_caps_no_image_processor():
    apply_pixel_caps(object())  # 예외 없이 통과해야 함
