import json

from src.infer.predict import select_caption
from src.preprocess.caption_augment import caption_variants

TWO_EVENT = "A man opens the door, then he walks inside."
ONE_EVENT = "A man walks inside."


def test_select_caption_zero_returns_original():
    assert select_caption(TWO_EVENT, 0) == TWO_EVENT


def test_select_caption_first_variant_matches_and_deterministic():
    expected = caption_variants(TWO_EVENT)[0]
    first = select_caption(TWO_EVENT, 1)
    assert first == expected
    # 규칙 기반이므로 두 번 호출해도 동일해야 한다 (재현성).
    assert select_caption(TWO_EVENT, 1) == first


def test_select_caption_missing_variant_returns_none():
    # 이벤트 1개 문장은 변형이 없으므로 이 실행에서 제외된다.
    assert caption_variants(ONE_EVENT) == []
    assert select_caption(ONE_EVENT, 1) is None


def _done_key(record):
    """predict.main의 재개 done-key 구성과 동일 — cap 없는 구 raw는 0으로 읽힘."""
    return (record["Id"], tuple(record["perm"]), record.get("cap", 0))


def test_done_key_backcompat_defaults_cap_zero():
    # cap 필드가 없는 (구) raw 레코드는 cap=0 뷰와 같은 키로 매칭돼야 한다.
    legacy = json.loads('{"Id": "abc", "perm": [0, 1, 2, 3], "text": "1 2 3 4"}')
    assert _done_key(legacy) == ("abc", (0, 1, 2, 3), 0)

    tagged = json.loads('{"Id": "abc", "perm": [0, 1, 2, 3], "text": "1 2 3 4", "cap": 1}')
    assert _done_key(tagged) == ("abc", (0, 1, 2, 3), 1)
    assert _done_key(legacy) != _done_key(tagged)
