"""VLSFTDataset: no_ordering 오버샘플링 (이미지 로드 없이 레코드 구성만 검증)."""

import json

from src.train.vl_dataset import VLSFTDataset


def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")


def _records():
    base = {"images": ["a.jpg"] * 4, "caption": "c", "events": None}
    return [
        {**base, "rank": [2, 1, 4, 3], "no_ordering": False},
        {**base, "rank": [1, 2, 3, 4], "no_ordering": True},
        {**base, "rank": [3, 1, 2, 4], "no_ordering": False},
    ]


def test_oversample_off_keeps_records(tmp_path):
    p = tmp_path / "sft.jsonl"
    _write_jsonl(p, _records())
    ds = VLSFTDataset(str(p), data_dir=".", oversample_no_ordering=1)
    assert len(ds) == 3


def test_oversample_duplicates_no_ordering_only(tmp_path):
    p = tmp_path / "sft.jsonl"
    _write_jsonl(p, _records())
    ds = VLSFTDataset(str(p), data_dir=".", oversample_no_ordering=3)
    assert len(ds) == 5  # 3 + no_ordering 1건 x 추가 2벌
    assert sum(r["no_ordering"] for r in ds.records) == 3
    # 원본 순서는 앞부분에 보존 (limit·재현성 로직과의 상호작용 방지)
    assert [r["no_ordering"] for r in ds.records[:3]] == [False, True, False]


def test_oversample_applies_after_limit(tmp_path):
    """limit은 원본 레코드에 먼저 적용된 뒤 오버샘플된다."""
    p = tmp_path / "sft.jsonl"
    _write_jsonl(p, _records())
    ds = VLSFTDataset(str(p), data_dir=".", limit=2, oversample_no_ordering=2)
    assert len(ds) == 3  # 원본 2 (False, True) + True 1벌
