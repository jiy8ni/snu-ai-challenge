"""VLSFTDataset: no_ordering 오버샘플링 + plain 진실 증강 (이미지 로드 없이 레코드 구성만 검증)."""

import json
import re

import pytest

from src.preprocess.caption_events import split_events
from src.train.targets import chronological_perm
from src.train.vl_dataset import VLSFTDataset
from src.utils.permutation import IDENTITY, parse_permutation, shuffle_rank_label


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


def test_caption_augmentation_changes_prompt_only_with_same_events(tmp_path):
    caption = "A man opens the door, then he walks inside, finally he sits down."
    events = split_events(caption)
    record = {
        "images": ["a.jpg"] * 4,
        "caption": caption,
        "events": events,
        "rank": [2, 1, 3, 4],
        "no_ordering": False,
    }
    p = tmp_path / "sft.jsonl"
    _write_jsonl(p, [record])
    ds = VLSFTDataset(
        str(p), data_dir=".", augment=False, crop=False, caption_aug_prob=1.0, seed=0
    )
    ds.load_images = lambda _record, _perm: ["im1", "im2", "im3", "im4"]

    item = ds[0]
    prompt = item["messages"][0]["content"][-1]["text"]
    augmented = re.search(
        r'Caption describing the video in temporal order: "(.+)"', prompt
    ).group(1)

    assert augmented != caption
    assert split_events(augmented) == events
    assert item["messages"][1]["content"][0]["text"].endswith("Answer: [2, 1, 3, 4]")


def test_caption_repeats_field_expands_records(tmp_path):
    """caption_aug_repeats(공유 재가중 필드)가 레코드를 물리 복제한다 (LLM 무관)."""
    caption = "A man opens the door, then he walks inside."
    record = {
        "Id": "hard-1",
        "images": ["a.jpg"] * 4,
        "caption": caption,
        "caption_aug_repeats": 3,
        "events": split_events(caption),
        "rank": [2, 1, 3, 4],
        "no_ordering": False,
    }
    p = tmp_path / "sft.jsonl"
    _write_jsonl(p, [record])

    ds = VLSFTDataset(str(p), data_dir=".", augment=False, crop=False)
    ds.load_images = lambda _record, _perm: ["im1", "im2", "im3", "im4"]

    assert len(ds) == 3  # caption_aug_repeats=3 -> 3벌


def test_hard_cases_path_overrides_repeats_and_prob(tmp_path):
    """hard_cases_path CSV가 caption_aug_repeats/prob를 레코드 위에 덮어쓴다 (모델 예측 기반, LLM 무관).

    또한 이제 제거된 caption_llm_variants 컬럼은 무시됨을 확인한다.
    """
    caption = "A skier jumps, then lands."
    llm_leftover = "A skier jumps and then lands."
    record = {
        "Id": "case-1",
        "images": ["a.jpg"] * 4,
        "caption": caption,
        "caption_llm_variants": [llm_leftover],  # 잔재 컬럼 — 이제 무시돼야 한다
        "events": split_events(caption),
        "rank": [1, 2, 3, 4],
        "no_ordering": True,
    }
    p = tmp_path / "sft.jsonl"
    _write_jsonl(p, [record])
    hard = tmp_path / "hard.csv"
    hard.write_text(
        "Id,caption_aug_repeats,caption_aug_prob\ncase-1,4,1.0\n",
        encoding="utf-8",
    )

    ds = VLSFTDataset(
        str(p),
        data_dir=".",
        augment=False,
        crop=False,
        hard_cases_path=str(hard),
        oversample_no_ordering=1,
    )
    ds.load_images = lambda _record, _perm: ["im1", "im2", "im3", "im4"]

    assert len(ds) == 4  # CSV repeats=4 오버라이드
    # LLM 변형 잔재는 프롬프트에 절대 나타나지 않는다 (rule 경로만 사용)
    for i in range(len(ds)):
        assert llm_leftover not in ds[i]["messages"][0]["content"][-1]["text"]


# ---------------------------------------------------------------------------
# plain 스타일 = 진실 라벨 증강 경로 (_0716). src/train/targets.py docstring 참조.
# ---------------------------------------------------------------------------


def _plain_ds(tmp_path, records, **kw):
    p = tmp_path / "sft.jsonl"
    _write_jsonl(p, records)
    ds = VLSFTDataset(str(p), data_dir=".", style="plain", crop=False, seed=7, **kw)
    # perm이 실제로 적용됐는지 보려면 이미지 자리에 인덱스를 넣어 되돌려 본다
    ds.load_images = lambda _record, perm: [f"im{perm[j]}" for j in range(4)]
    return ds


def _perm_of(item):
    """_plain_ds의 load_images 스텁이 심어둔 인덱스 -> 실제 적용된 perm."""
    return [int(c["image"][2:]) for c in item["messages"][0]["content"] if c["type"] == "image"]


def test_plain_labels_are_truthful_including_no_ordering(tmp_path):
    """핵심 회귀 방지: no_ordering 레코드도 섞였으면 라벨이 따라가야 한다.

    구 경로(augment_perm_and_rank)는 여기서 프레임을 섞고도 [1,2,3,4]를 가르쳤다.
    """
    records = [
        {"images": ["a.jpg"] * 4, "caption": "c", "events": None,
         "rank": [1, 2, 3, 4], "no_ordering": True},
        {"images": ["a.jpg"] * 4, "caption": "c", "events": None,
         "rank": [3, 1, 2, 4], "no_ordering": False},
    ]
    ds = _plain_ds(tmp_path, records)
    for i in range(len(ds)):
        for _ in range(40):
            item = ds[i]
            target = item["messages"][1]["content"][0]["text"]
            label = parse_permutation(target)
            perm = _perm_of(item)
            # 라벨이 실제 프레임 재배치와 일치하는가
            assert shuffle_rank_label(ds.records[i]["rank"], perm) == label
            assert "UNORDERABLE" not in target


def test_plain_identity_label_implies_chronological_perm(tmp_path):
    """identity 타깃이 나온 뷰는 프레임이 실제로 시간순으로 배치돼 있어야 한다."""
    records = [{"images": ["a.jpg"] * 4, "caption": "c", "events": None,
                "rank": [3, 1, 2, 4], "no_ordering": False}]
    ds = _plain_ds(tmp_path, records)
    n_identity = 0
    for _ in range(200):
        item = ds[0]
        label = parse_permutation(item["messages"][1]["content"][0]["text"])
        perm = _perm_of(item)
        if label == IDENTITY:
            n_identity += 1
            assert perm == chronological_perm([3, 1, 2, 4])
    assert n_identity > 0, "identity_prior=0.155인데 200뷰에서 identity가 한 번도 안 나왔다"


@pytest.mark.parametrize("prior,expect_identity", [(1.0, True), (0.0, False)])
def test_plain_identity_prior_boundaries(tmp_path, prior, expect_identity):
    records = [{"images": ["a.jpg"] * 4, "caption": "c", "events": None,
                "rank": [2, 4, 1, 3], "no_ordering": False}]
    ds = _plain_ds(tmp_path, records, identity_prior=prior)
    labels = [parse_permutation(ds[0]["messages"][1]["content"][0]["text"]) for _ in range(30)]
    assert all((lab == IDENTITY) is expect_identity for lab in labels)


def test_plain_rejects_bad_identity_prior(tmp_path):
    p = tmp_path / "sft.jsonl"
    _write_jsonl(p, _records())
    with pytest.raises(AssertionError):
        VLSFTDataset(str(p), data_dir=".", style="plain", identity_prior=1.5)


def test_old_styles_keep_no_ordering_special_case(tmp_path):
    """구 스타일(mid)은 건드리지 않았음을 명시 — 구 체크포인트 재현 보장."""
    records = [{"images": ["a.jpg"] * 4, "caption": "c", "events": None,
                "rank": [1, 2, 3, 4], "no_ordering": True}]
    p = tmp_path / "sft.jsonl"
    _write_jsonl(p, records)
    ds = VLSFTDataset(str(p), data_dir=".", style="mid", crop=False, seed=7)
    ds.load_images = lambda _record, _perm: ["im1", "im2", "im3", "im4"]
    target = ds[0]["messages"][1]["content"][0]["text"]
    assert "UNORDERABLE" in target
    assert parse_permutation(target) == IDENTITY
