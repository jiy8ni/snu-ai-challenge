"""VLM SFT용 구조화 CoT 타깃 자동 생성 (PLAN.md §8-3b).

train 라벨(rank)과 캡션 이벤트 절 분해를 규칙 기반으로 결합한다.
프레임별 시각 서술은 라벨에 없으므로 지어내지 않는다 — 이벤트 절이 4개일 때만
시간 위치와 이벤트를 1:1로 대응시켜 표기한다.

핵심: `build_cot`는 순수 함수다. 학습 시 순열 증강으로 입력 프레임을 재배치하면
rank 라벨을 `shuffle_rank_label`로 변환한 뒤 이 함수로 CoT를 **재생성**한다
(JSONL에 저장된 정적 cot는 검수용이지 증강 후에는 그대로 쓰면 안 된다).

사용:
  python -m src.train.cot_target    # -> outputs/sft_train.jsonl, outputs/sft_val.jsonl
"""

import json
import os

import pandas as pd

from src.data.loader import frame_paths, load_paths, load_split
from src.preprocess.caption_events import split_events
from src.utils.permutation import IDENTITY, is_valid_permutation, rank_to_order

_ORDINAL = {1: "1st", 2: "2nd", 3: "3rd", 4: "4th"}

UNORDERABLE_REASON = (
    "The four frames cannot be uniquely aligned to the caption's timeline: "
    "the caption does not determine the relative order of some frames "
    "(e.g., multiple frames depict the same described event)."
)


def build_cot(events, rank, no_ordering):
    """이벤트 절 목록 + rank 라벨 + 정렬 가능 여부 -> CoT 타깃 텍스트."""
    lines = ["Step 1 - Events described in the caption, in temporal order:"]
    for i, ev in enumerate(events, 1):
        lines.append(f"  E{i}. {ev}")

    if no_ordering:
        lines.append(f"Step 2 - {UNORDERABLE_REASON}")
        lines.append("Conclusion: UNORDERABLE")
        lines.append(f"Answer: {IDENTITY}")
        return "\n".join(lines)

    assert is_valid_permutation(rank), f"invalid rank: {rank}"
    order = rank_to_order(rank)  # order[k] = 시간순 k번째 프레임 번호
    lines.append("Step 2 - Matching frames to the timeline:")
    for k, frame_num in enumerate(order, 1):
        ev_note = f" (E{k})" if len(events) == 4 else ""
        lines.append(f"  {_ORDINAL[k]} in time: Frame {frame_num}{ev_note}")

    pos = ", ".join(f"Frame {i + 1} -> {_ORDINAL[r]}" for i, r in enumerate(rank))
    lines.append(f"Step 3 - Temporal position of each frame: {pos}")
    lines.append("Conclusion: ORDERABLE")
    lines.append(f"Answer: {rank}")
    return "\n".join(lines)


def build_records(df, split_df, data_dir):
    """샘플별 SFT 레코드 목록 생성. images는 data_dir 기준 상대 경로."""
    from src.utils.permutation import parse_answer_column

    merged = df.merge(split_df[["Id", "fold"]], on="Id")
    records = []
    for _, r in merged.iterrows():
        rank = parse_answer_column(r["Answer"])
        no_ordering = bool(r["No_ordering"])
        events = split_events(r["Sentence"]) or [r["Sentence"]]
        images = [os.path.relpath(p, data_dir) for p in frame_paths(r)]
        records.append(
            {
                "Id": r["Id"],
                "fold": r["fold"],
                "images": images,
                "caption": r["Sentence"],
                "events": events,
                "rank": rank,
                "no_ordering": no_ordering,
                "cot": build_cot(events, rank, no_ordering),
            }
        )
    return records


def main():
    paths = load_paths()
    df = load_split("train")
    split_df = pd.read_csv(os.path.join(paths["outputs_dir"], "split.csv"))
    records = build_records(df, split_df, paths["data_dir"])

    counts = {}
    for fold in ("train", "val"):
        out = os.path.join(paths["outputs_dir"], f"sft_{fold}.jsonl")
        subset = [r for r in records if r["fold"] == fold]
        with open(out, "w", encoding="utf-8") as f:
            for r in subset:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        counts[fold] = len(subset)
        print(f"saved: {out} ({len(subset)} records)")

    assert sum(counts.values()) == len(df)


if __name__ == "__main__":
    main()
