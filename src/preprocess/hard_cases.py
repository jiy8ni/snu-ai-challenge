"""Build per-sample text augmentation weights from validation predictions.

Example:
  python -m src.preprocess.hard_cases \
    --pred /workspace/snuai/outputs/pred_val.csv \
    --fold val \
    --out /workspace/snuai/outputs/hard_val_cases.csv

The output can be used by VLSFTDataset via data.hard_cases_path to physically
oversample and up-weight the samples that the model currently misses.  All
weights are derived from the model's own predictions (no external API).
"""

import argparse
import os

import pandas as pd

from src.data.loader import load_paths, load_split
from src.eval.em import kendall_tau
from src.preprocess.hard_weights import prob_from_score, repeats_from_score
from src.utils.permutation import IDENTITY, is_valid_permutation, parse_answer_column


def _score_case(pred, true, no_ordering):
    """Return (exact_match, tau, hard_score) in [0, 1]."""
    if pred is None or not is_valid_permutation(pred):
        return False, -1.0, 1.0

    em = pred == true
    tau = float(kendall_tau(pred, true))
    if em:
        return True, tau, 0.0

    score = max(0.5, (1.0 - tau) / 2.0)
    if no_ordering and pred != IDENTITY:
        score = max(score, 0.9)
    if (not no_ordering) and pred == IDENTITY:
        score = max(score, 0.75)
    return False, tau, min(1.0, max(0.0, score))


def _load_predictions(path):
    df = pd.read_csv(path)
    out = {}
    for _, row in df.iterrows():
        try:
            out[row["Id"]] = parse_answer_column(row["Answer"])
        except Exception:
            out[row["Id"]] = None
    return out


def build_hard_case_table(
    pred_csv,
    fold="val",
    max_extra_repeats=4,
    easy_caption_aug_prob=0.3,
    hard_caption_aug_prob=0.9,
    only_wrong=False,
):
    paths = load_paths()
    truth = load_split("train", paths=paths)
    split_df = pd.read_csv(os.path.join(paths["outputs_dir"], "split.csv"))
    truth = truth.merge(split_df[["Id", "fold"]], on="Id")
    truth = truth[truth["fold"] == fold]
    pred_by_id = _load_predictions(pred_csv)

    rows = []
    for _, row in truth.iterrows():
        true = parse_answer_column(row["Answer"])
        pred = pred_by_id.get(row["Id"])
        no_ordering = bool(row["No_ordering"])
        em, tau, score = _score_case(pred, true, no_ordering)
        if only_wrong and em:
            continue
        # 공유 수식(hard_weights.py) — VLSFTDataset 오버레이 경로와 항상 일치.
        # 구식 --max-extra-repeats N은 max_repeats=N+1로 매핑된다.
        repeats = repeats_from_score(score, max_extra_repeats + 1)
        prob = prob_from_score(score, easy_caption_aug_prob, hard_caption_aug_prob)
        rows.append(
            {
                "Id": row["Id"],
                "fold": fold,
                "em": em,
                "tau": round(tau, 6),
                "no_ordering": no_ordering,
                "pred": str(pred) if pred is not None else "",
                "truth": str(true),
                "hard_score": round(score, 6),
                "caption_aug_repeats": repeats,
                "caption_aug_prob": round(prob, 6),
            }
        )
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="prediction CSV with Id, Answer")
    ap.add_argument("--fold", default="val", help="fold to score against split.csv")
    ap.add_argument("--out", required=True, help="output CSV path")
    ap.add_argument("--max-extra-repeats", type=int, default=4)
    ap.add_argument("--easy-caption-aug-prob", type=float, default=0.3)
    ap.add_argument("--hard-caption-aug-prob", type=float, default=0.9)
    ap.add_argument("--only-wrong", action="store_true")
    args = ap.parse_args()

    df = build_hard_case_table(
        args.pred,
        fold=args.fold,
        max_extra_repeats=args.max_extra_repeats,
        easy_caption_aug_prob=args.easy_caption_aug_prob,
        hard_caption_aug_prob=args.hard_caption_aug_prob,
        only_wrong=args.only_wrong,
    )
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"saved: {args.out} ({len(df)} rows)")
    if len(df):
        print(df["hard_score"].describe().round(4).to_string())


if __name__ == "__main__":
    main()
