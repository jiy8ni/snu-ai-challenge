"""평가 하네스 (Track A/B 공용): Exact Match 중심 + 진단용 보조 지표.

공식 지표는 EM뿐이다 (docs/rules.md §2). Kendall τ·게이트 P/R·identity 비율은
의사결정이 아니라 오류 분석용 보조 지표로만 쓴다.

사용 (val 예측 CSV 채점):
  python -m src.eval.em --pred outputs/track_a/val_pred.csv
"""

import argparse
import json

import numpy as np
import pandas as pd

from src.utils.permutation import IDENTITY, is_valid_permutation, parse_answer_column


def kendall_tau(a, b):
    """두 rank 벡터(길이 4)의 Kendall τ. 완전일치 1.0, 완전역순 -1.0."""
    n = len(a)
    concordant = sum(
        1 if (a[i] - a[j]) * (b[i] - b[j]) > 0 else -1
        for i in range(n)
        for j in range(i + 1, n)
    )
    return concordant / (n * (n - 1) / 2)


def evaluate(pred_by_id, truth_df):
    """pred_by_id: {Id: rank list}. truth_df: Id, Answer[, No_ordering] 컬럼.

    반환: 지표 dict. EM은 전체/orderable/no_ordering으로 분해.
    """
    missing = set(truth_df["Id"]) - set(pred_by_id)
    assert not missing, f"예측 누락 {len(missing)}건: {sorted(missing)[:5]}..."

    rows = []
    for _, r in truth_df.iterrows():
        pred = pred_by_id[r["Id"]]
        assert is_valid_permutation(pred), f"{r['Id']}: invalid pred {pred}"
        true = parse_answer_column(r["Answer"]) if isinstance(r["Answer"], str) else list(r["Answer"])
        rows.append(
            {
                "em": pred == true,
                "tau": kendall_tau(pred, true),
                "pred_identity": pred == IDENTITY,
                "no_ordering": bool(r["No_ordering"]) if "No_ordering" in truth_df.columns else None,
            }
        )
    d = pd.DataFrame(rows)

    out = {
        "n": len(d),
        "em": float(d["em"].mean()),
        "kendall_tau": float(d["tau"].mean()),
        "identity_rate": float(d["pred_identity"].mean()),
    }
    if d["no_ordering"].notna().all():
        no = d[d["no_ordering"]]
        yes = d[~d["no_ordering"]]
        out["em_orderable"] = float(yes["em"].mean()) if len(yes) else float("nan")
        out["em_no_ordering"] = float(no["em"].mean()) if len(no) else float("nan")
        # 게이트 진단: "identity 예측"을 UNORDERABLE 판정으로 간주
        tp = int((d["pred_identity"] & d["no_ordering"]).sum())
        fp = int((d["pred_identity"] & ~d["no_ordering"]).sum())
        fn = int((~d["pred_identity"] & d["no_ordering"]).sum())
        out["gate_precision"] = tp / (tp + fp) if tp + fp else float("nan")
        out["gate_recall"] = tp / (tp + fn) if tp + fn else float("nan")
    return out


def load_pred_csv(path):
    """Id, Answer 형식의 예측 CSV -> {Id: rank list}."""
    df = pd.read_csv(path)
    return {r["Id"]: parse_answer_column(r["Answer"]) for _, r in df.iterrows()}


def main():
    from src.data.loader import load_paths, load_split

    ap = argparse.ArgumentParser()
    ap.add_argument("--pred", required=True, help="예측 CSV (Id, Answer)")
    ap.add_argument("--fold", default="val", help="채점 대상 fold (split.csv 기준)")
    args = ap.parse_args()

    paths = load_paths()
    truth = load_split("train")
    split_df = pd.read_csv(f"{paths['outputs_dir']}/split.csv")
    truth = truth.merge(split_df[["Id", "fold"]], on="Id")
    truth = truth[truth["fold"] == args.fold]

    preds = load_pred_csv(args.pred)
    preds = {k: v for k, v in preds.items() if k in set(truth["Id"])}
    metrics = evaluate(preds, truth)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
