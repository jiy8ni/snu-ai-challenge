"""raw jsonl 투표 프로파일 비교 + val 보정 LB 추정 — val↔LB 괴리 진단 (Phase 0).

val과 test의 TTA 투표 통계(합의도·parse fail·identity·게이트 발동)를 나란히 놓고,
val에서 추정한 P(정답 | 합의도)를 test의 합의도 분포에 적용해 "예상 LB"를 계산한다.

  예상 LB ≈ 실제 LB(0.71) 이고 val EM(0.44)과 갈라진다면
  → 괴리는 지표 차이·버그가 아니라 "test가 val보다 쉬운 분포"라는 양성 결론.

test의 정답은 전혀 쓰지 않는다 — 자기 예측의 합의도 분포와 train-fold 기반 보정만
사용하므로 test EDA 금지 규정(docs/rules.md)에 저촉되지 않는다.

--similarity(기본 on)는 val 각 샘플의 최근접 train-fold 캡션 TF-IDF 유사도와 EM의
관계를 본다: 유사도 상위 구간에서 EM이 뚜렷이 높으면, 그룹 분리 split(근접 중복을
val에서 완전 제거)이 val을 구조적으로 어렵게 만든 것 — test에 train 근접 중복이
남아 있다면 LB가 val보다 높게 나오는 기제가 된다.

사용 (Colab, A1·A3 실행 후 GPU 불필요):
  python -m src.eval.vote_stats \
      --raw-val  /content/drive/MyDrive/snuai/raw_val_0705.jsonl \
      --raw-test /content/drive/MyDrive/snuai/raw_test_0705.jsonl \
      --json /content/outputs/vote_stats.json
"""

import argparse
import json
import os
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from src.infer.aggregate import aggregate_votes, parse_vote
from src.utils.permutation import IDENTITY


def load_vote_df(raw_path, ban_identity=False, disperse_gate=True):
    """raw jsonl -> 샘플 단위 DataFrame (합의도·투표 구성·최종 pred·사유)."""
    by_id = defaultdict(list)
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            by_id[rec["Id"]].append(parse_vote(rec["text"], rec["perm"]))

    rows = []
    for sid, votes in by_id.items():
        ranks = [r for k, r in votes if k == "orderable"]
        counts = Counter(tuple(r) for r in ranks)
        pred, reason = aggregate_votes(votes, ban_identity, disperse_gate)
        rows.append(
            {
                "Id": sid,
                "n_votes": len(votes),
                "n_fail": sum(k == "fail" for k, _ in votes),
                "n_unorderable": sum(k == "unorderable" for k, _ in votes),
                # 합의도 = orderable 투표 중 최빈 rank의 표수 (orderable 투표 없으면 0)
                "consensus": max(counts.values()) if counts else 0,
                "identity_votes": sum(list(r) == IDENTITY for r in ranks),
                "pred": pred,
                "pred_identity": pred == list(IDENTITY),
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def profile(df):
    """샘플 DataFrame -> 투표 수준·샘플 수준 요약 통계 dict."""
    total = int(df["n_votes"].sum())
    n_orderable = total - int(df["n_fail"].sum()) - int(df["n_unorderable"].sum())
    consensus_dist = (
        df["consensus"].value_counts(normalize=True).sort_index().round(4).to_dict()
    )
    return {
        "n_samples": len(df),
        "votes_per_sample": round(total / max(len(df), 1), 2),
        "parse_fail_rate": round(float(df["n_fail"].sum()) / max(total, 1), 4),
        "unorderable_vote_rate": round(float(df["n_unorderable"].sum()) / max(total, 1), 4),
        "identity_vote_rate": round(float(df["identity_votes"].sum()) / max(n_orderable, 1), 4),
        "pred_identity_rate": round(float(df["pred_identity"].mean()), 4),
        "consensus_dist": consensus_dist,
        "reasons": {k: int(v) for k, v in Counter(df["reason"]).items()},
    }


def attach_truth(df, fold="val"):
    """val 샘플 DataFrame에 정답(Answer, No_ordering)과 em 컬럼을 붙인다."""
    from src.data.loader import load_paths, load_split
    from src.utils.permutation import parse_answer_column

    truth = load_split("train")
    split_df = pd.read_csv(os.path.join(load_paths()["outputs_dir"], "split.csv"))
    truth = truth.merge(split_df[["Id", "fold"]], on="Id")
    truth = truth[truth["fold"] == fold][["Id", "Sentence", "Answer", "No_ordering"]]

    out = df.merge(truth, on="Id", validate="one_to_one")
    assert len(out) == len(df), f"truth 매칭 실패: {len(df)} -> {len(out)}"
    out["em"] = [
        pred == parse_answer_column(ans) for pred, ans in zip(out["pred"], out["Answer"])
    ]
    return out


def em_by(df, col):
    """컬럼별 EM 분해 표 (n, em[, no_ordering 비율])."""
    agg = {"n": ("em", "size"), "em": ("em", "mean")}
    if "No_ordering" in df.columns:
        agg["no_ordering_rate"] = ("No_ordering", "mean")
    # observed=False: sim_bin처럼 카테고리형일 때 빈 구간도 표에 유지 (pandas 기본값 변경 대비 명시)
    return df.groupby(col, observed=False).agg(**agg).round(4)


def project_lb(val_df, test_df):
    """val의 P(정답|합의도)를 test 합의도 분포에 적용한 기대 EM (예상 LB).

    합의도 수준이 val에 없는 test 구간은 val 전체 EM으로 폴백한다.
    Public LB는 test의 70% 무작위 표본이므로 기대값은 그대로 적용 가능.
    """
    em_map = val_df.groupby("consensus")["em"].mean()
    fallback = float(val_df["em"].mean())
    dist = test_df["consensus"].value_counts(normalize=True)
    proj = sum(p * float(em_map.get(k, fallback)) for k, p in dist.items())
    return {
        "projected_lb": round(float(proj), 4),
        "val_em": round(fallback, 4),
        "em_by_consensus_val": em_map.round(4).to_dict(),
        "consensus_dist_test": dist.sort_index().round(4).to_dict(),
    }


def nearest_train_similarity(val_df):
    """val 각 샘플의 최근접 train-fold 캡션 TF-IDF 코사인 유사도 컬럼 추가."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    from src.data.loader import load_paths, load_split

    truth = load_split("train")
    split_df = pd.read_csv(os.path.join(load_paths()["outputs_dir"], "split.csv"))
    truth = truth.merge(split_df[["Id", "fold"]], on="Id")
    train_caps = truth[truth["fold"] == "train"]["Sentence"].tolist()

    vec = TfidfVectorizer(min_df=1).fit(train_caps)
    nn = NearestNeighbors(n_neighbors=1, metric="cosine").fit(vec.transform(train_caps))
    dist, _ = nn.kneighbors(vec.transform(val_df["Sentence"].tolist()))
    out = val_df.copy()
    out["nn_train_sim"] = 1.0 - dist[:, 0]
    out["sim_bin"] = pd.cut(out["nn_train_sim"], [0, 0.2, 0.4, 0.6, 0.8, 1.01], right=False)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-val", required=True, help="val fold raw jsonl (정답 보유)")
    ap.add_argument("--raw-test", default=None, help="test raw jsonl (선택)")
    ap.add_argument("--no-similarity", action="store_true", help="TF-IDF 유사도 분석 생략")
    ap.add_argument("--json", default=None, help="요약 JSON 저장 경로")
    args = ap.parse_args()

    report = {}

    val_df = attach_truth(load_vote_df(args.raw_val))
    report["val_profile"] = profile(val_df)
    print("=== val 투표 프로파일 ===")
    print(json.dumps(report["val_profile"], indent=2, ensure_ascii=False))
    print("\n--- val EM 분해: 합의도별 ---")
    print(em_by(val_df, "consensus").to_string())
    print("\n--- val EM 분해: 집계 사유별 ---")
    print(em_by(val_df, "reason").to_string())

    if args.raw_test:
        test_df = load_vote_df(args.raw_test)
        report["test_profile"] = profile(test_df)
        print("\n=== test 투표 프로파일 ===")
        print(json.dumps(report["test_profile"], indent=2, ensure_ascii=False))

        report["projection"] = project_lb(val_df, test_df)
        print("\n=== val 보정 예상 LB (P(정답|합의도) 전이 가정) ===")
        print(json.dumps(report["projection"], indent=2, ensure_ascii=False))

    if not args.no_similarity:
        val_df = nearest_train_similarity(val_df)
        sim_table = em_by(val_df, "sim_bin")
        corr = float(np.corrcoef(val_df["nn_train_sim"], val_df["em"].astype(float))[0, 1])
        report["similarity"] = {
            "corr_sim_em": round(corr, 4),
            "em_by_sim_bin": {str(k): v for k, v in sim_table["em"].to_dict().items()},
        }
        print("\n=== val: 최근접 train 캡션 유사도 vs EM ===")
        print(sim_table.to_string())
        print(f"corr(nn_train_sim, em) = {corr:.4f}")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False, default=str)
        print(f"\nsaved: {args.json}")


if __name__ == "__main__":
    main()
