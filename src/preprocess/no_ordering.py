"""No_ordering(정렬 불가) 판별 파이프라인.

train의 15.5%가 No_ordering=True(정답 [1,2,3,4] 고정)이고 orderable 샘플에는
identity가 0건 -> 이 이진 판별이 점수에 직결된다 (docs/findings.md).

이 모듈의 분류기는 분석·의사라벨 검증용이다. 최종 추론 파이프라인에는 넣지 않고
(앙상블 금지), 판별 능력이 확인되면 VLM SFT의 CoT 타깃에 UNORDERABLE 분기로 통합한다.

사용:
  python -m src.preprocess.no_ordering --features   # 피처 추출 -> outputs/no_ordering_features.csv
  python -m src.preprocess.no_ordering --evaluate   # 5-fold CV AUC/F1 리포트
"""

import argparse
import os

import numpy as np
import pandas as pd
from tqdm import tqdm

from src.data.loader import frame_paths, load_paths, load_split
from src.preprocess.caption_events import n_events
from src.preprocess.frame_quality import sample_features

FEATURE_COLS = [
    "n_events",
    "events_deficit",      # max(0, 4 - 이벤트 수): 이벤트가 프레임보다 적은 정도
    "caption_words",
    "min_phash_dist",      # 가장 가까운 프레임 쌍의 phash 거리 (중복일수록 작음)
    "mean_phash_dist",
    "n_dup_pairs",         # phash 중복 쌍 개수
    "n_low_info",          # 검은/단색 프레임 수
    "min_std",
    "min_entropy",
]


def build_features(split="train", use_images=True):
    df = load_split(split)
    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc=f"features:{split}"):
        ne = n_events(r["Sentence"])
        feat = {
            "Id": r["Id"],
            "n_events": ne,
            "events_deficit": max(0, 4 - ne),
            "caption_words": len(r["Sentence"].split()),
        }
        if use_images:
            sf = sample_features(frame_paths(r))
            feat.update(
                min_phash_dist=sf["min_phash_dist"],
                mean_phash_dist=float(np.mean(sf["phash_dists"])),
                n_dup_pairs=sf["n_dup_pairs"],
                n_low_info=sf["n_low_info"],
                min_std=sf["min_std"],
                min_entropy=sf["min_entropy"],
                any_letterbox=sf["any_letterbox"],
            )
        if "No_ordering" in df.columns:
            feat["No_ordering"] = bool(r["No_ordering"])
        rows.append(feat)
    return pd.DataFrame(rows)


def evaluate(features_df, n_splits=5, seed=42):
    """5-fold CV로 로지스틱/GBM의 AUC·F1을 측정해 dict로 반환."""
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import f1_score, roc_auc_score
    from sklearn.model_selection import StratifiedKFold
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = features_df[FEATURE_COLS].to_numpy(dtype=float)
    y = features_df["No_ordering"].to_numpy(dtype=int)
    models = {
        "logistic": lambda: make_pipeline(
            StandardScaler(), LogisticRegression(max_iter=1000, class_weight="balanced")
        ),
        "gbm": lambda: HistGradientBoostingClassifier(random_state=seed),
    }
    results = {}
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    for name, factory in models.items():
        aucs, f1s = [], []
        for tr_idx, va_idx in skf.split(X, y):
            model = factory()
            model.fit(X[tr_idx], y[tr_idx])
            prob = model.predict_proba(X[va_idx])[:, 1]
            aucs.append(roc_auc_score(y[va_idx], prob))
            f1s.append(f1_score(y[va_idx], prob >= 0.5))
        results[name] = {
            "auc_mean": float(np.mean(aucs)),
            "auc_std": float(np.std(aucs)),
            "f1_mean": float(np.mean(f1s)),
        }
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", action="store_true", help="피처 추출 후 CSV 저장")
    ap.add_argument("--evaluate", action="store_true", help="저장된 피처로 CV 평가")
    ap.add_argument("--split", default="train")
    args = ap.parse_args()

    paths = load_paths()
    out_csv = os.path.join(paths["outputs_dir"], f"no_ordering_features_{args.split}.csv")

    if args.features:
        feats = build_features(args.split)
        os.makedirs(paths["outputs_dir"], exist_ok=True)
        feats.to_csv(out_csv, index=False)
        print(f"saved: {out_csv} ({len(feats)} rows)")

    if args.evaluate:
        feats = pd.read_csv(out_csv)
        results = evaluate(feats)
        for name, r in results.items():
            print(f"{name}: AUC {r['auc_mean']:.4f} (+/- {r['auc_std']:.4f}), F1 {r['f1_mean']:.4f}")


if __name__ == "__main__":
    main()
