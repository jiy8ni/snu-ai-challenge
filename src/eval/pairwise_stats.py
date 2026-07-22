"""오답의 pairwise 일치쌍 분포 분석 — GRPO pairwise shaping의 λ 근거 (오프라인, CPU).

hard_cases.py가 만든 재채굴 CSV(컬럼 pred, truth, em, no_ordering)를 읽어, **오답(em=False)**의
일치쌍 수 분포를 집계한다. 목적은 "현재 모델의 오답이 pairwise 축에서 얼마나 퍼져 있느냐"를 보고
GRPO 보상의 λ(=reward_weights[1])를 정하는 것이다:

  - c=5(근접 오답) 비중이 유의(예: ≥15%)하면 shaping이 실질 그래디언트를 만든다 → λ=0.25 유지.
  - 분포가 우연 수준(c≈3)에 몰려 있으면 shaping 효용이 낮다 → λ=0.2로 보수화.

보상 코드와 동일 로직을 쓰도록 src/train/rewards.py의 pairwise_concordant/CHANCE_CONCORDANT를
그대로 import한다 (분석과 학습의 정의가 어긋나지 않게).

실행:
  python -m src.eval.pairwise_stats --cases outputs/hard_train_cases_0716.csv
"""

import argparse
import csv
import json

from src.train.rewards import CHANCE_CONCORDANT, pairwise_concordant
from src.utils.permutation import IDENTITY, parse_answer_column


def _shaping(pred, truth):
    """pairwise_reward와 동일한 값(텍스트 대신 파싱된 rank로). identity 예측 제외 + 음수 클램프."""
    if pred == IDENTITY and truth != IDENTITY:
        return 0.0
    return max(0.0, (pairwise_concordant(pred, truth) - CHANCE_CONCORDANT) / CHANCE_CONCORDANT)


def _as_bool(value):
    return str(value).strip().lower() in ("true", "1", "yes")


def analyze(rows, lambdas=(0.2, 0.25, 0.3)):
    """rows: dict 리스트(hard_cases CSV 행). 반환: 분석 결과 dict (JSON 직렬화 가능)."""
    wrong = []
    for r in rows:
        if _as_bool(r["em"]):
            continue
        pred = parse_answer_column(r["pred"])
        truth = parse_answer_column(r["truth"])
        wrong.append({
            "c": pairwise_concordant(pred, truth),
            "no_ordering": _as_bool(r.get("no_ordering", False)),
            "is_identity_pred": pred == IDENTITY,
            "shaping": _shaping(pred, truth),
        })

    n_wrong = len(wrong)

    def summary(items):
        n = len(items)
        hist = {c: sum(1 for w in items if w["c"] == c) for c in range(6)}  # 오답은 c<=5
        return {
            "n": n,
            "concordant_hist": hist,
            "concordant_frac": {c: (hist[c] / n if n else 0.0) for c in range(6)},
            "near_miss_frac_c5": (hist[5] / n if n else 0.0),
            "identity_pred_frac": (sum(w["is_identity_pred"] for w in items) / n if n else 0.0),
            "mean_concordant": (sum(w["c"] for w in items) / n if n else 0.0),
        }

    lambda_scan = {}
    for lam in lambdas:
        mean_shaping = (sum(w["shaping"] for w in wrong) / n_wrong if n_wrong else 0.0)
        # 완전일치(c=6) 합성 vs 최고 오답(c=5) 합성의 갭 = EM_REWARD + λ·(1 − 2/3).
        # EM이 지배하는지 확인용 (갭이 크게 양수여야 완전일치가 근접오답을 항상 이긴다).
        best_wrong_shaping = max((w["shaping"] for w in wrong), default=0.0)
        lambda_scan[f"{lam}"] = {
            "mean_wrong_shaping": mean_shaping,
            "correct_vs_best_wrong_gap": 1.0 + lam * (1.0 - best_wrong_shaping),
        }

    return {
        "n_rows": len(rows),
        "n_wrong": n_wrong,
        "overall": summary(wrong),
        "no_ordering": summary([w for w in wrong if w["no_ordering"]]),
        "orderable": summary([w for w in wrong if not w["no_ordering"]]),
        "lambda_scan": lambda_scan,
        "recommendation": _recommend(summary(wrong)),
    }


def _recommend(overall):
    """c=5 근접 오답 비중으로 λ 권고 (계획 §3 판단 기준)."""
    if overall["n"] == 0:
        return "오답 없음 — shaping 불필요"
    if overall["near_miss_frac_c5"] >= 0.15:
        return "λ=0.25 유지 (c=5 근접 오답 비중 유의 → shaping이 실질 그래디언트 생성)"
    if overall["mean_concordant"] <= CHANCE_CONCORDANT:
        return "λ=0.2로 보수화 (오답이 우연 수준 이하에 몰림 → shaping 효용 낮음)"
    return "λ=0.2~0.25 (경계 — 소규모 GRPO 스모크에서 성분 곡선으로 확인 권장)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", required=True, help="hard_cases.py 산출 CSV (pred,truth,em,no_ordering)")
    args = ap.parse_args()
    with open(args.cases, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    print(json.dumps(analyze(rows), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
