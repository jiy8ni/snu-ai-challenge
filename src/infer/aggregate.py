"""permutation-TTA 원시 생성 결과 집계 (Track B 추론 2단계).

입력: predict.py가 남긴 raw jsonl — 행마다 {Id, perm(0-indexed 재배치), text(모델 출력)}.
샘플당 여러 행(TTA 순열별 1행)을 다음 규칙으로 하나의 rank로 집계한다:

  1. parse_permutation으로 rank 추출 실패 행은 버림 (전부 실패 시 [1,2,3,4] fallback)
  2. UNORDERABLE 출력의 Answer [1,2,3,4]는 관례값이므로 **역변환하지 않고**
     "unorderable 투표"로만 계수한다. 과반이면 [1,2,3,4].
  3. orderable 투표는 unshuffle_rank_label로 원 좌표 rank로 되돌린 뒤
     완전일치 최빈값(mode) 투표. EM이 순열 단위 지표이므로 좌표별 평균보다 정합.
  4. 분산 게이트(기본 on): 최빈 표수가 1(합의 전무)이면 [1,2,3,4]. identity 금지는
     기본 off — 2026-07-05 val 스윕에서 금지가 -4.2pp, 분산 게이트가 +1pp
     (0.3924 -> 0.4439). 구 정책은 --ban-identity / --no-disperse-gate로 복원.
  5. 동률은 Borda(투표 평균 rank에 최소 제곱거리) -> 사전순으로 결정적 tie-break.

사용:
  python -m src.infer.aggregate --raw outputs/raw_test.jsonl --out outputs/pred_test.csv
  python -m src.infer.aggregate --raw outputs/raw_test.jsonl --submission outputs/submission.csv
"""

import argparse
import json
from collections import Counter, defaultdict

import numpy as np
import pandas as pd

from src.train.targets import is_unorderable_output
from src.utils.permutation import IDENTITY, parse_permutation, unshuffle_rank_label


def parse_vote(text, perm):
    """모델 출력 1건 -> ('unorderable'|'orderable'|'fail', 원좌표 rank|None)."""
    rank = parse_permutation(text)
    if rank is None:
        return "fail", None
    if is_unorderable_output(text):
        return "unorderable", None
    return "orderable", unshuffle_rank_label(rank, perm)


def aggregate_votes(votes, ban_identity=False, disperse_gate=True):
    """votes: (kind, rank) 리스트 -> (최종 rank, 결정 사유).

    기본 정책은 2026-07-05 Track B val 스윕으로 확정 (0.3924 -> 0.4439):
      - ban_identity=False: identity 금지는 -4.2pp 해악 (게이트 약할 때 역효과
        — Track A reports/preprocessing.md 관측과 동일)
      - disperse_gate=True: 투표 간 합의가 전무(최빈 표수 1)하면 모델이 순서를
        못 찾는 샘플로 보고 identity. 유효 표가 1표뿐인 경우도 이 게이트에 걸린다.
    """
    valid = [(k, r) for k, r in votes if k != "fail"]
    if not valid:
        return list(IDENTITY), "parse_fail"

    n_unord = sum(1 for k, _ in valid if k == "unorderable")
    if 2 * n_unord > len(valid):
        return list(IDENTITY), "unorderable_majority"

    ranks = [r for k, r in valid if k == "orderable"]
    if not ranks:
        return list(IDENTITY), "unorderable_only"

    counts = Counter(tuple(r) for r in ranks)
    if ban_identity:
        counts = Counter({r: c for r, c in counts.items() if list(r) != IDENTITY})
        if not counts:
            return list(IDENTITY), "all_identity_votes"

    top = max(counts.values())
    if disperse_gate and top == 1:
        return list(IDENTITY), "disperse_gate"

    tied = sorted(r for r, c in counts.items() if c == top)
    if len(tied) == 1:
        return list(tied[0]), "mode"

    mean_rank = np.mean(ranks, axis=0)
    best = min(tied, key=lambda r: (float(((np.array(r) - mean_rank) ** 2).sum()), r))
    return list(best), "borda_tie"


def aggregate_file(raw_path, ban_identity=False, disperse_gate=True):
    """raw jsonl -> (pred_by_id, 통계 dict)."""
    by_id = defaultdict(list)
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            by_id[rec["Id"]].append(parse_vote(rec["text"], rec["perm"]))

    preds, reasons = {}, Counter()
    n_votes = n_fail = 0
    for sid, votes in by_id.items():
        n_votes += len(votes)
        n_fail += sum(1 for k, _ in votes if k == "fail")
        preds[sid], reason = aggregate_votes(votes, ban_identity, disperse_gate)
        reasons[reason] += 1

    stats = {
        "n_samples": len(by_id),
        "votes_per_sample": n_votes / max(len(by_id), 1),
        "parse_fail_rate": n_fail / max(n_votes, 1),
        "identity_rate": sum(p == list(IDENTITY) for p in preds.values()) / max(len(preds), 1),
        "reasons": dict(reasons),
    }
    return preds, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True)
    ap.add_argument("--out", default=None, help="예측 CSV (Id, Answer)")
    ap.add_argument("--submission", default=None, help="검증 포함 제출 CSV 생성 (test 전용)")
    ap.add_argument("--ban-identity", action="store_true", help="구 정책: identity 후보 금지")
    ap.add_argument("--no-disperse-gate", action="store_true", help="구 정책: 분산 게이트 비활성")
    args = ap.parse_args()

    preds, stats = aggregate_file(
        args.raw, ban_identity=args.ban_identity, disperse_gate=not args.no_disperse_gate
    )
    print(json.dumps(stats, indent=2, ensure_ascii=False))

    if args.out:
        from src.train.targets import format_answer

        pd.DataFrame(
            {"Id": list(preds), "Answer": [format_answer(r) for r in preds.values()]}
        ).to_csv(args.out, index=False)
        print(f"saved: {args.out}")
    if args.submission:
        from src.infer.submission import build_submission

        build_submission(preds, args.submission)


if __name__ == "__main__":
    main()
