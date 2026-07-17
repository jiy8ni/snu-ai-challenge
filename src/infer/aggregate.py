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
import os
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


def aggregate_votes(votes, ban_identity=False, disperse_gate=True,
                    disperse_top=1, identity_quota=None):
    """votes: (kind, rank) 리스트 -> (최종 rank, 결정 사유).

    기본 정책은 2026-07-05 Track B val 스윕으로 확정 (0.3924 -> 0.4439):
      - ban_identity=False: identity 금지는 -4.2pp 해악 (게이트 약할 때 역효과
        — Track A reports/preprocessing.md 관측과 동일)
      - disperse_gate=True: 투표 간 합의가 부족(최빈 표수 <= disperse_top)하면
        모델이 순서를 못 찾는 샘플로 보고 identity. 유효 표가 1표뿐인 경우도 걸린다.
      - disperse_top: 게이트 문턱. tta4 확정값은 1. tta8은 "top==1"이 거의 안
        나와 게이트가 실종되므로 (2026-07-07 실측: 발동 70->0, em_no_ordering
        -10pp) 표 수에 맞춰 val로 재튜닝한다.
      - identity_quota: identity 표가 이 수 이상이면 최빈과 무관하게 identity
        (None=off). identity 표는 no_ordering의 강한 신호(gate_precision 0.95).
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

    if identity_quota and sum(r == list(IDENTITY) for r in ranks) >= identity_quota:
        return list(IDENTITY), "identity_quota"

    counts = Counter(tuple(r) for r in ranks)
    if ban_identity:
        counts = Counter({r: c for r, c in counts.items() if list(r) != IDENTITY})
        if not counts:
            return list(IDENTITY), "all_identity_votes"

    top = max(counts.values())
    if disperse_gate and top <= disperse_top:
        return list(IDENTITY), "disperse_gate"

    tied = sorted(r for r, c in counts.items() if c == top)
    if len(tied) == 1:
        return list(tied[0]), "mode"

    mean_rank = np.mean(ranks, axis=0)
    best = min(tied, key=lambda r: (float(((np.array(r) - mean_rank) ** 2).sum()), r))
    return list(best), "borda_tie"


def votes_per_sample_dist(by_id):
    """{표 수: 샘플 수}. 표 수가 균일해야 disperse_top이 전 샘플에 같은 의미를 갖는다."""
    return dict(sorted(Counter(len(v) for v in by_id.values()).items()))


def warn_if_inhomogeneous(by_id, label="raw"):
    """샘플별 표 수가 섞여 있으면 경고 (2026-07-15 사고: cap 뷰 혼입).

    predict.py의 재개 키는 (Id, perm, cap)이고 aggregate는 cap을 무시해 캡션-TTA
    뷰를 그냥 표로 센다. 그래서 --cap-variant 실행이 중단되면 한 파일 안에
    8표 샘플과 16표 샘플이 섞이고, 표 수에 맞춰 튜닝하는 disperse_top이
    샘플마다 다른 의미가 된다 (조용히 틀린 답). 반환값 = 균일 여부.
    """
    dist = votes_per_sample_dist(by_id)
    if len(dist) > 1:
        print(f"[경고] {label}: 샘플별 표 수가 불균일 {dist} — cap 뷰 혼입이나 "
              "중단된 predict 실행일 수 있다. disperse_top/합의도 게이트가 샘플마다 "
              "다른 의미를 가지므로 결과를 의사결정에 쓰지 말 것 "
              "(cap 분포 확인: Counter(r.get('cap',0) for r in raw))")
    return len(dist) == 1


def aggregate_file(raw_path, ban_identity=False, disperse_gate=True,
                   disperse_top=1, identity_quota=None):
    """raw jsonl -> (pred_by_id, 통계 dict)."""
    by_id = defaultdict(list)
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            by_id[rec["Id"]].append(parse_vote(rec["text"], rec["perm"]))
    warn_if_inhomogeneous(by_id, label=os.path.basename(raw_path))

    preds, reasons = {}, Counter()
    n_votes = n_fail = 0
    for sid, votes in by_id.items():
        n_votes += len(votes)
        n_fail += sum(1 for k, _ in votes if k == "fail")
        preds[sid], reason = aggregate_votes(
            votes, ban_identity, disperse_gate, disperse_top, identity_quota)
        reasons[reason] += 1

    stats = {
        "n_samples": len(by_id),
        "votes_per_sample": n_votes / max(len(by_id), 1),
        "votes_per_sample_dist": votes_per_sample_dist(by_id),
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
    ap.add_argument("--disperse-top", type=int, default=1,
                    help="게이트 문턱: 최빈 표수 <= 이 값이면 identity (tta4=1, tta8은 val 스윕)")
    ap.add_argument("--identity-quota", type=int, default=None,
                    help="identity 표가 이 수 이상이면 identity (기본 off)")
    args = ap.parse_args()

    preds, stats = aggregate_file(
        args.raw, ban_identity=args.ban_identity, disperse_gate=not args.no_disperse_gate,
        disperse_top=args.disperse_top, identity_quota=args.identity_quota,
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
