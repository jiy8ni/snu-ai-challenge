"""생성-투표 + LL 스코어링 융합 결정 (단일 체크포인트의 두 추론 경로 결합).

근거 (reports/val_gap.md, 3B 실측): 투표 합의도는 잘 보정된 신뢰도 지표다
(4/4 합의 EM 0.79 vs 1표 0.21). 고합의 구간은 투표를 그대로 쓰고, 투표가
약한 저합의 구간만 LL 전수 스코어링으로 교체하는 것이 융합의 골자.
UNORDERABLE margin 판별(ll_score)은 LL 경로에서만 작동한다.

정책 (전부 동일 모델·동일 raw 입력 — 앙상블 아님, 추론 전략 조합):
  vote : 생성-투표 단독 (aggregate.py 기본 정책) — 기준선
  ll   : LL 스코어링 단독 (ll_score.decide_all) — 기준선
  h1   : 합의도 >= k 면 투표 채택, 미만이면 LL 24후보 전수 판정
  h2   : 합의도 >= k 면 투표 채택, 미만이면 투표된 후보들로 제한한 LL 재순위
         (투표의 사전 선별 + LL의 정밀 비교. 후보가 없으면 전수로 폴백)
  --ban-identity(H3)는 LL 판정 경로의 orderable 후보에서 identity를 제외한다
  (margin 게이트의 UNORDERABLE->identity 출력은 유지 — findings.md §1-2 재검증).

사용:
  python -m src.infer.fuse sweep --raw outputs/raw_val_0706.jsonl \
      --ll outputs/ll_val_0706.jsonl --fold val
  python -m src.infer.fuse decide --raw RAW --ll LL --policy h1 --k 3 --margin M \
      [--ban-identity] [--out pred.csv | --submission submission.csv]
"""

import argparse
import json
import os
from collections import Counter, defaultdict

from src.infer.aggregate import aggregate_votes, parse_vote
from src.infer.ll_score import aggregate_ll, decide_one, load_ll

POLICIES = ("vote", "ll", "h1", "h2")


def load_votes(raw_path):
    """raw jsonl -> {Id: [(kind, 원좌표 rank|None), ...]}."""
    by_id = defaultdict(list)
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            by_id[rec["Id"]].append(parse_vote(rec["text"], rec["perm"]))
    return by_id


def consensus_of(votes):
    """orderable 투표 최빈 rank의 표수 (orderable 투표가 없으면 0)."""
    counts = Counter(tuple(r) for k, r in votes if k == "orderable")
    return max(counts.values()) if counts else 0


def fuse_all(votes_by_id, ll_by_id, policy, k=3, margin=0.0,
             length_norm=True, ban_identity=False):
    """{Id: 최종 rank}. 투표에만 있고 LL이 없는 샘플은 투표 결과로 폴백."""
    assert policy in POLICIES, f"unknown policy: {policy}"
    if policy == "ll":
        return {
            sid: decide_one(aggregate_ll(views, length_norm), margin, ban_identity)
            for sid, views in ll_by_id.items()
        }

    preds = {}
    for sid, votes in votes_by_id.items():
        pred_vote, _ = aggregate_votes(votes)  # 투표 경로는 확정 기본 정책 유지
        if policy == "vote" or sid not in ll_by_id:
            preds[sid] = pred_vote
            continue
        if consensus_of(votes) >= k:
            preds[sid] = pred_vote
            continue
        scores = aggregate_ll(ll_by_id[sid], length_norm)
        allowed = None
        if policy == "h2":
            voted = [r for kind, r in votes if kind == "orderable"]
            allowed = voted or None  # 투표 후보 전무 시 전수 폴백
        preds[sid] = decide_one(scores, margin, ban_identity, allowed=allowed)
    return preds


def _load_truth(fold):
    import pandas as pd

    from src.data.loader import load_paths, load_split

    truth = load_split("train")
    split_df = pd.read_csv(os.path.join(load_paths()["outputs_dir"], "split.csv"))
    truth = truth.merge(split_df[["Id", "fold"]], on="Id")
    return truth[truth["fold"] == fold]


def cmd_sweep(args):
    import numpy as np
    import pandas as pd

    from src.eval.em import evaluate

    votes_by_id = load_votes(args.raw)
    ll_by_id = load_ll(args.ll)
    truth = _load_truth(args.fold)
    n_fold = len(truth)
    covered = set(votes_by_id) & set(truth["Id"])
    truth = truth[truth["Id"].isin(covered)]
    if len(truth) < n_fold:
        print(f"[경고] fold {args.fold} {n_fold}개 중 {len(truth)}개만 raw에 존재")
    print(f"sweep 대상: {len(truth)}샘플 (LL 보유 {sum(t in ll_by_id for t in covered)}개), "
          f"margin [{args.margin_min}, {args.margin_max}] step {args.margin_step}")

    margins = [round(float(m), 4)
               for m in np.arange(args.margin_min, args.margin_max + 1e-9, args.margin_step)]
    configs = [("vote", None, False, [0.0])]
    for ban in (False, True):
        configs.append(("ll", None, ban, margins))
        for policy in ("h1", "h2"):
            for k in args.k_grid:
                configs.append((policy, k, ban, margins))

    rows = []
    for policy, k, ban, mgrid in configs:
        best = None
        for margin in mgrid:
            preds = fuse_all(votes_by_id, ll_by_id, policy, k=k or 0, margin=margin,
                             length_norm=not args.no_length_norm, ban_identity=ban)
            m = evaluate(preds, truth)
            if best is None or m["em"] > best[1]["em"]:
                best = (margin, m)
        margin, m = best
        rows.append({"policy": policy, "k": k, "ban_identity": ban,
                     "best_margin": margin if policy != "vote" else None,
                     "em": m["em"], "em_orderable": m.get("em_orderable"),
                     "em_no_ordering": m.get("em_no_ordering"),
                     "identity_rate": m["identity_rate"]})

    table = pd.DataFrame(rows).sort_values("em", ascending=False)
    print(table.round(4).to_string(index=False))
    top = table.iloc[0]
    print(f"\nbest: policy={top['policy']} k={top['k']} ban={top['ban_identity']} "
          f"margin={top['best_margin']} em={top['em']:.4f}")


def cmd_decide(args):
    import pandas as pd

    from src.train.targets import format_answer

    preds = fuse_all(
        load_votes(args.raw), load_ll(args.ll), args.policy, k=args.k,
        margin=args.margin, length_norm=not args.no_length_norm,
        ban_identity=args.ban_identity,
    )
    from src.utils.permutation import IDENTITY

    print(f"n={len(preds)}, identity_rate="
          f"{sum(p == list(IDENTITY) for p in preds.values()) / max(len(preds), 1):.4f}")
    if args.out:
        pd.DataFrame(
            {"Id": list(preds), "Answer": [format_answer(r) for r in preds.values()]}
        ).to_csv(args.out, index=False)
        print(f"saved: {args.out}")
    if args.submission:
        from src.infer.submission import build_submission

        build_submission(preds, args.submission)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sw = sub.add_parser("sweep", help="val 정답으로 정책×k×margin×ban 그리드 EM")
    sw.add_argument("--raw", required=True, help="생성-투표 raw jsonl")
    sw.add_argument("--ll", required=True, help="LL 스코어 raw jsonl")
    sw.add_argument("--fold", default="val")
    sw.add_argument("--k-grid", type=int, nargs="+", default=[2, 3, 4])
    sw.add_argument("--margin-min", type=float, default=-0.5)
    sw.add_argument("--margin-max", type=float, default=0.5)
    sw.add_argument("--margin-step", type=float, default=0.025)
    sw.add_argument("--no-length-norm", action="store_true")
    sw.set_defaults(fn=cmd_sweep)

    de = sub.add_parser("decide", help="확정 정책으로 최종 rank CSV/submission")
    de.add_argument("--raw", required=True)
    de.add_argument("--ll", required=True)
    de.add_argument("--policy", required=True, choices=POLICIES)
    de.add_argument("--k", type=int, default=3)
    de.add_argument("--margin", type=float, required=True)
    de.add_argument("--no-length-norm", action="store_true")
    de.add_argument("--ban-identity", action="store_true")
    de.add_argument("--out", default=None)
    de.add_argument("--submission", default=None)
    de.set_defaults(fn=cmd_decide)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
