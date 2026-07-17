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
  vgate: 투표가 **순서**를 정하고 LL의 identity_posterior가 **identity 베팅**만 결정.
         합의도 k를 쓰지 않아 h1/h2와 직교하다. 합집합 규칙 —
         p_identity > 문턱 **또는** 투표가 identity면 identity, 아니면 투표의 rank.
         근거: 투표 게이트는 P 0.9496/R 0.2411(정밀하나 recall 부족),
         p_identity는 AUC 0.8844로 recall을 보탠다 (문턱 0.0125에서 R 0.166->0.5894).
         단조적이라 투표 기준선의 gate recall을 낮출 수 없고 순서는 건드리지 않는다.
  --ban-identity(H3)는 LL 판정 경로의 orderable 후보에서 identity를 제외한다
  (margin 게이트의 UNORDERABLE->identity 출력은 유지 — findings.md §1-2 재검증).

sweep은 vgate를 plain vote가 아니라 **공짜 레버 기준선**(disperse_top/identity_quota —
GPU도 LL도 없이 동작)과도 비교한다. LL은 3090에서 ~1.9h가 들므로, 공짜 레버가 이득을
이미 가져간다면 LL 다리를 추가할 명분이 없다.

사용:
  python -m src.infer.fuse sweep --raw outputs/raw_val_0706.jsonl \
      --ll outputs/ll_val_0706.jsonl --fold val
  python -m src.infer.fuse decide --raw RAW --ll LL --policy vgate --pid-threshold T \
      [--out pred.csv | --submission submission.csv]
"""

import argparse
import json
import os
from collections import Counter, defaultdict

from src.infer.aggregate import aggregate_votes, parse_vote, warn_if_inhomogeneous
from src.infer.ll_score import aggregate_ll, decide_one, identity_posterior, load_ll
from src.utils.permutation import IDENTITY

POLICIES = ("vote", "ll", "h1", "h2", "vgate")


def load_votes(raw_path):
    """raw jsonl -> {Id: [(kind, 원좌표 rank|None), ...]}."""
    by_id = defaultdict(list)
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            by_id[rec["Id"]].append(parse_vote(rec["text"], rec["perm"]))
    warn_if_inhomogeneous(by_id, label=os.path.basename(raw_path))
    return by_id


def consensus_of(votes):
    """orderable 투표 최빈 rank의 표수 (orderable 투표가 없으면 0)."""
    counts = Counter(tuple(r) for k, r in votes if k == "orderable")
    return max(counts.values()) if counts else 0


def fuse_all(votes_by_id, ll_by_id, policy, k=3, margin=0.0,
             length_norm=True, ban_identity=False, p_identity_threshold=None,
             disperse_gate=True, disperse_top=1, identity_quota=None):
    """{Id: 최종 rank}. 투표에만 있고 LL이 없는 샘플은 투표 결과로 폴백."""
    assert policy in POLICIES, f"unknown policy: {policy}"
    assert policy != "vgate" or p_identity_threshold is not None, (
        "vgate는 p_identity_threshold 필수 — fuse sweep로 val에서 확정할 것"
    )
    if policy == "ll":
        return {
            sid: decide_one(
                aggregate_ll(views, length_norm), margin, ban_identity,
                p_identity=(identity_posterior(views)
                            if p_identity_threshold is not None else None),
                p_identity_threshold=p_identity_threshold,
            )
            for sid, views in ll_by_id.items()
        }

    preds = {}
    for sid, votes in votes_by_id.items():
        pred_vote, _ = aggregate_votes(
            votes, disperse_gate=disperse_gate, disperse_top=disperse_top,
            identity_quota=identity_quota,
        )
        if policy == "vote" or sid not in ll_by_id:
            preds[sid] = pred_vote
            continue
        if policy == "vgate":
            # 투표가 순서를 정하고 LL의 identity_posterior가 identity 베팅만 결정한다.
            # 합집합(둘 중 하나라도 identity면 identity): 투표의 identity는 P 0.9496로
            # 정밀하니 버리지 않고, 사후확률(AUC 0.8844)이 recall만 보탠다. 단조적이라
            # 투표 기준선의 gate recall을 낮출 수 없고 순서는 건드리지 않는다.
            if (identity_posterior(ll_by_id[sid]) > p_identity_threshold
                    or pred_vote == list(IDENTITY)):
                preds[sid] = list(IDENTITY)
            else:
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
    prior = truth["No_ordering"].astype(bool).mean()
    print(f"sweep 대상: {len(truth)}샘플 (LL 보유 {sum(t in ll_by_id for t in covered)}개), "
          f"no_ordering 기저율 {prior:.4f}")

    margins = [round(float(m), 4)
               for m in np.arange(args.margin_min, args.margin_max + 1e-9, args.margin_step)]

    # 각 config = (라벨용 축 dict, fuse_all 고정 kwargs, 튜닝 축 이름, 튜닝 격자)
    configs = [(dict(policy="vote"), {}, None, [None])]
    # 공짜 레버 기준선: LL 없이 투표만으로 no_ordering recall을 얼마나 사는가.
    # LL은 3090에서 ~1.9h가 드는 반면 이쪽은 무료 — vgate는 plain vote가 아니라
    # 이 기준선을 이겨야 LL 다리를 추가할 명분이 선다.
    for dt in args.disperse_top_grid:
        if dt != 1:
            configs.append((dict(policy="vote", disperse_top=dt),
                            dict(disperse_top=dt), None, [None]))
    for q in args.identity_quota_grid:
        configs.append((dict(policy="vote", identity_quota=q),
                        dict(identity_quota=q), None, [None]))
    for ban in (False, True):
        configs.append((dict(policy="ll", ban_identity=ban),
                        dict(ban_identity=ban), "margin", margins))
        for policy in ("h1", "h2"):
            for k in args.k_grid:
                configs.append((dict(policy=policy, k=k, ban_identity=ban),
                                dict(k=k, ban_identity=ban), "margin", margins))
    # vgate: 투표가 순서, LL 사후확률이 identity 베팅
    for dt in args.disperse_top_grid:
        configs.append((dict(policy="vgate", disperse_top=dt),
                        dict(disperse_top=dt), "pid", list(args.pid_grid)))

    rows = []
    for axis, fixed, tune_name, tune_grid in configs:
        # config별로 격자 전체를 평가한 뒤 **기저율 정합점**을 대표로 뽑는다.
        # EM argmax로 먼저 접으면 안 된다: EM은 평탄면 위 노이즈라 config마다 서로 다른
        # identity율(0.09 ~ 0.43)에서 argmax가 잡히고, 그러면 정책끼리 **다른 작동점**을
        # 비교하게 된다. 정합점이 없으면 가장 가까운 점으로 폴백하고 prior_gap이 드러낸다.
        cands = []
        for val in tune_grid:
            kw = dict(fixed)
            if tune_name == "margin":
                kw["margin"] = val
            elif tune_name == "pid":
                kw["p_identity_threshold"] = val
            preds = fuse_all(votes_by_id, ll_by_id, axis["policy"],
                             length_norm=not args.no_length_norm, **kw)
            m = evaluate(preds, truth)
            cands.append((val, m, abs(m["identity_rate"] - prior)))
        # config별 대표는 EM argmax. (2026-07-16 정정: 이전 판은 "기저율 정합점"을 골랐는데
        # 근거 없는 휴리스틱이었다 — identity 베팅 손익분기는 기저율이 아니라
        # em_orderable/(1+em_orderable) ~ 0.34이고, identity율이 기저율을 넘는 건 정상이다.)
        val, m, gap = max(cands, key=lambda c: c[1]["em"])
        rows.append({**{"policy": None, "k": None, "ban_identity": None,
                        "disperse_top": None, "identity_quota": None}, **axis,
                     "tuned": f"{tune_name}={val}" if tune_name else None,
                     "em": m["em"], "em_orderable": m.get("em_orderable"),
                     "em_no_ordering": m.get("em_no_ordering"),
                     "gate_p": m.get("gate_precision"), "gate_r": m.get("gate_recall"),
                     "identity_rate": m["identity_rate"], "prior_gap": gap,
                     "em_argmax": max(c[1]["em"] for c in cands)})

    table = pd.DataFrame(rows).sort_values("em", ascending=False)
    print("모든 행 = 각 정책이 자기 격자에서 낸 **EM 최대 작동점**. "
          "identity_rate가 기저율(%.3f)을 넘는 것은 정상 — 손익분기가 0.5가 아니라 "
          "em_orderable/(1+em_orderable) ~ 0.34이기 때문." % prior)
    print(table.round(4).to_string(index=False))

    base = table[(table["policy"] == "vote") & table["identity_quota"].isna()
                 & table["disperse_top"].isna()]
    top = table.iloc[0]
    print(f"\n최고: policy={top['policy']} {top['tuned']} em={top['em']:.4f} "
          f"gate_r={top['gate_r']:.4f} identity_rate={top['identity_rate']:.4f}")
    if len(base):
        b = base.iloc[0]
        delta = top["em"] - b["em"]
        n = len(truth)
        # 유의성 눈금: EM 1pp = n/100 샘플. 페어드 비교라도 한 자릿수 샘플 차이는 노이즈다.
        print(f"기준선 vote: em={b['em']:.4f} gate_r={b['gate_r']:.4f} -> "
              f"차이 {delta:+.4f} ({delta * n:+.0f}샘플 / n={n}, 1pp={n / 100:.0f}샘플)")
        if abs(delta) * n < 10:
            print("[판정] 기준선과의 차이가 10샘플 미만 — **노이즈. 채택 근거 없음.**")
    free = table[table["identity_quota"].notna() | table["disperse_top"].notna()]
    free = free[free["policy"] == "vote"]
    vg = table[table["policy"] == "vgate"]
    if len(free) and len(vg):
        fb, vb = free.iloc[0], vg.iloc[0]
        print(f"공짜 레버 최고(vote+quota/disperse): em={fb['em']:.4f} gate_r={fb['gate_r']:.4f} | "
              f"vgate 최고: em={vb['em']:.4f} gate_r={vb['gate_r']:.4f} -> "
              f"vgate 이득 {vb['em'] - fb['em']:+.4f} ({(vb['em'] - fb['em']) * len(truth):+.0f}샘플)")
        print("  LL은 3090에서 ~1.9h. 이득이 노이즈면 공짜 레버를 쓰고 LL 다리는 버린다.")


def cmd_decide(args):
    import pandas as pd

    from src.train.targets import format_answer

    preds = fuse_all(
        load_votes(args.raw), load_ll(args.ll), args.policy, k=args.k,
        margin=args.margin, length_norm=not args.no_length_norm,
        ban_identity=args.ban_identity, p_identity_threshold=args.pid_threshold,
        disperse_top=args.disperse_top, identity_quota=args.identity_quota,
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

    sw = sub.add_parser("sweep", help="val 정답으로 정책×k×margin×ban×pid 그리드 EM")
    sw.add_argument("--raw", required=True, help="생성-투표 raw jsonl")
    sw.add_argument("--ll", required=True, help="LL 스코어 raw jsonl")
    sw.add_argument("--fold", default="val")
    sw.add_argument("--k-grid", type=int, nargs="+", default=[2, 3, 4])
    sw.add_argument("--pid-grid", type=float, nargs="+",
                    default=[0.003, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.02, 0.03, 0.05],
                    help="vgate의 identity_posterior 문턱 격자")
    sw.add_argument("--disperse-top-grid", type=int, nargs="+", default=[1, 2],
                    help="투표 분산 게이트 문턱 (tta8은 최빈 표수 최솟값이 2라 1이면 미발동)")
    sw.add_argument("--identity-quota-grid", type=int, nargs="+", default=[1, 2],
                    help="identity 표가 이 수 이상이면 identity (공짜 레버 기준선)")
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
    de.add_argument("--margin", type=float, default=None,
                    help="ll/h1/h2 필수 (sweep 확정값). vgate/vote는 미사용")
    de.add_argument("--pid-threshold", type=float, default=None,
                    help="vgate 필수: identity_posterior 문턱 (sweep 확정값)")
    de.add_argument("--disperse-top", type=int, default=1)
    de.add_argument("--identity-quota", type=int, default=None)
    de.add_argument("--no-length-norm", action="store_true")
    de.add_argument("--ban-identity", action="store_true")
    de.add_argument("--out", default=None)
    de.add_argument("--submission", default=None)
    de.set_defaults(fn=cmd_decide)

    args = ap.parse_args()
    # 튜닝 필수값 누락 방어: 스윕 없이 제출본을 만드는 사고를 막는다
    if args.cmd == "decide":
        if args.policy == "vgate" and args.pid_threshold is None:
            ap.error("--policy vgate 는 --pid-threshold 필수 (fuse sweep로 val에서 확정)")
        if args.policy in ("ll", "h1", "h2") and args.margin is None:
            ap.error(f"--policy {args.policy} 는 --margin 필수 (fuse sweep로 val에서 확정)")
    args.fn(args)


if __name__ == "__main__":
    main()
