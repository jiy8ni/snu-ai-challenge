"""24-순열 전수 로그우도 스코어링 + UNORDERABLE LL 판별 (생성-파싱 대안 경로).

생성 방식(predict.py -> aggregate.py)의 두 약점을 동시에 제거한다:
  1. parse fail 원천 차단 — 생성 없이 25개 후보(24 orderable + UNORDERABLE)의
     teacher-forcing 로그우도를 직접 비교해 답을 고른다.
  2. UNORDERABLE 미출력 — greedy 생성에서 0회였던 UNORDERABLE을 채점으로 되찾는다.

  ⚠️ 2번의 원래 수단이던 **unord 마진(score(unorderable) - max score(orderable) > margin)은
  무용함이 실측됐다: no_ordering 판별 AUC 0.5555** (val 953, 7B_0706). 모델은 no_ordering을
  UNORDERABLE 후보의 우도가 아니라 **Answer 슬롯의 identity**로 표현하기 때문이다.
  실제로 쓸 신호는 `identity_posterior` — 24-way softmax에서 identity 후보의 사후확률로
  **AUC 0.8844**. (엔트로피 0.5532 / top1-top2 margin 0.5553도 무용 — "불확실함" != "정렬 불가".)
  margin 경로는 하위 호환으로 남겨두되 신규 튜닝은 --pid-grid를 쓴다.

2단계 구조 (predict/aggregate와 동일한 분리):
  score : 샘플×TTA뷰마다 25후보의 (logprob 합, 토큰 수) -> raw jsonl. GPU 필요.
  decide: LL 합산 -> 최종 rank CSV/submission. 문턱 재튜닝 시 이 단계만 반복.
  sweep : val 정답으로 margin 또는 p_identity 그리드 EM 평가 (--pid-grid 권장).
          문턱 선택은 EM argmax가 아니라 **기저율 정합**으로 한다 — EM은 평탄면이고
          val 953에서 1pp = 10샘플이라 argmax는 노이즈를 고른다.

후보 rank 키는 항상 **원 좌표** 기준으로 저장하므로 TTA 뷰 간 합산이 바로 된다.
orderable 24후보는 mid 타깃 텍스트 길이가 전부 같아 length-norm이 argmax를 바꾸지
않지만, UNORDERABLE 후보(Order 줄 없음)와의 비교에는 length-norm(평균 logprob)이
기본이다 (--no-length-norm으로 합 비교).

메모리: --chunk가 한 forward의 후보 수. A100 40GB는 25(한 방), 3090 24GB는 4 권장
(로짓 텐서가 chunk x seq x vocab로 커진다).

사용:
  python -m src.infer.ll_score score  --model M --split train --fold val --tta 1 \
      --out outputs/ll_val.jsonl [--chunk 25] [--limit 5]
  python -m src.infer.ll_score sweep  --raw outputs/ll_val.jsonl --fold val
  python -m src.infer.ll_score decide --raw outputs/ll_test.jsonl --margin M \
      [--out pred.csv | --submission submission.csv]
"""

import argparse
import json
import math
import os
from collections import defaultdict

import torch
from tqdm import tqdm

from src.data.loader import load_paths, load_split
from src.infer.predict import load_frames, load_model, tta_perms
from src.train.targets import build_target, format_answer
from src.train.vl_dataset import build_messages
from src.utils.permutation import (
    ALL_PERMUTATIONS,
    IDENTITY,
    N_FRAMES,
    parse_answer_column,
    shuffle_rank_label,
)

UNORD_KEY = "unorderable"

# 픽셀 캡 정상 적용 시 4프레임 mid 프롬프트는 ~700토큰. 이를 크게 넘으면
# 캡이 무시된 것(2026-07-06 OOM 원인)이므로 OOM 전에 명시적으로 중단한다.
MAX_PROMPT_TOKENS = 3000


def candidate_texts(perm, style="mid"):
    """(키, 셔플 좌표 타깃 텍스트) 목록. 키는 원 좌표 rank 문자열 (+ mid는 UNORD_KEY).

    프레임이 perm으로 재배치된 뷰에서 원 좌표 rank r의 정답 텍스트는
    shuffle_rank_label(r, perm)을 담아야 한다 (vl_dataset 증강과 동일 규칙).

    style="plain"이면 **24개** (UNORDERABLE 후보 없음 — plain에는 그 개념이 자체가
    없다, src/train/targets.py docstring 참조). mid이면 25개.
    """
    cands = []
    for rank in ALL_PERMUTATIONS:
        shuffled = shuffle_rank_label(rank, perm)
        cands.append((format_answer(rank), build_target(style, None, shuffled, False)))
    if style != "plain":
        cands.append((UNORD_KEY, build_target(style, None, None, True)))
    return cands


@torch.inference_mode()
def score_view(model, processor, frames, caption, texts, device, chunk, style="mid"):
    """한 (샘플, TTA뷰)의 후보 텍스트들 -> [(logprob 합, 응답 토큰 수)] 목록."""
    user_msgs = build_messages(frames, caption, style)
    prompt = processor.apply_chat_template(user_msgs, tokenize=False, add_generation_prompt=True)
    fulls = []
    for t in texts:
        conv = user_msgs + [{"role": "assistant", "content": [{"type": "text", "text": t}]}]
        full = processor.apply_chat_template(conv, tokenize=False)
        assert full.startswith(prompt), "chat template: 프롬프트가 전체 대화의 접두어가 아님"
        fulls.append(full)

    prompt_ids = processor(text=[prompt], images=[frames], return_tensors="pt")["input_ids"][0]
    L_p = prompt_ids.shape[0]
    if L_p > MAX_PROMPT_TOKENS:
        raise RuntimeError(
            f"프롬프트가 {L_p}토큰 (> {MAX_PROMPT_TOKENS}) — 픽셀 캡 미적용 신호. "
            "cap_pixels/apply_pixel_caps가 동작하는 코드 버전인지 확인할 것."
        )

    results = []
    for s in range(0, len(fulls), chunk):
        batch = fulls[s : s + chunk]
        inputs = processor(
            text=batch, images=[frames] * len(batch), return_tensors="pt", padding=True
        )
        # 공유 프롬프트 접두어 정합 검증 (BPE 경계가 어긋나면 마스크가 틀어진다)
        assert torch.equal(inputs["input_ids"][0, :L_p], prompt_ids), "prompt prefix mismatch"
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # 응답 구간 로짓만 계산 — 전체 시퀀스 로짓(seq x 152k vocab)은 대형 이미지
        # 샘플에서 OOM (2026-07-06 test 실측: seq ~8400 -> 60GB 할당 시도)
        n_keep = inputs["input_ids"].shape[1] - L_p + 1
        try:
            logits = model(**inputs, logits_to_keep=n_keep).logits
        except TypeError:
            try:  # 구명명 (transformers 구버전)
                logits = model(**inputs, num_logits_to_keep=n_keep).logits
            except TypeError:  # 미지원: 전체 로짓 후 슬라이스
                logits = model(**inputs).logits
        if logits.shape[1] > n_keep:  # kwarg가 조용히 무시된 경우도 여기서 정합
            logits = logits[:, L_p - 1 :]
        resp_logits = logits[:, :-1].float()
        resp_labels = inputs["input_ids"][:, L_p:]
        resp_mask = inputs["attention_mask"][:, L_p:].bool()
        token_lp = torch.log_softmax(resp_logits, dim=-1).gather(
            -1, resp_labels.unsqueeze(-1)
        ).squeeze(-1)
        token_lp = token_lp * resp_mask
        for i in range(len(batch)):
            results.append((float(token_lp[i].sum()), int(resp_mask[i].sum())))
        del logits, resp_logits, token_lp
    return results


def cmd_score(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model, processor = load_model(args.model, device)
    processor.tokenizer.padding_side = "right"  # 스코어링은 right-pad (접두어 정렬)

    df = load_split(args.split)
    if args.fold:
        import pandas as pd

        split_df = pd.read_csv(os.path.join(load_paths()["outputs_dir"], "split.csv"))
        df = df.merge(split_df[["Id", "fold"]], on="Id")
        df = df[df["fold"] == args.fold]
    if args.limit:
        df = df.head(args.limit)

    perms = tta_perms(args.tta, args.seed)

    done = set()
    if os.path.exists(args.out):  # 재개: 이미 스코어된 (Id, perm) 건너뜀
        with open(args.out, encoding="utf-8") as f:
            done = {(r["Id"], tuple(r["perm"])) for r in map(json.loads, f)}

    jobs = [
        (row, perm)
        for _, row in df.iterrows()
        for perm in perms
        if (row["Id"], tuple(perm)) not in done
    ]

    chunk = args.chunk
    with open(args.out, "a", encoding="utf-8") as fout:
        for row, perm in tqdm(jobs, desc=f"ll_score:{args.split}"):
            try:
                frames = load_frames(row, crop=not args.no_crop)
            except OSError as e:
                # predict.py와 동일: 손상 이미지는 스킵, 데이터 복구 후 재개 시 재시도
                print(f"[손상 이미지 스킵] Id={row['Id']} perm={perm}: {e}")
                continue
            frames = [frames[perm[j]] for j in range(N_FRAMES)]
            cands = candidate_texts(perm, args.style)
            while True:
                try:
                    scores = score_view(
                        model, processor, frames, row["Sentence"],
                        [t for _, t in cands], device, chunk, args.style,
                    )
                    break
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    if chunk <= 1:
                        raise
                    chunk //= 2  # 이후 샘플에도 유지 (반복 OOM 방지)
                    print(f"[OOM] Id={row['Id']}: chunk을 {chunk}로 줄여 재시도")
            ll = {key: [lp, n] for (key, _), (lp, n) in zip(cands, scores)}
            fout.write(json.dumps(
                {"Id": row["Id"], "perm": perm, "ll": ll}, ensure_ascii=False
            ) + "\n")
            fout.flush()
    print(f"saved: {args.out} (+{len(jobs)} views)")


def load_ll(raw_path):
    """raw jsonl -> {Id: 뷰별 ll dict 리스트}."""
    by_id = defaultdict(list)
    with open(raw_path, encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            by_id[rec["Id"]].append(rec["ll"])
    return by_id


def aggregate_ll(views, length_norm=True):
    """뷰별 {키: [logprob 합, 토큰 수]} 목록 -> 합산 후 {키: 스코어}."""
    agg = defaultdict(lambda: [0.0, 0])
    for ll in views:
        for key, (lp, n) in ll.items():
            agg[key][0] += lp
            agg[key][1] += n
    return {k: (lp / max(n, 1) if length_norm else lp) for k, (lp, n) in agg.items()}


def identity_posterior(views):
    """뷰별 24-orderable softmax에서 identity 후보의 사후확률 -> 뷰 평균.

    orderable 정답에는 identity가 0건(train 7,255건 전수)이므로 이 값은 곧 모델이
    말하는 P(no_ordering)이다. 생성이 아니라 채점이라 greedy 디코드의 포맷 붕괴
    (생성 6,552건 전부 "Order:"로 시작, UNORDERABLE 출력 0건)를 우회한다.
    val 953 실측 AUC 0.8844 — unord_gap(0.5555)·엔트로피(0.5532)는 무용.

    설계상 주의 두 가지:
      * 반드시 **원 logprob 합**에 softmax를 건다. 24후보는 토큰 수가 39로 모두
        같아 length-norm은 온도 1/39와 동치이고, 분포를 1/24 근처로 뭉갠다
        (AUC 0.8844 -> 0.8028, p 범위 (0.034, 0.056)로 붕괴). 그래서 이 함수는
        aggregate_ll을 거치지 않고 views를 직접 받으며 length_norm 인자가 없다.
      * 뷰 간에는 logprob 합산(온도가 뷰 수만큼 예리해짐) 대신 **사후확률 평균**을
        쓴다. 문턱이 --tta 수에 불변이 되어 val에서 튜닝한 값이 그대로 전이된다.

    UNORD_KEY는 길이(22 vs 39)도 사건도 달라 후보 집합에서 제외한다.
    """
    id_key = format_answer(IDENTITY)
    total = 0.0
    for ll in views:
        lps = {k: lp for k, (lp, _) in ll.items() if k != UNORD_KEY}
        assert len(lps) == len(ALL_PERMUTATIONS), (
            f"orderable 후보가 {len(lps)}개 (24 기대) — score 중단본이면 재개 후 사용할 것"
        )
        top = max(lps.values())
        exp = {k: math.exp(lp - top) for k, lp in lps.items()}
        total += exp[id_key] / sum(exp.values())
    return total / max(len(views), 1)


def decide_one(scores, margin=0.0, ban_identity=False, allowed=None,
               p_identity=None, p_identity_threshold=None):
    """{키: 스코어} -> 최종 rank 1건.

    allowed: orderable 후보를 이 rank(원 좌표 리스트) 집합으로 제한 (fuse의 투표
    후보 재순위용). 제한 결과가 비면 전수로 폴백한다. ban_identity도 동일하게
    비면 폴백 — 24후보 전수에서는 발생하지 않지만 allowed와 조합 시 가능.

    p_identity/p_identity_threshold: identity_posterior 게이트 (둘 다 주어질 때만
    작동). unord_gap margin(AUC 0.5555)보다 강한 신호(AUC 0.8844)이므로 먼저 본다.
    게이트가 "orderable"로 판정하면 identity는 정의상 오답이므로 후보에서 뺀다
    (튜닝 손잡이가 아니라 규칙의 정합성 속성).
    margin=None이면 unord_gap 게이트를 끈다 (스윕에서 두 게이트를 분리 평가).
    """
    armed = p_identity is not None and p_identity_threshold is not None
    if armed and p_identity > p_identity_threshold:
        return list(IDENTITY)

    orderable = sorted(k for k in scores if k != UNORD_KEY)
    if allowed is not None:
        allow = {tuple(r) for r in allowed}
        restricted = [k for k in orderable if tuple(parse_answer_column(k)) in allow]
        if restricted:
            orderable = restricted
    if ban_identity or armed:
        non_id = [k for k in orderable if parse_answer_column(k) != IDENTITY]
        if non_id:
            orderable = non_id
    best = max(orderable, key=scores.__getitem__)  # sorted 선행으로 동률 시 사전순 결정적

    if UNORD_KEY in scores and margin is not None and scores[UNORD_KEY] - scores[best] > margin:
        return list(IDENTITY)
    return parse_answer_column(best)


def decide_all(by_id, margin=0.0, length_norm=True, ban_identity=False,
               p_identity_threshold=None):
    """뷰별 LL 합산 -> {Id: 최종 rank}. margin: UNORDERABLE 판정 문턱.

    p_identity_threshold: identity_posterior 게이트 문턱 (None=off). 사후확률은
    length_norm과 무관하게 views 원본에서 계산된다.
    """
    return {
        sid: decide_one(
            aggregate_ll(views, length_norm), margin, ban_identity,
            p_identity=(identity_posterior(views)
                        if p_identity_threshold is not None else None),
            p_identity_threshold=p_identity_threshold,
        )
        for sid, views in by_id.items()
    }


def cmd_decide(args):
    import pandas as pd

    preds = decide_all(
        load_ll(args.raw),
        margin=None if args.pid_threshold is not None else args.margin,
        length_norm=not args.no_length_norm, ban_identity=args.ban_identity,
        p_identity_threshold=args.pid_threshold,
    )
    print(f"n={len(preds)}, identity_rate={sum(p == list(IDENTITY) for p in preds.values()) / max(len(preds), 1):.4f}")
    if args.out:
        pd.DataFrame(
            {"Id": list(preds), "Answer": [format_answer(r) for r in preds.values()]}
        ).to_csv(args.out, index=False)
        print(f"saved: {args.out}")
    if args.submission:
        from src.infer.submission import build_submission

        build_submission(preds, args.submission)


def cmd_sweep(args):
    import numpy as np
    import pandas as pd

    from src.eval.em import evaluate

    by_id = load_ll(args.raw)

    truth = load_split("train")
    split_df = pd.read_csv(os.path.join(load_paths()["outputs_dir"], "split.csv"))
    truth = truth.merge(split_df[["Id", "fold"]], on="Id")
    truth = truth[truth["fold"] == args.fold]
    n_fold = len(truth)
    truth = truth[truth["Id"].isin(by_id)]  # --limit 스코어링 대응
    if len(truth) < n_fold:
        print(f"[경고] fold {args.fold} {n_fold}개 중 {len(truth)}개만 스코어됨 — "
              "score가 중단된 상태라면 재개 완료 후 margin을 확정할 것")
    prior = truth["No_ordering"].astype(bool).mean()

    def row_for(preds, **axis):
        m = evaluate(preds, truth)
        return {**axis, "em": m["em"], "em_orderable": m.get("em_orderable"),
                "em_no_ordering": m.get("em_no_ordering"),
                "gate_p": m.get("gate_precision"), "gate_r": m.get("gate_recall"),
                "identity_rate": m["identity_rate"],
                "prior_gap": abs(m["identity_rate"] - prior)}

    if args.pid_grid:
        # identity_posterior 게이트 (AUC 0.8844). unord margin(AUC 0.5555)은 꺼서
        # 두 게이트를 분리 평가한다 — 격자를 교차시켜봐야 무신호 축만 늘어난다.
        print(f"sweep 대상: {len(truth)}샘플, p_identity 문턱 {args.pid_grid} (unord margin off)")
        rows = [row_for(decide_all(by_id, margin=None,
                                   length_norm=not args.no_length_norm,
                                   ban_identity=args.ban_identity,
                                   p_identity_threshold=float(t)), pid=float(t))
                for t in args.pid_grid]
        axis_name = "pid"
    else:
        print(f"sweep 대상: {len(truth)}샘플, margin [{args.margin_min}, {args.margin_max}] step {args.margin_step}")
        rows = [row_for(decide_all(by_id, margin=float(m0),
                                   length_norm=not args.no_length_norm,
                                   ban_identity=args.ban_identity),
                        margin=round(float(m0), 4))
                for m0 in np.arange(args.margin_min, args.margin_max + 1e-9, args.margin_step)]
        axis_name = "margin"

    table = pd.DataFrame(rows)
    print(table.round(4).to_string(index=False))
    best = table.loc[table["em"].idxmax()]
    n = len(truth)
    print(f"\nEM argmax: {axis_name}={best[axis_name]} em={best['em']:.4f} "
          f"gate_p={best['gate_p']:.4f} identity_rate={best['identity_rate']:.4f}")
    # identity 베팅의 손익분기: 이득 q*1 vs 손실 (1-q)*em_orderable
    #   -> q* = em_orderable / (1 + em_orderable)
    # gate_p가 q*보다 높으면 과소베팅(더 걸어야 이득), 낮으면 과다베팅.
    # 주의: identity율이 기저율을 넘는 것은 **정상**이다 (q* < 0.5이므로).
    #   2026-07-16 정정 — 이전 판의 "기저율 정합으로 문턱을 고른다"는 근거 없는 휴리스틱이었다.
    eo = best.get("em_orderable")
    if eo:
        q_star = eo / (1 + eo)
        print(f"손익분기 gate_p = em_orderable/(1+em_orderable) = {eo:.4f}/{1 + eo:.4f} = {q_star:.4f}"
              f"  (기저율 {prior:.4f}이 아님)")
        print(f"  -> EM argmax의 gate_p {best['gate_p']:.4f}는 손익분기 "
              f"{'위 (이론 정합)' if best['gate_p'] >= q_star else '아래 (과다베팅)'}")
    # 실측(val 953, 7B_0706): recall 0.497->0.821로 65% 올려도 EM은 +6샘플뿐이다.
    # no_ordering(15.8%)에서 +5.1pp 벌고 orderable(84.2%)에서 -4.5pp 잃는 1:1 트레이드.
    print(f"[유의성] n={n}, EM 1pp = {n / 100:.0f}샘플. 표 안의 EM 폭 "
          f"{table['em'].max() - table['em'].min():.4f} "
          f"({(table['em'].max() - table['em'].min()) * n:.0f}샘플) — "
          "이 폭이 10샘플 수준이면 문턱 선택은 무의미하다 (EM 표면이 평평)")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    sc = sub.add_parser("score", help="후보 LL 스코어링 -> raw jsonl (GPU)")
    sc.add_argument("--model", required=True)
    sc.add_argument("--split", default="test", choices=["train", "test"])
    sc.add_argument("--fold", default=None, help="train일 때 split.csv fold 필터 (예: val)")
    sc.add_argument("--tta", type=int, default=1, help="1=identity 뷰만 (기본). 확장은 val 검증 후")
    sc.add_argument("--style", default="mid", choices=["mid", "plain"],
                    help="학습 스타일과 일치시킬 것. plain이면 UNORDERABLE 후보 없이 24후보만")
    sc.add_argument("--chunk", type=int, default=25, help="forward당 후보 수 (3090은 4 권장)")
    sc.add_argument("--out", required=True)
    sc.add_argument("--limit", type=int, default=None)
    sc.add_argument("--no-crop", action="store_true")
    sc.add_argument("--seed", type=int, default=42)
    sc.set_defaults(fn=cmd_score)

    de = sub.add_parser("decide", help="LL raw -> 최종 rank CSV/submission")
    de.add_argument("--raw", required=True)
    de.add_argument("--margin", type=float, required=True, help="sweep로 val에서 튜닝한 값")
    de.add_argument("--pid-threshold", type=float, default=None,
                    help="identity_posterior 게이트 문턱 (sweep --pid-grid로 튜닝). 주면 margin 무시")
    de.add_argument("--no-length-norm", action="store_true")
    de.add_argument("--ban-identity", action="store_true")
    de.add_argument("--out", default=None)
    de.add_argument("--submission", default=None)
    de.set_defaults(fn=cmd_decide)

    sw = sub.add_parser("sweep", help="val 정답으로 margin 또는 p_identity 그리드 EM")
    sw.add_argument("--raw", required=True)
    sw.add_argument("--fold", default="val")
    sw.add_argument("--pid-grid", type=float, nargs="+", default=None,
                    help="identity_posterior 문턱 격자 (주면 margin 축 대신 이걸 스윕). "
                         "권장: 0.002 0.003 0.005 0.0075 0.01 0.0125 0.015 0.02 0.03 0.05 0.1")
    sw.add_argument("--margin-min", type=float, default=-0.5)
    sw.add_argument("--margin-max", type=float, default=0.5)
    sw.add_argument("--margin-step", type=float, default=0.025)
    sw.add_argument("--no-length-norm", action="store_true")
    sw.add_argument("--ban-identity", action="store_true")
    sw.set_defaults(fn=cmd_sweep)

    args = ap.parse_args()
    args.fn(args)


if __name__ == "__main__":
    main()
