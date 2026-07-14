"""Track A 학습·평가·추론 CLI (CPU, 분 단위 재학습 — 전처리 ablation 엔진).

  python -m src.track_a.train --mode holdout                       # 의사결정 기본 (train->val)
  python -m src.track_a.train --mode holdout --variant nocrop      # ablation: 크롭 효과
  python -m src.track_a.train --mode holdout --no-events           # ablation: 이벤트 토큰
  python -m src.track_a.train --mode holdout --noise-policy a      # ablation: 노이즈 정책
  python -m src.track_a.train --mode cv                            # 5-fold (caption_group)
  python -m src.track_a.train --mode full --epochs 30 --tau 0.55   # 전체 재학습(단일 최종)
  python -m src.track_a.train --mode predict --ckpt outputs/track_a/full_crop/ckpt.pt

모든 모드의 지표는 순열 유효성이 보장된 디코딩(decode.py) 기준 EM이다.
"""

import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import yaml

from src.data.loader import load_paths
from src.eval.em import evaluate
from src.track_a.dataset import N_SCALARS, load_track_a
from src.track_a.decode import calibrate_tau, decode, perm_nll, perm_scores
from src.track_a.model import OrderHead


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path=None, **overrides):
    cfg_path = path or os.path.join(load_paths()["outputs_dir"], "..", "configs", "track_a.yaml")
    with open(cfg_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    return cfg


def make_model(cfg):
    return OrderHead(
        d_model=cfg["d_model"], n_scalars=N_SCALARS, n_layers=cfg["n_layers"],
        n_heads=cfg["n_heads"], d_ffn=cfg["d_ffn"], dropout=cfg["dropout"],
        use_events=cfg["use_events"], use_scalars=cfg["use_scalars"],
    )


def _losses(model, data, idx, cfg, pos_weight):
    img, scalars, events, mask, caption = data.tensors(idx)
    rank_logits, gate_logit = model(img, scalars, events, mask, caption)

    meta = data.meta.iloc[idx.numpy()]
    orderable = torch.tensor((~meta["no_ordering"]).values)
    weights = torch.ones(len(idx))
    if cfg["noise_policy"] == "b":
        weights[torch.tensor(meta["noise_flag"].values)] = cfg["noise_weight"]

    gate_target = torch.tensor(meta["no_ordering"].values, dtype=torch.float32)
    gate_loss_all = F.binary_cross_entropy_with_logits(
        gate_logit, gate_target, reduction="none", pos_weight=pos_weight
    )
    loss = (gate_loss_all * weights).mean() * cfg["gate_w"]

    if orderable.any():
        o = orderable.nonzero(as_tuple=True)[0]
        ranks = torch.tensor(np.stack(meta["rank"].values)[o.numpy()])
        w_o = weights[o]
        scores = perm_scores(rank_logits[o])
        target_idx = torch.tensor([_rank_index(r) for r in ranks.tolist()])
        perm_ce = F.cross_entropy(
            scores, target_idx, label_smoothing=cfg["label_smoothing"], reduction="none"
        )
        frame_ce = F.cross_entropy(
            rank_logits[o].reshape(-1, 4), (ranks - 1).reshape(-1), reduction="none"
        ).view(-1, 4).mean(dim=1)
        loss = loss + (perm_ce * w_o).mean() + cfg["aux_w"] * (frame_ce * w_o).mean()
    return loss


def _rank_index(rank):
    from src.track_a.decode import perm_index

    return perm_index([int(v) for v in rank])


@torch.inference_mode()
def _forward_all(model, data, idx, batch=1024):
    outs_r, outs_g = [], []
    for s in range(0, len(idx), batch):
        chunk = idx[s : s + batch]
        r, g = model(*data.tensors(chunk))
        outs_r.append(r)
        outs_g.append(torch.sigmoid(g))
    return torch.cat(outs_r), torch.cat(outs_g)


def eval_indices(model, data, idx, ban_identity=False):
    """디코딩 EM + τ 캘리브 + 상세 지표 + 순수 rank 정확도(perm_acc)."""
    model.eval()
    rank_logits, gate_probs = _forward_all(model, data, idx)
    meta = data.meta.iloc[idx.numpy()]
    truth_ranks = list(meta["rank"])
    tau, em, curve = calibrate_tau(
        rank_logits, gate_probs, truth_ranks, ban_identity=ban_identity
    )

    preds = decode(rank_logits, gate_probs, tau, ban_identity=ban_identity)
    pred_by_id = dict(zip(meta["Id"], preds))
    truth_df = pd.DataFrame(
        {"Id": meta["Id"], "Answer": truth_ranks, "No_ordering": meta["no_ordering"]}
    )
    detail = evaluate(pred_by_id, truth_df)

    # 게이트를 배제한 순수 순열 정확도 (orderable만) — EM이 identity 하한에 눌려 있을 때
    # 학습 진행을 감지하는 유일한 신호
    pure = decode(rank_logits, gate_probs, tau=2.0, ban_identity=ban_identity)
    orderable = (~meta["no_ordering"]).values
    perm_acc = float(
        np.mean([p == t for p, t, o in zip(pure, truth_ranks, orderable) if o])
    ) if orderable.any() else float("nan")

    detail.update({"tau": tau, "em_at_tau": em, "perm_acc": perm_acc})
    return detail, curve


def train_one(data, train_idx, val_idx, cfg, log_prefix=""):
    """단일 학습 런. val_idx가 있으면 val EM 기준 early stop + best state 반환."""
    set_seed(cfg["seed"])
    model = make_model(cfg)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    steps_per_epoch = max(1, len(train_idx) // cfg["batch_size"])
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=cfg["epochs"] * steps_per_epoch)

    meta = data.meta.iloc[train_idx.numpy()]
    n_pos = int(meta["no_ordering"].sum())
    pos_weight = torch.tensor((len(meta) - n_pos) / max(n_pos, 1))

    if cfg["noise_policy"] == "a":
        keep = ~meta["noise_flag"].values
        train_idx = train_idx[torch.tensor(keep)]

    best = {"em": -1.0, "epoch": -1, "state": None, "detail": None}
    best_progress = -1.0  # patience용 진행 신호: EM이 하한에 눌려 있어도 perm_acc 개선을 인정
    patience_left = cfg["patience"]
    min_epochs = cfg.get("min_epochs", 20)
    g = torch.Generator().manual_seed(cfg["seed"])

    for epoch in range(cfg["epochs"]):
        model.train()
        order = train_idx[torch.randperm(len(train_idx), generator=g)]
        total = 0.0
        for s in range(0, len(order), cfg["batch_size"]):
            batch_idx = order[s : s + cfg["batch_size"]]
            loss = _losses(model, data, batch_idx, cfg, pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            total += float(loss.detach())

        if val_idx is None:
            continue
        detail, _ = eval_indices(model, data, val_idx, ban_identity=cfg.get("ban_identity", False))
        em, perm_acc = detail["em_at_tau"], detail["perm_acc"]
        marker = ""
        if em > best["em"]:
            best.update(
                {"em": em, "epoch": epoch, "detail": detail,
                 "state": {k: v.clone() for k, v in model.state_dict().items()}}
            )
            marker = " *"
        progress = em + 0.1 * perm_acc  # EM 우선, 동률이면 perm_acc 개선도 진행으로 간주
        if progress > best_progress + 1e-6:
            best_progress = progress
            patience_left = cfg["patience"]
        elif epoch >= min_epochs:
            patience_left -= 1
        print(f"{log_prefix}ep{epoch:02d} loss {total / steps_per_epoch:.4f} "
              f"val_em {em:.4f} perm_acc {perm_acc:.4f} (tau {detail['tau']:.2f}){marker}")
        if patience_left <= 0:
            break

    if val_idx is not None and best["state"] is not None:
        model.load_state_dict(best["state"])
    return model, best


def run_holdout(cfg, run_dir):
    data = load_track_a("train", cfg["variant"])
    fold = data.meta["fold"]
    train_idx = torch.tensor(np.flatnonzero(fold == "train"))
    val_idx = torch.tensor(np.flatnonzero(fold == "val"))
    model, best = train_one(data, train_idx, val_idx, cfg)
    detail, curve = eval_indices(
        model, data, val_idx, ban_identity=cfg.get("ban_identity", False)
    )
    save_run(run_dir, cfg, {"mode": "holdout", "best_epoch": best["epoch"], **detail},
             model=model, curve=curve)
    return detail


def run_cv(cfg, run_dir):
    from sklearn.model_selection import GroupKFold

    data = load_track_a("train", cfg["variant"])
    groups = data.meta["caption_group"].values
    gkf = GroupKFold(n_splits=cfg["cv_folds"])
    results = []
    for k, (tr, va) in enumerate(gkf.split(np.arange(len(data)), groups=groups)):
        _, best = train_one(
            data, torch.tensor(tr), torch.tensor(va), cfg, log_prefix=f"[fold{k}] "
        )
        results.append({"fold": k, "em": best["em"], "tau": best["detail"]["tau"],
                        "epoch": best["epoch"]})
        print(f"[fold{k}] best em {best['em']:.4f}")
    ems = [r["em"] for r in results]
    summary = {"mode": "cv", "em_mean": float(np.mean(ems)), "em_std": float(np.std(ems)),
               "folds": results}
    save_run(run_dir, cfg, summary)
    return summary


def run_full(cfg, run_dir):
    assert cfg.get("tau") is not None, "--tau 필요 (holdout 캘리브 값 사용)"
    data = load_track_a("train", cfg["variant"])
    model, _ = train_one(data, torch.arange(len(data)), None, cfg)
    ckpt = {"state_dict": model.state_dict(), "cfg": cfg, "tau": cfg["tau"]}
    os.makedirs(run_dir, exist_ok=True)
    torch.save(ckpt, os.path.join(run_dir, "ckpt.pt"))
    save_run(run_dir, cfg, {"mode": "full", "n_train": len(data), "tau": cfg["tau"]})
    print(f"saved: {run_dir}/ckpt.pt")


def run_decode_ablation(args):
    """저장된 체크포인트로 디코딩 규칙 ablation (재학습 불필요).

    게이트 사용(τ 캘리브) vs 미사용(τ>1) × identity 금지 on/off 4조합의 val EM.
    reports/preprocessing.md 행 2·3의 근거.
    """
    from src.track_a.decode import IDENTITY_IDX, perm_scores
    from src.utils.permutation import ALL_PERMUTATIONS, IDENTITY

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg = ckpt["cfg"]
    model = make_model(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    data = load_track_a("train", cfg["variant"])
    fold = data.meta["fold"]
    val_idx = torch.tensor(np.flatnonzero(fold == "val"))
    rank_logits, gate_probs = _forward_all(model, data, val_idx)
    meta = data.meta.iloc[val_idx.numpy()]
    truth = [list(r) for r in meta["rank"]]

    scores = perm_scores(rank_logits)
    results = {}
    for ban in (True, False):
        s = scores.clone()
        if ban:
            s[:, IDENTITY_IDX] = float("-inf")
        best_perm = [list(ALL_PERMUTATIONS[int(b)]) for b in s.argmax(dim=1)]
        # 게이트 미사용: 순수 rank 경로
        em_nogate = float(np.mean([p == t for p, t in zip(best_perm, truth)]))
        # 게이트 사용: τ 스윕 최적
        taus = [round(t, 2) for t in np.arange(0.0, 1.02, 0.02)]
        gate = gate_probs.numpy()
        is_id_truth = np.array([t == IDENTITY for t in truth])
        pc = np.array([p == t for p, t in zip(best_perm, truth)])
        em_gate = max(
            float(np.where(gate > tau, is_id_truth, pc).mean()) for tau in taus
        )
        results[f"ban={ban}"] = {"em_gate_off": em_nogate, "em_gate_on": em_gate}
    print(json.dumps(results, indent=2))
    return results


def run_predict(args):
    from src.infer.submission import build_submission

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    cfg, tau = ckpt["cfg"], ckpt["tau"]
    model = make_model(cfg)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    data = load_track_a(args.split, cfg["variant"])
    idx = torch.arange(len(data))
    rank_logits, gate_probs = _forward_all(model, data, idx)
    preds = decode(
        rank_logits, gate_probs, tau, ban_identity=cfg.get("ban_identity", False)
    )
    pred_by_id = dict(zip(data.ids, preds))

    out_csv = args.out or os.path.join(os.path.dirname(args.ckpt), f"{args.split}_pred.csv")
    if args.split == "test":
        build_submission(pred_by_id, out_csv)
    else:
        pd.DataFrame(
            {"Id": data.ids, "Answer": [str(p) for p in preds]}
        ).to_csv(out_csv, index=False)
        print(f"saved: {out_csv}")


def save_run(run_dir, cfg, metrics, model=None, curve=None):
    os.makedirs(run_dir, exist_ok=True)
    payload = {"config": {k: v for k, v in cfg.items()}, "metrics": metrics}
    if curve:
        payload["tau_curve"] = curve
    with open(os.path.join(run_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    if model is not None:
        torch.save({"state_dict": model.state_dict(), "cfg": cfg,
                    "tau": metrics.get("tau")}, os.path.join(run_dir, "ckpt.pt"))
    print(f"metrics: {json.dumps(metrics, ensure_ascii=False)}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--mode", required=True,
        choices=["holdout", "cv", "full", "predict", "decode-ablation"],
    )
    ap.add_argument("--variant", default=None, choices=["crop", "nocrop"])
    ap.add_argument("--no-events", action="store_true")
    ap.add_argument("--no-scalars", action="store_true")
    ap.add_argument("--noise-policy", default=None, choices=["a", "b", "c"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--tau", type=float, default=None)
    ap.add_argument("--run-name", default=None)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    if args.mode == "predict":
        run_predict(args)
        return
    if args.mode == "decode-ablation":
        run_decode_ablation(args)
        return

    cfg = load_config(
        variant=args.variant, epochs=args.epochs, seed=args.seed,
        noise_policy=args.noise_policy, tau=args.tau,
    )
    if args.no_events:
        cfg["use_events"] = False
    if args.no_scalars:
        cfg["use_scalars"] = False

    name = args.run_name or "_".join(
        [args.mode, cfg["variant"]]
        + (["noev"] if not cfg["use_events"] else [])
        + (["nosc"] if not cfg["use_scalars"] else [])
        + ([f"noise{cfg['noise_policy']}"] if cfg["noise_policy"] != "c" else [])
    )
    run_dir = os.path.join(load_paths()["outputs_dir"], "track_a", name)

    {"holdout": run_holdout, "cv": run_cv, "full": run_full}[args.mode](cfg, run_dir)


if __name__ == "__main__":
    main()
