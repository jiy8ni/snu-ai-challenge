"""CLIP 기반 의미론적 피처: 프레임-프레임 유사도 + 프레임↔캡션 이벤트 매칭.

배경: phash·픽셀 통계·이벤트 수 같은 저수준 피처는 No_ordering 판별력이 전무했다
(AUC 0.49, reports/preprocessing.md). "같은 이벤트에서 뽑힌 프레임 쌍"은 카메라가
움직여 픽셀 수준에서는 다르므로, 의미 임베딩 수준의 유사도가 필요하다.

이 모듈은 전처리/분석 전용이다 (최종 추론 파이프라인에는 미포함, 앙상블 금지 규칙).

사용:
  python -m src.preprocess.semantic_features --n 1500          # 서브셋 피처 추출
  python -m src.preprocess.semantic_features --n 1500 --evaluate
"""

import argparse
import os

import numpy as np
import pandas as pd
import torch
from PIL import Image
from tqdm import tqdm

from src.data.loader import frame_paths, load_paths, load_split
from src.preprocess.caption_events import split_events

MODEL_NAME = "openai/clip-vit-base-patch32"

SEMANTIC_FEATURE_COLS = [
    "ff_max_sim",        # 프레임 쌍 최대 유사도 (의미적 근접 중복)
    "ff_mean_sim",
    "ff_min_sim",
    "fe_collision",      # 같은 이벤트에 argmax로 붙은 프레임 수의 최대값 (>=2면 충돌)
    "fe_min_maxsim",     # 프레임별 최고 이벤트 유사도의 최소값 (정합성 하한)
    "fe_mean_margin",    # 프레임별 1위-2위 이벤트 유사도 차 평균 (매칭 모호도)
    "fc_min_sim",        # 프레임-전체캡션 유사도 최소값 (노이즈 프레임 신호)
    "fc_mean_sim",
]


class ClipScorer:
    def __init__(self, model_name=MODEL_NAME, device=None):
        from transformers import CLIPModel, CLIPProcessor

        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device).eval()
        self.processor = CLIPProcessor.from_pretrained(model_name)

    @staticmethod
    def _to_tensor(out):
        # transformers v4는 텐서, v5는 BaseModelOutputWithPooling(pooler_output=투영 임베딩)
        return out if isinstance(out, torch.Tensor) else out.pooler_output

    @torch.no_grad()
    def embed_images(self, paths):
        images = [Image.open(p).convert("RGB") for p in paths]
        inputs = self.processor(images=images, return_tensors="pt").to(self.device)
        emb = self._to_tensor(self.model.get_image_features(**inputs))
        return torch.nn.functional.normalize(emb, dim=-1).cpu().numpy()

    @torch.no_grad()
    def embed_texts(self, texts):
        inputs = self.processor(
            text=texts, return_tensors="pt", padding=True, truncation=True, max_length=77
        ).to(self.device)
        emb = self._to_tensor(self.model.get_text_features(**inputs))
        return torch.nn.functional.normalize(emb, dim=-1).cpu().numpy()


def sample_semantic_features(scorer, paths, sentence):
    img = scorer.embed_images(paths)                       # (4, d)
    events = split_events(sentence) or [sentence]
    txt = scorer.embed_texts(events + [sentence])          # (E+1, d)
    ev, cap = txt[:-1], txt[-1]

    ff = img @ img.T                                       # 프레임-프레임
    iu = np.triu_indices(4, k=1)
    ff_pairs = ff[iu]

    fe = img @ ev.T                                        # (4, E) 프레임-이벤트
    assign = fe.argmax(axis=1)
    collision = int(np.bincount(assign, minlength=len(events)).max())
    maxsim = fe.max(axis=1)
    if fe.shape[1] >= 2:
        part = np.sort(fe, axis=1)
        margin = float((part[:, -1] - part[:, -2]).mean())
    else:
        margin = 0.0

    fc = img @ cap                                         # (4,) 프레임-전체캡션

    return {
        "ff_max_sim": float(ff_pairs.max()),
        "ff_mean_sim": float(ff_pairs.mean()),
        "ff_min_sim": float(ff_pairs.min()),
        "fe_collision": collision,
        "fe_min_maxsim": float(maxsim.min()),
        "fe_mean_margin": margin,
        "fc_min_sim": float(fc.min()),
        "fc_mean_sim": float(fc.mean()),
        "n_events": len(events),
    }


def build_features(split="train", n=None, seed=42):
    df = load_split(split)
    if n:
        df = df.sample(n=min(n, len(df)), random_state=seed)
    scorer = ClipScorer()
    rows = []
    for _, r in tqdm(df.iterrows(), total=len(df), desc=f"clip:{split}"):
        feat = {"Id": r["Id"]}
        feat.update(sample_semantic_features(scorer, frame_paths(r), r["Sentence"]))
        if "No_ordering" in df.columns:
            feat["No_ordering"] = bool(r["No_ordering"])
        rows.append(feat)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", default="train")
    ap.add_argument("--n", type=int, default=None, help="서브셋 크기 (미지정 시 전체)")
    ap.add_argument("--evaluate", action="store_true")
    args = ap.parse_args()

    paths = load_paths()
    tag = f"_{args.n}" if args.n else ""
    out_csv = os.path.join(paths["outputs_dir"], f"semantic_features_{args.split}{tag}.csv")

    if not os.path.exists(out_csv):
        feats = build_features(args.split, n=args.n)
        os.makedirs(paths["outputs_dir"], exist_ok=True)
        feats.to_csv(out_csv, index=False)
        print(f"saved: {out_csv} ({len(feats)} rows)")

    if args.evaluate:
        from src.preprocess.no_ordering import evaluate

        feats = pd.read_csv(out_csv)
        cols = SEMANTIC_FEATURE_COLS + ["n_events"]
        import src.preprocess.no_ordering as no_mod

        orig = no_mod.FEATURE_COLS
        no_mod.FEATURE_COLS = cols
        try:
            results = evaluate(feats)
        finally:
            no_mod.FEATURE_COLS = orig
        for name, r in results.items():
            print(f"{name}: AUC {r['auc_mean']:.4f} (+/- {r['auc_std']:.4f}), F1 {r['f1_mean']:.4f}")


if __name__ == "__main__":
    main()
