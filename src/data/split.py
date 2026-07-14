"""캡션 유사도 기반 train/val 분리.

동일/유사 비디오 유래 샘플(캡션 근접 중복: 코사인 >= 0.9가 26쌍, >= 0.8이 113쌍)이
train과 val에 갈라져 들어가는 누수를 막기 위해, TF-IDF 코사인 유사도 >= SIM_THRESHOLD
인 캡션 쌍을 union-find로 묶고 그룹 단위로 90/10 분리한다.

사용:
  python -m src.data.split          # -> outputs/split.csv (Id, fold)
"""

import os

import numpy as np
import pandas as pd

from src.data.loader import load_paths, load_split

SIM_THRESHOLD = 0.8
VAL_FRACTION = 0.1
SEED = 42


class _UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))

    def find(self, x):
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def caption_groups(sentences, threshold=SIM_THRESHOLD):
    """근접 중복 캡션을 같은 그룹으로 묶은 그룹 id 배열."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.neighbors import NearestNeighbors

    X = TfidfVectorizer(min_df=1).fit_transform(sentences)
    # 이웃 수는 넉넉히 잡되 임계값으로 거른다 (중복 그룹은 작다)
    nn = NearestNeighbors(n_neighbors=min(10, len(sentences)), metric="cosine").fit(X)
    dist, idx = nn.kneighbors(X)
    uf = _UnionFind(len(sentences))
    for i in range(len(sentences)):
        for d, j in zip(dist[i], idx[i]):
            if i != j and 1.0 - d >= threshold:
                uf.union(i, j)
    return np.array([uf.find(i) for i in range(len(sentences))])


def make_split(df, val_fraction=VAL_FRACTION, seed=SEED):
    """그룹 단위 셔플로 fold 컬럼('train'/'val')을 만들어 반환.

    No_ordering 비율이 두 fold에서 비슷하게 유지되는지는 호출측에서 확인한다
    (그룹이 작아 통계적으로 자연히 맞는다).
    """
    groups = caption_groups(df["Sentence"].tolist())
    rng = np.random.default_rng(seed)
    unique_groups = np.unique(groups)
    rng.shuffle(unique_groups)

    n_val_target = int(len(df) * val_fraction)
    val_groups = set()
    n_val = 0
    group_sizes = pd.Series(groups).value_counts()
    for g in unique_groups:
        if n_val >= n_val_target:
            break
        val_groups.add(g)
        n_val += int(group_sizes[g])

    fold = np.where(np.isin(groups, list(val_groups)), "val", "train")
    out = df[["Id"]].copy()
    out["fold"] = fold
    out["caption_group"] = groups
    return out


def main():
    paths = load_paths()
    df = load_split("train")
    split = make_split(df)
    os.makedirs(paths["outputs_dir"], exist_ok=True)
    out_csv = os.path.join(paths["outputs_dir"], "split.csv")
    split.to_csv(out_csv, index=False)

    merged = df.merge(split, on="Id")
    print(f"saved: {out_csv}")
    print(merged["fold"].value_counts().to_string())
    print("No_ordering rate by fold:")
    print(merged.groupby("fold")["No_ordering"].mean().round(4).to_string())
    # 누수 검증: 같은 caption_group이 두 fold에 걸치면 안 된다
    leak = merged.groupby("caption_group")["fold"].nunique()
    assert (leak == 1).all(), "caption group leaks across folds!"
    print("leak check: OK (no caption group spans both folds)")


if __name__ == "__main__":
    main()
