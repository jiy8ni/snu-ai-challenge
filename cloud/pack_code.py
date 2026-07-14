"""repo 코드 + 설정 + SFT 타깃을 Kaggle Dataset용 zip으로 묶는다 (본선 재현성).

대회 이미지 데이터는 절대 포함하지 않는다 (Kaggle 대회 소스로 attach). 모델 가중치도
용량상 별도 Dataset으로 올린다. 이 zip은 src/, configs/, cloud/, colab/,
outputs/sft_*.jsonl, outputs/split.csv, docs/, tests/ 만 담는다.

주의: 이 폴더는 pip의 kaggle CLI 패키지와 임포트 충돌을 피해 'kaggle'에서 'cloud'로 개명됨.

사용:
  python -m cloud.pack_code                        # -> outputs/snuai_code.zip
그 후 Kaggle에서 "New Dataset" -> zip 업로드 -> 노트북에 attach.
"""

import os
import zipfile

from src.data.loader import load_paths

INCLUDE_DIRS = ["src", "configs", "cloud", "colab", "docs", "tests"]
INCLUDE_FILES = [
    "outputs/split.csv",
    "outputs/sft_train.jsonl",
    "outputs/sft_val.jsonl",
    "PLAN.md",
]
EXCLUDE_SUFFIX = (".pyc",)
EXCLUDE_DIRS = {"__pycache__"}
# 인증키는 어떤 경우에도 번들에 넣지 않는다
EXCLUDE_NAMES = {"kaggle.json"}


def iter_files(root):
    for base, dirs, files in os.walk(root):
        dirs[:] = [d for d in dirs if d not in EXCLUDE_DIRS]
        for name in files:
            if name in EXCLUDE_NAMES:
                continue
            if not name.endswith(EXCLUDE_SUFFIX):
                yield os.path.join(base, name)


def main():
    paths = load_paths()
    repo_root = os.path.dirname(paths["outputs_dir"])
    out_zip = os.path.join(paths["outputs_dir"], "snuai_code.zip")

    written = 0
    with zipfile.ZipFile(out_zip, "w", zipfile.ZIP_DEFLATED) as zf:
        for d in INCLUDE_DIRS:
            for path in iter_files(os.path.join(repo_root, d)):
                zf.write(path, os.path.relpath(path, repo_root))
                written += 1
        for rel in INCLUDE_FILES:
            path = os.path.join(repo_root, rel)
            if os.path.exists(path):
                zf.write(path, rel)
                written += 1
            else:
                print(f"[경고] 누락 (건너뜀): {rel}")

    size_mb = os.path.getsize(out_zip) / 1e6
    print(f"saved: {out_zip} ({written} files, {size_mb:.1f} MB)")
    print("다음: Kaggle > Datasets > New Dataset 으로 이 zip 업로드 후 노트북에 attach")


if __name__ == "__main__":
    main()
