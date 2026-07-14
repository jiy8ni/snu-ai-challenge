"""제출 CSV 생성 + 사전 검증 (제출 게이트).

모든 제출 파일은 반드시 이 모듈을 거친다:
  - test.csv의 Id 전수·순서 일치
  - 모든 행이 {1,2,3,4}의 유효 순열
  - identity([1,2,3,4]) 비율이 기대 범위(5~30%) 밖이면 경고 (train 기준 ~15.5%)

사용:
  from src.infer.submission import build_submission
  build_submission(pred_by_id, "outputs/submission.csv")
"""

import pandas as pd

from src.data.loader import load_paths, load_split
from src.train.targets import format_answer
from src.utils.permutation import IDENTITY, is_valid_permutation, parse_answer_column

IDENTITY_RATE_RANGE = (0.05, 0.30)


def build_submission(pred_by_id, out_path, split="test"):
    """pred_by_id: {Id: rank list} -> 검증 후 CSV 저장. 반환: 요약 dict."""
    ref = load_split(split)
    ids = list(ref["Id"])

    missing = [i for i in ids if i not in pred_by_id]
    assert not missing, f"예측 누락 {len(missing)}건: {missing[:5]}..."

    answers = []
    for i in ids:
        rank = list(pred_by_id[i])
        assert is_valid_permutation(rank), f"{i}: 유효하지 않은 순열 {rank}"
        answers.append(format_answer(rank))

    identity_rate = sum(a == format_answer(IDENTITY) for a in answers) / len(answers)
    lo, hi = IDENTITY_RATE_RANGE
    if not lo <= identity_rate <= hi:
        # cp949 콘솔 호환을 위해 ASCII/한글만 사용 (em-dash 등 금지)
        print(f"[경고] identity 비율 {identity_rate:.1%}: 기대 범위({lo:.0%}~{hi:.0%}) 밖. "
              "게이트/디코딩 점검 필요. (tau 캘리브가 identity 다수 베팅을 택한 경우는 정상)")

    df = pd.DataFrame({"Id": ids, "Answer": answers})
    df.to_csv(out_path, index=False)
    summary = {"n": len(df), "identity_rate": identity_rate, "path": out_path}
    print(f"saved: {out_path} (n={len(df)}, identity={identity_rate:.1%})")
    validate_submission(out_path, split=split)
    return summary


def validate_submission(path, split="test"):
    """저장된 CSV를 다시 읽어 형식을 재검증 (제출 직전 최종 게이트)."""
    sub = pd.read_csv(path)
    ref = load_split(split)
    assert list(sub.columns) == ["Id", "Answer"], f"컬럼 불일치: {list(sub.columns)}"
    assert len(sub) == len(ref), f"행 수 불일치: {len(sub)} != {len(ref)}"
    assert list(sub["Id"]) == list(ref["Id"]), "Id 집합/순서가 test.csv와 다름"
    for _, r in sub.iterrows():
        parse_answer_column(r["Answer"])  # 유효 순열 아니면 raise
    return True


def main():
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--check", required=True, help="검증할 제출 CSV 경로")
    ap.add_argument("--split", default="test")
    args = ap.parse_args()
    validate_submission(args.check, split=args.split)
    print("OK: 형식 검증 통과")


if __name__ == "__main__":
    main()
