"""em.py 지표 + submission.py 형식 게이트 검증."""

import pandas as pd
import pytest

from src.data.loader import load_split
from src.eval.em import evaluate, kendall_tau
from src.infer.submission import build_submission, validate_submission
from src.utils.permutation import IDENTITY


def test_kendall_tau_bounds():
    assert kendall_tau([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0
    assert kendall_tau([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0


def test_evaluate_decomposition():
    truth = pd.DataFrame(
        {
            "Id": ["a", "b", "c"],
            "Answer": ["[2, 1, 3, 4]", "[1, 2, 3, 4]", "[3, 4, 1, 2]"],
            "No_ordering": [False, True, False],
        }
    )
    preds = {"a": [2, 1, 3, 4], "b": [1, 2, 3, 4], "c": [4, 3, 1, 2]}
    m = evaluate(preds, truth)
    assert m["em"] == pytest.approx(2 / 3)
    assert m["em_orderable"] == pytest.approx(1 / 2)
    assert m["em_no_ordering"] == 1.0
    assert m["identity_rate"] == pytest.approx(1 / 3)
    assert m["gate_precision"] == 1.0 and m["gate_recall"] == 1.0


def test_evaluate_rejects_missing_or_invalid():
    truth = pd.DataFrame({"Id": ["a"], "Answer": ["[1, 2, 3, 4]"], "No_ordering": [True]})
    with pytest.raises(AssertionError):
        evaluate({}, truth)
    with pytest.raises(AssertionError):
        evaluate({"a": [1, 1, 2, 3]}, truth)


def test_build_submission_roundtrip(tmp_path):
    test_ids = list(load_split("test")["Id"])
    pred = {i: [2, 1, 4, 3] for i in test_ids}
    pred[test_ids[0]] = list(IDENTITY)
    out = tmp_path / "submission.csv"
    summary = build_submission(pred, str(out))
    assert summary["n"] == len(test_ids)
    assert validate_submission(str(out))

    sub = pd.read_csv(out)
    assert sub.loc[0, "Answer"] == "[1, 2, 3, 4]"
    assert sub.loc[1, "Answer"] == "[2, 1, 4, 3]"


def test_build_submission_rejects_bad_perm(tmp_path):
    test_ids = list(load_split("test")["Id"])
    pred = {i: [2, 1, 4, 3] for i in test_ids}
    pred[test_ids[5]] = [1, 1, 2, 3]
    with pytest.raises(AssertionError):
        build_submission(pred, str(tmp_path / "bad.csv"))
