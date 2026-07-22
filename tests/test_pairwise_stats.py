"""pairwise_stats.analyze: 오답 일치쌍 분포 집계 (소형 고정 케이스)."""

import pytest

from src.eval.pairwise_stats import analyze


def _row(id_, em, pred, truth, no_ordering=False):
    return {
        "Id": id_, "em": str(em), "no_ordering": str(no_ordering),
        "pred": str(pred), "truth": str(truth),
    }


def test_analyze_filters_wrong_and_bins_concordant():
    rows = [
        _row("a", True, [2, 3, 1, 4], [2, 3, 1, 4]),           # 정답 → 제외
        _row("b", False, [1, 3, 2, 4], [2, 3, 1, 4]),          # 오답 c=5
        _row("c", False, [1, 4, 2, 3], [2, 3, 1, 4]),          # 오답 c=4
        _row("d", False, [3, 1, 4, 2], [2, 3, 1, 4]),          # 오답 c=1
    ]
    out = analyze(rows)
    assert out["n_rows"] == 4
    assert out["n_wrong"] == 3
    assert out["overall"]["concordant_hist"] == {0: 0, 1: 1, 2: 0, 3: 0, 4: 1, 5: 1}
    assert out["overall"]["near_miss_frac_c5"] == pytest.approx(1 / 3)


def test_analyze_separates_no_ordering():
    rows = [
        _row("a", False, [1, 3, 2, 4], [2, 3, 1, 4], no_ordering=True),
        _row("b", False, [1, 4, 2, 3], [2, 3, 1, 4], no_ordering=False),
    ]
    out = analyze(rows)
    assert out["no_ordering"]["n"] == 1
    assert out["orderable"]["n"] == 1


def test_analyze_identity_prediction_counted():
    rows = [_row("a", False, [1, 2, 3, 4], [2, 3, 1, 4])]     # 섞인 뷰에 identity 예측
    out = analyze(rows)
    assert out["overall"]["identity_pred_frac"] == pytest.approx(1.0)


def test_analyze_empty_wrong_recommends_no_shaping():
    rows = [_row("a", True, [1, 2, 3, 4], [1, 2, 3, 4])]
    out = analyze(rows)
    assert out["n_wrong"] == 0
    assert "shaping 불필요" in out["recommendation"]
