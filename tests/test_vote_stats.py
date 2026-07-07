"""vote_stats: 투표 프로파일·합의도·예상 LB 투영 검증 (파일 IO는 tmp_path)."""

import json

import pandas as pd

from src.eval.vote_stats import load_vote_df, profile, project_lb
from src.train.targets import build_target
from src.utils.permutation import shuffle_rank_label

ID_PERM = [0, 1, 2, 3]


def _write_raw(tmp_path, rows):
    p = tmp_path / "raw.jsonl"
    with open(p, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(p)


def _text(rank_shuffled):
    return build_target("mid", None, rank_shuffled, no_ordering=False)


def test_load_vote_df_consensus_and_fail(tmp_path):
    rank = [2, 1, 4, 3]
    perm = [1, 0, 3, 2]
    rows = [
        # s1: 3표 일치(그중 1표는 셔플 뷰) + 1표 parse fail -> consensus 3
        {"Id": "s1", "perm": ID_PERM, "text": _text(rank)},
        {"Id": "s1", "perm": ID_PERM, "text": _text(rank)},
        {"Id": "s1", "perm": perm, "text": _text(shuffle_rank_label(rank, perm))},
        {"Id": "s1", "perm": ID_PERM, "text": "garbage"},
        # s2: 전부 fail -> consensus 0
        {"Id": "s2", "perm": ID_PERM, "text": "no list"},
        {"Id": "s2", "perm": ID_PERM, "text": ""},
    ]
    df = load_vote_df(_write_raw(tmp_path, rows)).set_index("Id")
    assert df.loc["s1", "consensus"] == 3
    assert df.loc["s1", "n_fail"] == 1
    assert df.loc["s1", "pred"] == rank and df.loc["s1", "reason"] == "mode"
    assert df.loc["s2", "consensus"] == 0
    assert df.loc["s2", "reason"] == "parse_fail" and df.loc["s2", "pred_identity"]


def test_load_vote_df_unorderable_votes(tmp_path):
    unord = build_target("mid", None, None, no_ordering=True)
    rows = [
        {"Id": "s1", "perm": ID_PERM, "text": unord},
        {"Id": "s1", "perm": ID_PERM, "text": unord},
        {"Id": "s1", "perm": ID_PERM, "text": _text([2, 1, 4, 3])},
    ]
    df = load_vote_df(_write_raw(tmp_path, rows))
    assert df.iloc[0]["n_unorderable"] == 2
    assert df.iloc[0]["reason"] == "unorderable_majority"


def test_profile_rates(tmp_path):
    rank = [2, 1, 4, 3]
    rows = [
        {"Id": "s1", "perm": ID_PERM, "text": _text(rank)},
        {"Id": "s1", "perm": ID_PERM, "text": _text(rank)},
        {"Id": "s2", "perm": ID_PERM, "text": "garbage"},
        {"Id": "s2", "perm": ID_PERM, "text": _text([1, 2, 3, 4])},
    ]
    p = profile(load_vote_df(_write_raw(tmp_path, rows)))
    assert p["n_samples"] == 2
    assert p["parse_fail_rate"] == 0.25          # 4표 중 1표 fail
    assert abs(p["identity_vote_rate"] - 1 / 3) < 1e-3  # orderable 3표 중 identity 1표 (round 4)
    assert p["consensus_dist"] == {1: 0.5, 2: 0.5}


def test_project_lb_weighted_transfer():
    """val P(정답|합의도) x test 합의도 분포 가중합 + 미관측 수준은 전체 EM 폴백."""
    val = pd.DataFrame({"consensus": [4, 4, 2, 2], "em": [True, True, True, False]})
    # val: EM|4 = 1.0, EM|2 = 0.5, 전체 0.75
    test = pd.DataFrame({"consensus": [4, 4, 4, 2, 3, 3, 3, 3]})
    # test 분포: 4 -> 3/8, 2 -> 1/8, 3(미관측) -> 4/8
    out = project_lb(val, test)
    expected = (3 / 8) * 1.0 + (1 / 8) * 0.5 + (4 / 8) * 0.75
    assert abs(out["projected_lb"] - expected) < 1e-6
    assert out["val_em"] == 0.75
