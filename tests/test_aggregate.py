"""TTA 집계: 역변환·투표·identity 금지·tie-break 검증."""

import random

from src.infer.aggregate import aggregate_votes, parse_vote
from src.train.targets import build_target
from src.utils.permutation import IDENTITY, shuffle_rank_label

EVENTS = ["a", "b", "c"]


def _vote_for(true_rank, perm, style="mid"):
    """true_rank를 아는 오라클 모델이 perm 셔플 입력에 내놓을 출력 -> parse_vote 결과."""
    shuffled_rank = shuffle_rank_label(true_rank, perm)
    text = build_target(style, EVENTS, shuffled_rank, no_ordering=False)
    return parse_vote(text, perm)


def test_tta_roundtrip_recovers_true_rank():
    """셔플 좌표로 답한 완벽한 모델의 TTA 투표는 원 좌표 rank로 만장일치."""
    true_rank = [3, 1, 4, 2]
    rng = random.Random(0)
    votes = []
    for _ in range(6):
        perm = list(range(4))
        rng.shuffle(perm)
        votes.append(_vote_for(true_rank, perm))
    assert all(k == "orderable" and r == true_rank for k, r in votes)
    assert aggregate_votes(votes) == (true_rank, "mode")


def test_unorderable_votes_are_not_unshuffled():
    """UNORDERABLE의 [1,2,3,4]는 관례값 — 역변환 없이 unorderable 투표로 계수."""
    perm = [2, 0, 3, 1]
    text = build_target("mid", EVENTS, None, no_ordering=True)
    kind, rank = parse_vote(text, perm)
    assert kind == "unorderable" and rank is None


def test_unorderable_majority():
    votes = [("unorderable", None), ("unorderable", None), ("orderable", [2, 1, 3, 4])]
    assert aggregate_votes(votes) == (IDENTITY, "unorderable_majority")


def test_orderable_majority_wins():
    votes = [("unorderable", None), ("orderable", [2, 1, 3, 4]), ("orderable", [2, 1, 3, 4])]
    assert aggregate_votes(votes) == ([2, 1, 3, 4], "mode")


def test_identity_allowed_by_default_banned_on_request():
    """기본(2026-07-05 val 스윕): identity도 정상 후보. --ban-identity로만 구 정책."""
    votes = [
        ("orderable", list(IDENTITY)),
        ("orderable", list(IDENTITY)),
        ("orderable", [2, 1, 3, 4]),
    ]
    assert aggregate_votes(votes) == (IDENTITY, "mode")
    # 구 정책 재현은 분산 게이트도 꺼야 함 (금지 후 남는 후보가 1표뿐이라 게이트에 걸림)
    assert aggregate_votes(votes, ban_identity=True, disperse_gate=False) == ([2, 1, 3, 4], "mode")


def test_all_identity_votes():
    votes = [("orderable", list(IDENTITY))] * 3
    assert aggregate_votes(votes) == (IDENTITY, "mode")
    assert aggregate_votes(votes, ban_identity=True) == (IDENTITY, "all_identity_votes")


def test_disperse_gate_no_consensus_falls_back_to_identity():
    """합의 전무(최빈 표수 1) -> identity. 게이트 끄면 borda tie-break로 진행."""
    votes = [
        ("orderable", [2, 1, 3, 4]),
        ("orderable", [1, 2, 4, 3]),
        ("orderable", [2, 1, 4, 3]),
    ]
    assert aggregate_votes(votes) == (IDENTITY, "disperse_gate")
    assert aggregate_votes(votes, disperse_gate=False)[1] == "borda_tie"


def test_parse_fail_fallback():
    assert aggregate_votes([("fail", None), ("fail", None)]) == (IDENTITY, "parse_fail")


def test_tie_break_is_deterministic_and_borda_guided():
    # 2표 동률 두 후보 + 평균을 [2,1,3,4] 쪽으로 미는 제5 투표 (분산 게이트 미발동)
    votes = [
        ("orderable", [2, 1, 3, 4]),
        ("orderable", [2, 1, 3, 4]),
        ("orderable", [1, 2, 4, 3]),
        ("orderable", [1, 2, 4, 3]),
        ("orderable", [2, 1, 4, 3]),
    ]
    rank, reason = aggregate_votes(votes)
    assert reason == "borda_tie"
    assert sorted(rank) == IDENTITY
    # 재실행해도 같은 결과 (결정적)
    assert aggregate_votes(votes)[0] == rank


def test_parse_vote_fail():
    assert parse_vote("no list here", [0, 1, 2, 3]) == ("fail", None)


def test_disperse_top_scales_gate_to_vote_count():
    """tta8 대응: 최빈 표수 2까지도 '합의 부족'으로 게이트 (문턱 파라미터화)."""
    votes = [
        ("orderable", [2, 1, 3, 4]), ("orderable", [2, 1, 3, 4]),
        ("orderable", [1, 2, 4, 3]), ("orderable", [1, 2, 4, 3]),
        ("orderable", [2, 1, 4, 3]), ("orderable", [3, 1, 4, 2]),
        ("orderable", [4, 1, 3, 2]), ("orderable", [1, 4, 3, 2]),
    ]
    assert aggregate_votes(votes)[1] == "borda_tie"  # 기본 문턱 1은 top=2라 통과
    assert aggregate_votes(votes, disperse_top=2) == (IDENTITY, "disperse_gate")
    # 문턱보다 강한 합의(top=3)는 게이트를 통과한다
    votes[3] = ("orderable", [2, 1, 3, 4])
    assert aggregate_votes(votes, disperse_top=2) == ([2, 1, 3, 4], "mode")


def test_identity_quota_overrides_mode():
    """identity 표가 쿼터 이상이면 최빈이 아니어도 identity (no_ordering 신호)."""
    votes = [
        ("orderable", list(IDENTITY)), ("orderable", list(IDENTITY)),
        ("orderable", [2, 1, 3, 4]), ("orderable", [2, 1, 3, 4]),
        ("orderable", [2, 1, 3, 4]),
    ]
    assert aggregate_votes(votes) == ([2, 1, 3, 4], "mode")
    assert aggregate_votes(votes, identity_quota=2) == (IDENTITY, "identity_quota")
    assert aggregate_votes(votes, identity_quota=3) == ([2, 1, 3, 4], "mode")
