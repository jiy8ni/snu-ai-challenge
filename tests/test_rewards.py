"""rewards.py: GRPO EM 보상 함수 검증 (Phase 2 준비물).

보상 = 파싱가능 보너스 + EM − identity-collapse 페널티. plain 레시피 정합:
identity는 정답일 때 벌하지 않고(사전확률 0.155), 섞인 뷰에 identity를 뱉을 때만 감점.
"""

import pytest

from src.eval.em import kendall_tau
from src.train.rewards import (
    EM_REWARD,
    IDENTITY_COLLAPSE_PENALTY,
    PAIRWISE_LAMBDA,
    PARSE_BONUS,
    _as_rank,
    _completion_text,
    batch_pairwise_rewards,
    batch_rewards,
    make_grpo_pairwise_reward,
    make_grpo_reward,
    pairwise_concordant,
    pairwise_reward,
    sequence_reward,
)
from src.train.targets import build_target
from src.utils.permutation import ALL_PERMUTATIONS, IDENTITY

EVENTS = ["a man opens a door", "he walks in", "he sits down"]
CORRECT = PARSE_BONUS + EM_REWARD              # 1.1
WRONG = PARSE_BONUS                            # 0.1
PARSE_FAIL = 0.0
COLLAPSE = PARSE_BONUS - IDENTITY_COLLAPSE_PENALTY   # -0.1


def _plain(rank):
    """plain 포맷 completion 텍스트 (실제 학습 타깃과 동일 경로)."""
    return build_target("plain", EVENTS, rank, no_ordering=False)


def test_correct_non_identity():
    assert sequence_reward(_plain([2, 3, 1, 4]), [2, 3, 1, 4]) == pytest.approx(CORRECT)


def test_correct_identity_is_not_penalized():
    """진실이 identity(이미 시간순)이고 identity를 맞히면 만점 — 붕괴 페널티 없음."""
    assert sequence_reward(_plain(IDENTITY), IDENTITY) == pytest.approx(CORRECT)


def test_wrong_non_identity():
    assert sequence_reward(_plain([4, 3, 2, 1]), [2, 3, 1, 4]) == pytest.approx(WRONG)


def test_identity_collapse_on_orderable_is_penalized():
    """섞인 뷰(진실 비-identity)에 identity를 뱉으면 파싱실패보다도 낮게."""
    assert sequence_reward(_plain(IDENTITY), [2, 3, 1, 4]) == pytest.approx(COLLAPSE)


def test_parse_failure_gets_floor():
    assert sequence_reward("I cannot determine the order.", [2, 3, 1, 4]) == PARSE_FAIL
    assert sequence_reward("", [2, 3, 1, 4]) == PARSE_FAIL


def test_reward_ordering_invariant():
    """정답 > 오답·비identity > 파싱실패 > orderable에 identity 붕괴."""
    true = [2, 3, 1, 4]
    correct = sequence_reward(_plain(true), true)
    wrong = sequence_reward(_plain([3, 4, 1, 2]), true)
    fail = sequence_reward("no list here", true)
    collapse = sequence_reward(_plain(IDENTITY), true)
    assert correct > wrong > fail > collapse


def test_answer_string_true_rank_accepted():
    """true_rank가 train.csv Answer 문자열이어도 동작."""
    assert sequence_reward(_plain([2, 3, 1, 4]), "[2, 3, 1, 4]") == pytest.approx(CORRECT)


def test_custom_shaping_params():
    true = [2, 3, 1, 4]
    assert sequence_reward(_plain(IDENTITY), true, identity_collapse_penalty=0.5) == pytest.approx(-0.4)
    assert sequence_reward(_plain(true), true, em_reward=2.0) == pytest.approx(PARSE_BONUS + 2.0)


def test_as_rank_roundtrip_and_validation():
    assert _as_rank([3, 1, 2, 4]) == [3, 1, 2, 4]
    assert _as_rank("[3, 1, 2, 4]") == [3, 1, 2, 4]
    with pytest.raises(AssertionError):
        _as_rank([1, 2, 3, 3])   # 순열 아님


def test_batch_rewards_maps_and_checks_length():
    texts = [_plain([2, 3, 1, 4]), "garbage"]
    assert batch_rewards(texts, [[2, 3, 1, 4], [1, 2, 3, 4]]) == pytest.approx([CORRECT, PARSE_FAIL])
    with pytest.raises(AssertionError):
        batch_rewards(texts, [[2, 3, 1, 4]])


def test_completion_text_handles_plain_and_chat():
    assert _completion_text("Answer: [1, 2, 3, 4]") == "Answer: [1, 2, 3, 4]"
    chat = [{"role": "assistant", "content": [{"type": "text", "text": "Answer: [2, 3, 1, 4]"}]}]
    assert "[2, 3, 1, 4]" in _completion_text(chat)
    chat_str = [{"role": "assistant", "content": "Answer: [4, 3, 2, 1]"}]
    assert "[4, 3, 2, 1]" in _completion_text(chat_str)


def test_make_grpo_reward_adapter():
    """TRL 호출 규약: reward_func(prompts=, completions=, **columns) → list[float]."""
    reward = make_grpo_reward("true_rank")
    completions = [
        [{"role": "assistant", "content": [{"type": "text", "text": _plain([2, 3, 1, 4])}]}],
        [{"role": "assistant", "content": _plain(IDENTITY)}],
    ]
    out = reward(prompts=[None, None], completions=completions, true_rank=[[2, 3, 1, 4], [2, 3, 1, 4]])
    assert out == pytest.approx([CORRECT, COLLAPSE])


def test_make_grpo_reward_passes_shaping_kw():
    reward = make_grpo_reward("true_rank", identity_collapse_penalty=0.5)
    completions = [_plain(IDENTITY)]
    assert reward(completions=completions, true_rank=[[2, 3, 1, 4]]) == pytest.approx([-0.4])


# ---------------------------------------------------------------------------
# pairwise shaping 보상 (별도 보상 함수, TRL reward_weights로 합성)
# ---------------------------------------------------------------------------

TRUE = [2, 3, 1, 4]
# TRUE 기준 일치쌍 수를 통제한 예측들 (아래 test_pairwise_concordant_known_values로 고정):
PRED_C6 = [2, 3, 1, 4]   # 완전일치
PRED_C5 = [1, 3, 2, 4]   # 인접 값(1,2) 스왑 → 1쌍 반전
PRED_C4 = [1, 4, 2, 3]   # 값(1,2)+(3,4) 스왑 → 2쌍 반전
PRED_C1 = [3, 1, 4, 2]   # 거의 역순 → 우연 수준 아래


def test_pairwise_concordant_known_values():
    assert pairwise_concordant([1, 2, 3, 4], [1, 2, 3, 4]) == 6   # 완전일치
    assert pairwise_concordant([2, 1, 3, 4], [1, 2, 3, 4]) == 5   # 인접 스왑
    assert pairwise_concordant([4, 3, 2, 1], [1, 2, 3, 4]) == 0   # 완전역순
    assert pairwise_concordant(PRED_C4, TRUE) == 4
    assert pairwise_concordant(PRED_C1, TRUE) == 1


def test_pairwise_matches_kendall_tau():
    """수학적 동치: (concordant − 3) / 3 == kendall_tau."""
    for pred in ALL_PERMUTATIONS:
        c = pairwise_concordant(list(pred), TRUE)
        assert (c - 3) / 3 == pytest.approx(kendall_tau(list(pred), TRUE))


def test_pairwise_reward_centered_and_clamped():
    assert pairwise_reward(_plain(PRED_C6), TRUE) == pytest.approx(1.0)      # (6-3)/3
    assert pairwise_reward(_plain(PRED_C5), TRUE) == pytest.approx(2 / 3)    # (5-3)/3
    assert pairwise_reward(_plain(PRED_C4), TRUE) == pytest.approx(1 / 3)    # (4-3)/3
    assert pairwise_reward(_plain(PRED_C1), TRUE) == pytest.approx(0.0)      # 음수 클램프


def test_pairwise_reward_parse_failure_is_zero():
    assert pairwise_reward("no list here", TRUE) == 0.0


def test_pairwise_reward_identity_prediction_excluded():
    """섞인 뷰에 identity를 뱉으면 근접 뷰라도 shaping 수확 0 (collapse 고정 목적)."""
    near = [1, 2, 4, 3]                                  # identity와 1쌍 차이 (c=5)
    assert pairwise_concordant(IDENTITY, near) == 5
    assert pairwise_reward(_plain(IDENTITY), near) == 0.0


def test_pairwise_reward_identity_is_rewarded_when_true():
    """진실이 identity면 identity 예측은 정상적으로 만점 shaping."""
    assert pairwise_reward(_plain(IDENTITY), IDENTITY) == pytest.approx(1.0)


def _composite(text, true, lam=PAIRWISE_LAMBDA):
    """TRL reward_weights=[1.0, λ] 합성 보상."""
    return sequence_reward(text, true) + lam * pairwise_reward(text, true)


def test_composite_reward_strict_ordering():
    """완전일치 > c5 오답 > c4 오답 > c≤3 오답 > 파싱실패 > identity 붕괴 (λ=0.25)."""
    correct = _composite(_plain(PRED_C6), TRUE)
    near5 = _composite(_plain(PRED_C5), TRUE)
    near4 = _composite(_plain(PRED_C4), TRUE)
    far = _composite(_plain(PRED_C1), TRUE)
    fail = _composite("no list here", TRUE)
    collapse = _composite(_plain(IDENTITY), TRUE)
    assert correct > near5 > near4 > far > fail > collapse
    assert correct == pytest.approx(1.35)
    assert collapse == pytest.approx(-0.1)   # pairwise 제외로 collapse는 항상 −0.1 고정


def test_batch_pairwise_rewards_maps_and_checks_length():
    texts = [_plain(PRED_C6), "garbage"]
    assert batch_pairwise_rewards(texts, [TRUE, IDENTITY]) == pytest.approx([1.0, 0.0])
    with pytest.raises(AssertionError):
        batch_pairwise_rewards(texts, [TRUE])


def test_make_grpo_pairwise_reward_adapter():
    """TRL 규약: reward_func(prompts=, completions=, **columns) → list[float], 이름 태그."""
    reward = make_grpo_pairwise_reward("true_rank")
    assert reward.__name__ == "grpo_pairwise_reward"
    completions = [
        [{"role": "assistant", "content": [{"type": "text", "text": _plain(PRED_C5)}]}],
        [{"role": "assistant", "content": _plain(IDENTITY)}],
    ]
    out = reward(prompts=[None, None], completions=completions, true_rank=[TRUE, TRUE])
    assert out == pytest.approx([2 / 3, 0.0])   # c5 shaping, identity 붕괴 제외
