"""rewards.py: GRPO EM 보상 함수 검증 (Phase 2 준비물).

보상 = 파싱가능 보너스 + EM − identity-collapse 페널티. plain 레시피 정합:
identity는 정답일 때 벌하지 않고(사전확률 0.155), 섞인 뷰에 identity를 뱉을 때만 감점.
"""

import pytest

from src.train.rewards import (
    EM_REWARD,
    IDENTITY_COLLAPSE_PENALTY,
    PARSE_BONUS,
    _as_rank,
    _completion_text,
    batch_rewards,
    make_grpo_reward,
    sequence_reward,
)
from src.train.targets import build_target
from src.utils.permutation import IDENTITY

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
