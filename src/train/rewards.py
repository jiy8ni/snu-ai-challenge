"""GRPO 보상 함수 (Phase 2). SFT의 간접 우도 최적화 대신 EM을 직접 보상으로 준다.

설계 출처: 계획 e2-purring-island.md §L1 ("EM 보상·단일 체크포인트"). 핵심 원칙:
  - **순수 함수**로 작성해 GRPO 프레임워크(TRL/Unsloth)에 비의존 — 로컬 CPU에서 단위
    테스트 가능. 프레임워크 연결은 make_grpo_reward의 얇은 어댑터가 담당하며, 이 모듈은
    trl/unsloth를 import하지 않는다.
  - plain 레시피와 정합: 출력은 parse_permutation으로 rank를 뽑고, identity([1,2,3,4])는
    "이미 시간순"인 정당한 답(사전확률 0.155)이므로 **정답일 때는 벌하지 않는다**.
  - anti-identity-collapse: 진실 라벨이 비-identity(=섞인 뷰)인데 identity를 뱉으면 감점 —
    구 mid 라운드를 망친 "특정 불가 시 무조건 identity" 붕괴를 직접 억제한다. plain은 구조적
    으로 이 붕괴가 약하지만, GRPO의 그룹-상대 어드밴티지에서 identity 표류를 명시적으로 누른다.

보상 = 파싱가능 보너스 + EM(완전일치) − identity-collapse 페널티.
기본값 구간: 정답 1.1 > 오답·비identity 0.1 > 파싱실패 0.0 > orderable 뷰에 identity −0.1.

**pairwise shaping (별도 보상 함수)**: EM만으로는 프롬프트의 8개 생성이 전부 오답일 때 보상이
동점이라 GRPO 어드밴티지가 0이 되어(희소 보상) hard case에서 학습 신호가 사라진다. pairwise_reward
는 C(4,2)=6쌍의 정답 선후관계 일치 수를 우연 수준(3쌍)으로 센터링해 "거의 맞은 오답"과 "완전 오답"을
차등한다. EM 보상과 **합성하지 않고** 별도 함수로 두어, TRL이 reward_weights로 λ를 조정하고 두 성분을
개별 로깅하게 한다 (make_grpo_reward + make_grpo_pairwise_reward를 함께 등록).
"""

from src.utils.permutation import (
    IDENTITY,
    is_valid_permutation,
    parse_answer_column,
    parse_permutation,
)

# 보상 구성요소 기본값 (계획 L1: EM +1, 포맷 유효 +0.1, collapse 페널티).
EM_REWARD = 1.0
PARSE_BONUS = 0.1
IDENTITY_COLLAPSE_PENALTY = 0.2

# pairwise shaping: 4프레임 6쌍에서 무작위 순열의 기대 일치쌍 = 3 (우연 수준 센터링 기준점).
CHANCE_CONCORDANT = 3
# 기본 λ (권장 작동점). 실제 가중은 GRPOConfig(reward_weights=[1.0, λ])가 담당하며 config에서 조정.
PAIRWISE_LAMBDA = 0.25


def _as_rank(value):
    """list[int] 또는 "[1, 2, 3, 4]" 문자열을 검증된 rank 리스트로 정규화."""
    if isinstance(value, str):
        return parse_answer_column(value)
    assert is_valid_permutation(value), f"invalid rank: {value}"
    return list(value)


def sequence_reward(text, true_rank, em_reward=EM_REWARD, parse_bonus=PARSE_BONUS,
                    identity_collapse_penalty=IDENTITY_COLLAPSE_PENALTY):
    """모델 출력 텍스트 하나에 대한 스칼라 보상.

    text: 생성된 completion (plain 포맷 "Order: ...\\nAnswer: [..]"). parse_permutation이
      마지막 유효 순열을 뽑는다 — 실패하면 파싱 불가로 본다.
    true_rank: 이 프롬프트(뷰)의 진실 rank (list[int] 또는 Answer 문자열).

    반환(기본값): 정답 1.1 / 오답·비identity 0.1 / 파싱실패 0.0 / orderable에 identity −0.1.
    마지막 순서가 anti-collapse 그래디언트를 만든다 (identity 붕괴를 파싱실패보다도 낮게 둔다).
    """
    true_rank = _as_rank(true_rank)
    pred = parse_permutation(text)
    if pred is None:
        return 0.0                       # 파싱 불가 — 보너스 없음 (플로어)
    reward = parse_bonus
    if pred == true_rank:
        reward += em_reward              # 완전일치 = 대회 지표 그 자체
    if pred == IDENTITY and true_rank != IDENTITY:
        reward -= identity_collapse_penalty   # 섞인 뷰인데 identity — 붕괴 억제
    return reward


def batch_rewards(texts, true_ranks, **kw):
    """길이가 같은 texts/true_ranks에 sequence_reward를 매핑한다."""
    assert len(texts) == len(true_ranks), f"length mismatch {len(texts)} != {len(true_ranks)}"
    return [sequence_reward(t, r, **kw) for t, r in zip(texts, true_ranks)]


def pairwise_concordant(pred, true):
    """두 4-rank 순열의 일치 쌍 수 (0..6). C(4,2)=6쌍 각각 선후관계 부호가 같으면 센다.

    src/eval/em.kendall_tau와 동일 로직: kendall_tau(pred, true) == (concordant − 3) / 3.
    """
    n = len(pred)
    return sum(
        1
        for i in range(n)
        for j in range(i + 1, n)
        if (pred[i] - pred[j]) * (true[i] - true[j]) > 0
    )


def pairwise_reward(text, true_rank):
    """순서 근접도 shaping ∈ [0, 1]. EM 희소 보상을 보완한다 (전멸 그룹에도 그래디언트).

    text: 생성 completion. true_rank: 이 뷰의 진실 rank (list[int] 또는 Answer 문자열).

    - 파싱 불가 → 0.0 (파싱 성공 > 실패 불변식 유지)
    - identity 예측인데 true가 비-identity → 0.0: 센터링만으로는 identity가 근접 뷰
      (예: true=[1,2,4,3])에서 shaping을 수확해 collapse 페널티를 상쇄하므로, 아예 제외해
      collapse 합성 보상을 항상 −0.1로 고정한다.
    - 그 외 → max(0, (concordant − 3) / 3). 우연 수준(3쌍)을 0점으로 센터링하고 음수는
      클램프한다 (완전역순 오답이 파싱실패보다 낮아지는 역전 차단).
    """
    true_rank = _as_rank(true_rank)
    pred = parse_permutation(text)
    if pred is None:
        return 0.0
    if pred == IDENTITY and true_rank != IDENTITY:
        return 0.0
    centered = (pairwise_concordant(pred, true_rank) - CHANCE_CONCORDANT) / CHANCE_CONCORDANT
    return max(0.0, centered)


def batch_pairwise_rewards(texts, true_ranks):
    """길이가 같은 texts/true_ranks에 pairwise_reward를 매핑한다."""
    assert len(texts) == len(true_ranks), f"length mismatch {len(texts)} != {len(true_ranks)}"
    return [pairwise_reward(t, r) for t, r in zip(texts, true_ranks)]


def _completion_text(completion):
    """TRL completion을 평문 텍스트로. 평문·대화형(list[{role,content}]) 모두 지원하며,
    content가 문자열이든 [{type,text}] 파트 리스트든 텍스트만 이어붙인다."""
    if isinstance(completion, str):
        return completion
    content = completion
    if isinstance(completion, list) and completion and isinstance(completion[-1], dict):
        content = completion[-1].get("content", "")   # 마지막 assistant 턴
    elif isinstance(completion, dict):
        content = completion.get("content", "")
    if isinstance(content, list):
        return " ".join(
            part.get("text", "") for part in content
            if isinstance(part, dict) and part.get("type", "text") == "text"
        )
    return str(content or "")


def make_grpo_reward(true_rank_field="true_rank", **kw):
    """TRL GRPOTrainer용 보상 함수 어댑터를 만든다 (이 모듈은 trl을 import하지 않는다).

    TRL은 reward_func(prompts=..., completions=..., **columns)를 키워드로 호출하고
    데이터셋 컬럼을 리스트로 넘긴다. true_rank_field 컬럼(뷰별 진실 rank)을 정답으로 쓴다.
    kw는 sequence_reward의 보상 형태 파라미터(em_reward 등)로 전달된다.
    """
    def reward_func(prompts=None, completions=None, **kwargs):
        true_ranks = kwargs[true_rank_field]
        texts = [_completion_text(c) for c in (completions or [])]
        return batch_rewards(texts, true_ranks, **kw)

    reward_func.__name__ = "grpo_em_reward"
    return reward_func


def make_grpo_pairwise_reward(true_rank_field="true_rank"):
    """TRL GRPOTrainer용 pairwise shaping 보상 어댑터 (make_grpo_reward와 동일 규약).

    make_grpo_reward와 함께 reward_funcs 리스트에 등록하고, GRPOConfig(reward_weights=[1.0, λ])로
    λ를 준다. TRL이 rewards/grpo_em_reward·rewards/grpo_pairwise_reward를 개별 로깅한다.
    """
    def reward_func(prompts=None, completions=None, **kwargs):
        true_ranks = kwargs[true_rank_field]
        texts = [_completion_text(c) for c in (completions or [])]
        return batch_pairwise_rewards(texts, true_ranks)

    reward_func.__name__ = "grpo_pairwise_reward"
    return reward_func
