# GRPO 라운드 — 로직 차근차근 설명 (팀원용)

Phase 2에서 SFT 다음 단계로 **GRPO(Group Relative Policy Optimization)** 를 붙였습니다.
이 문서는 코드를 처음 보는 팀원도 따라올 수 있게, "왜 하는지 → 어떻게 돌아가는지 → 뭘 조심해야 하는지"
순서로 설명합니다.

관련 파일:

| 파일 | 역할 |
|---|---|
| `cloud/grpo_custom.py` | **메인 학습 루프** (TRL/vLLM 비의존 커스텀 GRPO) |
| `src/train/rewards.py` | 보상 함수 (EM 보상 + pairwise shaping, 순수 함수라 CPU에서 테스트 가능) |
| `cloud/grpo_smoke.py` | 데이터셋 빌드 + 스모크 테스트 |
| `cloud/grpo_train.py` | (참고) TRL 기반 시도 — 비전 모델에서 폐기됨, 아래 §5 참조 |
| `configs/grpo_qwen8b_runpod.yaml` | RunPod A100용 설정 |
| `tests/test_rewards.py`, `tests/test_pairwise_stats.py` | 보상 로직 단위 테스트 |
| `docs/runpod.md` §8 | 실행 러너북 |

---

## 1. 왜 GRPO인가 — SFT와의 차이

- **SFT**: "정답 텍스트를 그대로 따라 써라." 정답 문장의 우도(likelihood)를 높인다.
  대회 지표인 **EM(완전일치)은 간접적으로만** 올라간다.
- **GRPO**: "네가 직접 답을 여러 개 내봐. 잘한 답은 더 자주 나오게, 못한 답은 덜 나오게 해줄게."
  **EM 자체를 보상으로 직접 최적화**한다.

GRPO의 핵심은 이름 그대로 **Group-Relative**. PPO처럼 별도의 가치(critic) 모델을 두지 않고,
**같은 프롬프트에서 K개(기본 8개)를 뽑아 그 그룹 안에서 서로 비교**해 "그룹 평균보다 잘했나/못했나"를
학습 신호로 쓴다. critic이 없어서 메모리·구현이 가볍다.

한 줄 요약: **"같은 문제를 8번 풀게 하고, 그룹 평균보다 잘 푼 답의 확률은 올리고
못 푼 답(특히 identity 도망)의 확률은 내리는 루프."**

## 2. 프롬프트 하나당 벌어지는 일 (5단계)

`cloud/grpo_custom.py`의 메인 루프 기준. 프롬프트 = 4프레임 + 순서 질문.

### 1단계 — K개 샘플링 (`_generate_k`)

같은 프롬프트를 넣고 `do_sample=True, temperature=1.0`으로 K=8개의 서로 다른 completion을 생성한다.
생성 경로는 **predict.py의 검증된 경로**(chat template + 4프레임)를 샘플링 모드로 재사용한 것.

```
completion 1: "Order: ...\nAnswer: [2, 1, 3, 4]"   ← 정답!
completion 2: "Order: ...\nAnswer: [2, 1, 4, 3]"   ← 거의 맞음
completion 3: "Order: ...\nAnswer: [1, 2, 3, 4]"   ← identity로 도망감
...
completion 8: "프레임에는 고양이가..."              ← 파싱 불가
```

### 2단계 — 각 completion에 보상 점수 매기기

```
보상 = 1.0 × sequence_reward + 0.25 × pairwise_reward
```

(가중치는 `configs/grpo_qwen8b_runpod.yaml`의 `grpo.reward_weights`)

**`sequence_reward`** — EM 중심 (`src/train/rewards.py`):

| 상황 | 보상 |
|---|---|
| 완전 정답 | **1.1** (파싱 보너스 0.1 + EM 1.0) |
| 오답이지만 파싱 됨 | 0.1 |
| 파싱 불가 | 0.0 |
| 섞인 뷰인데 identity `[1,2,3,4]` 출력 | **−0.1** (붕괴 페널티) |

마지막 줄이 중요하다. 구 mid 라운드를 망친 "모르겠으면 무조건 [1,2,3,4]" 붕괴를
파싱실패보다도 낮은 점수로 눌러버린다. 단, **진짜 identity 뷰에서 identity를 답하면 1.1**
— 정당한 답(사전확률 0.155)은 벌하지 않는다.

**`pairwise_reward`** — 근접도 shaping ∈ [0, 1]:

EM만 쓰면 8개가 **전부 오답**일 때 보상이 전부 0.1 동점 → 어드밴티지가 전부 0 →
그 프롬프트에선 아무것도 못 배운다(희소 보상 문제, 3단계 참조). 그래서 4프레임의
6개 쌍(C(4,2)) 중 **선후관계를 몇 쌍 맞혔는지** 세서 "거의 맞은 오답"(위 completion 2)에
부분 점수를 준다. 우연 수준인 3쌍을 0점으로 센터링: `max(0, (일치쌍 − 3) / 3)`.

λ=0.25인 이유: 실제 오답 분포에서 일치쌍 5(거의 정답)가 38%를 차지 —
shaping이 EM 보상(1.0)을 압도하지 않으면서 오답 간 차등을 만드는 작동점.
(분석 도구: `src/eval/pairwise_stats.py`)

### 3단계 — 그룹-상대 어드밴티지 ★ GRPO의 심장

```
A_i = (r_i − 그룹평균) / (그룹표준편차 + ε)
```

절대 점수가 아니라 **"이 그룹 안에서 상대적으로 얼마나 잘했나"** 로 변환한다.
정답(1.1)은 양수 어드밴티지, identity 도망(−0.1)은 큰 음수 어드밴티지.
그룹 평균이 곧 베이스라인이므로 PPO의 critic 모델이 필요 없다.

주의: 8개 보상이 전부 같으면 어드밴티지 전부 0 → 학습 신호 없음.
pairwise shaping이 이 "전멸 그룹"을 구제한다.

### 4단계 — completion의 log-prob을 gradient 켜고 재계산 (`_completion_logprobs`)

생성(1단계)은 `inference_mode`라 gradient가 없다. 이번엔 **grad를 켜고**
"prompt + completion" 전체를 forward해서, completion 토큰 구간만의 평균 log-prob을 구한다.
= "현재 정책이 이 답을 낼 확률"을 미분 가능한 형태로 얻는 것.

구현 디테일: 오른쪽 패딩으로 K개 배치 forward, 프롬프트 토큰 길이(이미지 확장 포함) `Lp`로
completion 경계를 잡는다.

### 5단계 — loss와 업데이트

```
loss = −mean( A_i × logprob_i )
```

이걸 최소화하면 어드밴티지가 **양수**인 completion은 log-prob이 올라가고(→ 더 자주 생성),
**음수**인 completion은 내려간다(→ 덜 생성). REINFORCE-with-baseline이고,
baseline을 그룹 통계로 잡은 것이 GRPO다.

역전파는 **LoRA 언어층만** 업데이트. `--accum 4`(프롬프트 4개)마다 옵티마이저 스텝
+ grad clip 1.0.

## 3. KL 페널티는 v1에서 생략 (β=0)

원래 GRPO는 참조 모델과의 KL 항으로 정책 폭주를 막는다. v1에서 뺀 근거:

1. **보상이 규칙 기반 검증가능(verifiable) 보상** — 파싱된 순열을 정답과 기계 비교할 뿐이라
   reward hacking 여지가 거의 없다. 보상↑ = 대회 지표↑. (DAPO, Dr. GRPO 등도 KL 제거 방향)
2. **LoRA만 업데이트** — 정책이 갈 수 있는 거리 자체가 제한된다 (암묵적 정규화).
3. **짧은 학습 + 외부 게이트** — 끝나면 Gate C(val EM > 0.5687)로 어차피 걸러진다.

**감시할 증상 2가지** (매 스텝 로그가 카나리아):

- **엔트로피 붕괴**: 8개 샘플이 전부 똑같아지면 어드밴티지 0 → 학습 정체.
  `sample:` 프리뷰가 매 스텝 거의 같은 문장이면 신호. 처방: temperature↑ 또는 lr↓.
- **포맷 표류**: Order: 추론 문장이 무너지고 최소 파싱 답만 뱉는 퇴화.
  `sample:` 프리뷰로 감시.

필요해지면 값싼 보강책: LoRA라서 PEFT `disable_adapter()`로 어댑터 끈 forward 한 번이
곧 참조 모델 log-prob — 메모리 추가 없이 KL 항(β≈0.01~0.04)을 v2로 넣을 수 있다.

## 4. 학습 후 확인 사항

- **Gate C**: val EM > **0.5687** (클린 SFT 기준선) — 못 넘으면 제출하지 않는다.
- 로그의 `mean_reward` / `EM` 추세 (스모크 기준점: mean_reward 1.147, EM 0.81).
- identity 출력 비율이 사전확률 0.155 근처인지 — anti-identity 페널티 과잉으로
  identity를 과소 출력하면 em_no_ordering이 떨어진다.

## 5. 왜 TRL을 안 쓰고 커스텀 루프인가

TRL 기반 GRPO(`cloud/grpo_train.py`)는 HF 생성·vLLM 생성 **둘 다** 비전 모델에서
우리 멀티모달 프롬프트를 생성 경로에 제대로 전달하지 못했다(2026-07-21~22 확정:
완성물이 프롬프트와 무관한 캡션/랜덤 텍스트 → 전 보상 0). 모델·프롬프트·보상 자체는
정상임을 확인한 뒤 프레임워크 생성 배선을 폐기하고, **predict.py의 검증된 생성 경로를
샘플링으로 바꿔 재사용**하는 커스텀 루프(`cloud/grpo_custom.py`)로 전환했다.
의존성은 unsloth + torch뿐이라 버전 벽이 없다.

## 6. 실행 방법

RunPod A100 전용 (로컬 CPU에선 unsloth 미설치라 import 불가). 상세는 `docs/runpod.md` §8.

```bash
# 스모크 (옵티마이저 2스텝만)
python -m cloud.grpo_custom --limit 16 --max-steps 2

# 실학습 (기본: K=8, 스텝당 ~2분 → 500스텝 ≈ 17h)
python -m cloud.grpo_custom --limit 2000

# 빠른 설정 (K=4 + 짧은 생성 + grad-ckpt off → 스텝당 ~1분 이하)
python -m cloud.grpo_custom --limit 2000 \
  --num-generations 4 --max-new-tokens 48 --no-grad-ckpt
```

속도 레버 3종:

- `--num-generations 4`: 생성·forward 비용 절반. pairwise shaping 덕에 K=4에서도
  그룹 어드밴티지가 살아있다.
- `--max-new-tokens 48`: plain 정상 출력(~40토큰)엔 영향 없고, EOS 없이 끝까지
  달리는 퇴화 샘플의 낭비만 절감.
- `--no-grad-ckpt`: gradient checkpointing을 꺼서 backward의 activation 재계산 제거
  (30~40% 가속). A100 80GB + 8B 4bit면 감당 — **OOM이 나면 이 플래그만 빼면 원복**.

중단·재개: `--save-steps`(기본 100)마다 `<out_dir>/lora`에 어댑터가 저장된다
(`[checkpoint]` 로그, 같은 폴더 덮어쓰기 — 보관하려면 `cp -r`로 백업).
`--resume-lora <out_dir>/lora`로 그 가중치에서 이어서 학습(옵티마이저 상태는 리셋 —
이 규모에선 무시 가능). **재개 시 `--skip-prompts N`을 함께 줄 것** — seed가 같아
셔플 순서가 동일하므로 N = 마지막 step × accum(예: step 300 × 4 = 1200)을 건너뛰면
이미 학습한 구간을 재방문하지 않고 남은 프롬프트만 돈다. 2000개 완주는 필수가
아니다 — 중간 체크포인트로 언제든 Gate C(val EM > 0.5687)를 재서 판정하면 된다.

이어달리기(새 데이터 구간): `--offset N`은 jsonl 앞 N개 레코드를 아예 건너뛰고
데이터를 슬라이스한다 — 이전 run이 `--limit 2000`으로 본 구간을 피해 안 본 프롬프트로
계속 학습할 때 사용(예: `--offset 2000 --limit 2500` = 레코드 2000~4499).
`--skip-prompts`(같은 데이터 내 재개)와 혼동 주의 — 함께 쓰지 않는다.

참고: GRPO는 명시적 hard-mining 없이도 hard-case 집중이 내장돼 있다 — 그룹 보상이
전부 같은 쉬운(또는 일관 오답) 프롬프트는 어드밴티지 0 = gradient 0이고, 맞았다
틀렸다 하는 경계선 프롬프트에만 학습 신호가 몰린다. `hard_cases_path` 오버샘플의
추가 가치는 hard 프롬프트 "재방문"이므로 2바퀴째부터 의미가 생긴다.

보상 로직만은 로컬에서 테스트 가능:

```bash
python -m pytest tests/test_rewards.py tests/test_pairwise_stats.py -q
```
