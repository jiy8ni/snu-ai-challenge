# Track B 1차 확정 결과 — 2026-07-05 (LB 0.71)

## 요약

| 항목 | 값 |
|---|---|
| **Public LB (2026-07-05 제출)** | **0.71** (baseline 0.1, 당시 1등 0.87) |
| val EM (953개) | 0.4439 |
| val Kendall τ | 0.523 |
| val em_orderable / em_no_ordering | 0.458 / 0.046 |
| 모델 | unsloth/Qwen2.5-VL-3B-Instruct, QLoRA r16 (vision+language 전체) |
| 학습 | Colab A100, 2 epoch, 유효 배치 16, lr 1e-4 cosine, bf16, mid 스타일 + 순열 증강 |
| 추론 | 병합 bf16 + permutation-TTA 4 + 투표 집계 (`src/infer/predict.py` → `aggregate.py`) |

재현 파이프라인: `colab/qwen_vl_colab.ipynb` (_0705 최종본, A1→…→C2 순서 셀).

## Drive 동결 자산 (`MyDrive/snuai/`)

- `qwen25vl3b_merged_0705/` — 제출에 쓴 병합 모델 (B4 검증 통과: train 8샘플 Answer 일치 4/8)
- `outputs/qwen25vl3b_0705/lora/` — LoRA 어댑터 원본 (재병합·본선 제출용. 규정상 LoRA 명시 허용)
- `raw_val_0705.jsonl` / `raw_test_0705.jsonl` — TTA 생성 원본 (집계 정책 변경 시 재추론 불필요)
- `submission_0705.csv` — LB 0.71 제출본
- `snuai_code_0705.zip` — 코드 스냅샷

## 이번 라운드에서 확정한 사실

1. **unsloth `save_pretrained_merged`는 vision 모델에서 LoRA를 병합하지 않는다** (unsloth#1352).
   베이스만 저장돼 val EM이 0.046(랜덤)으로 추락했었다. → `merge_and_unload()`(peft 표준)로 교체,
   병합 후 train 샘플 생성 비교(B4)를 필수 관문으로 도입.
2. **train loss는 EM의 대리 지표가 아니다.** loss 0.057은 포맷 보일러플레이트 암기로도 달성되며,
   병합이 깨진 상태에서도 눈치챌 수 없었다. 제출 전 val 리허설(C1)이 유일한 안전망.
3. **집계 정책 (val 스윕, 0.3924 → 0.4439)**: identity-ban은 -4.2pp 해악, TTA 분산 게이트
   (최빈 표수 1 → [1,2,3,4])는 +1pp. → `aggregate.py` 기본값 변경
   (구 정책: `--ban-identity` / `--no-disperse-gate`).
4. **모델이 UNORDERABLE을 단 1회도 출력하지 않는다** (val 3812 생성 중 0회).
   no_ordering 대응은 전적으로 분산 게이트가 담당 중.
5. Qwen2.5-VL fp16 추론은 test에서 parse_fail 19% 등 불안정 → predict.py가 bf16 자동 선택
   (T4만 fp16 폴백).
6. 손상 이미지 1장이 50분 추론을 99%에서 죽인 사고 → predict.py에 스킵 방어 + resume 활용.
   (원인은 세션 unzip 불량으로 추정, 대회 데이터 자체는 정상)

## 남은 개선 후보 (다음 라운드)

- **val(0.44) ↔ LB(0.71) 괴리 +27pp**: 원인 미상. val fold가 test보다 어렵거나 split 편향 의심.
  split.csv 구성 재점검 필요 — val이 실제보다 비관적이면 ablation 판단이 왜곡된다.
- **UNORDERABLE 전이 실패**: 학습 타깃의 15%였는데 출력 0회. no_ordering 오버샘플링,
  게이트 전용 손실/타깃 보강 검토.
- em_orderable 0.458 → 오답의 다수가 인접 스왑(B4 관찰). CoT 스타일, epoch 추가,
  TTA 수 증가(전수 24?는 24h 제한 내 계산), 전수 로그우도 스코어링(PLAN.md) 등.
- 본선 환경(RTX 3090, 오프라인) 리허설: 병합 모델 로드·24h 내 추론 시간 실측.
