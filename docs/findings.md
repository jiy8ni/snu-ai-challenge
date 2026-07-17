# EDA 핵심 발견 사항 (2026-07-03)

## 1. `No_ordering` 컬럼 — 이 대회의 숨은 핵심 (train.csv에만 존재)

> **⚠️ [2026-07-16 전제 정정] 이 절의 "정렬 불가(unorderable)" 표현은 오독이다.**
> 공식 정의는 **"No_ordering=True면 이미지는 셔플링되지 않았으며 정답은 [1,2,3,4]로 고정"** —
> 즉 *생성 과정*(출제자가 안 섞음)이지 *콘텐츠 속성*(정렬 불가)이 아니다. no_ordering 샘플은
> 원래 시간순 그대로인 **정상 영상**이다(프레임 실물·`em_no_ordering` 0.596 > `em_orderable` 0.5025로
> 확정). 아래 "이벤트 수 < 프레임 수라 특정 불가"·"CoT의 UNORDERABLE 분기"·"identity 금지 후처리"는
> 이 오독의 산물이다. 상세와 수정(plain 레시피)은 `reports/preprocessing.md` §B3-3 참조.
> 이 절 원문은 당시 판단 이력으로 남겨둔다.

- train 9,535개 중 **1,478개(15.5%)가 `No_ordering=True`이고, 이들의 Answer는 전부 `[1,2,3,4]`** (1478/1478).
- orderable(`False`) 샘플 8,057개에서는 **identity 순열 `[1,2,3,4]`가 단 한 번도 등장하지 않음** (0/8057).
- 나머지 23개 순열은 각 ~360개로 거의 균등 분포 (셔플 편향 없음).

### 시사점
1. test에도 ~15%의 정렬 불가 샘플이 있을 것 → **"orderable 여부" 이진 분류만 정확히 하면 ~15%p 확보.**
2. orderable로 판단한 샘플에는 **identity를 절대 출력하지 않는 후처리** 가능 (identity가 나오면 재생성/2순위 순열 대체).
   **[2026-07-04 조건부 반증]** 이 규칙은 **상류의 unorderable 판별이 강할 때만** 유효. Track A(게이트 precision ~0.21)에서는 identity 금지가 val EM을 0.2088 → 0.1878로 **-2.1pp 악화** — 게이트를 새어나간 진짜 no_ordering 샘플의 identity 정답을 강제로 버리기 때문. 판별력 있는 게이트(Track B SFT)에서는 재검증 필요. 근거: `src.track_a.train --mode decode-ablation`.
3. 학습 시 No_ordering 샘플을 단순 제외하면 안 됨 — 모델이 "정렬 불가 → [1,2,3,4]" 패턴 자체를 배워야 한다.

### No_ordering의 실체
- 검은 프레임이 아니라 **"캡션 이벤트 수 < 프레임 수"** 케이스가 다수. 예: `diZi5g` — 캡션은 2개 이벤트(얼음낚시 장면 전환, 스키어 레일 점프)만 서술하는데 프레임은 이벤트당 2장 → 이벤트 내 순서를 캡션으로 특정 불가.
- 연결어(then/before/after/finally 등) 개수만으로는 예측 불가: True/False 양쪽 모두 평균 ~0.90개, 개수별 True 비율도 12~16%로 평탄.
- **[2026-07-03 확정] 저수준 피처(phash·픽셀 통계)도, CLIP 의미 요약 피처도 판별력 전무** (양쪽 모두 5-fold AUC ≈ 0.49, 클래스별 피처 평균 동일). 같은 이벤트의 프레임들도 카메라 이동 때문에 픽셀 수준에선 다르고, CLIP 요약 수준에서도 구분되지 않는다.
- → **전략 확정: 규칙/보조모델 사전 판별은 폐기. VLM SFT가 train의 No_ordering 라벨 1,478개로 end-to-end 학습** (CoT의 UNORDERABLE 분기). "orderable이면 identity 금지" 후처리는 유지. 상세는 `reports/preprocessing.md` 결론 참조.

## 2. 데이터 규모·형태

| 항목 | 값 |
|---|---|
| train / test 샘플 | 9,535 / 819 |
| train 이미지 수 | 38,140 (= 9,535 × 4) |
| 해상도 | 320×240 ~ 640×360 혼재 (360p 계열 다수, 레터박스 검은 띠 포함) |
| 캡션 | 영어, 평균 24단어 (5~69), "then/followed by/finally/as" 등 시간 연결어 사용 |
| Input_1~4 | `{Id}_{랜덤3글자}.jpg`, 알파벳 정렬 순 (셔플은 파일명 접미사로 구현됨) |

## 3. baseline 성능 해석

- baseline(Qwen2-VL-2B zero-shot) test 점수 0.1 ≈ 항상 `[1,2,3,4]`를 내는 상수 예측의 기대 점수(~15%)와 비슷한 수준. 파싱 실패 fallback이 `[1,2,3,4]`인 점을 감안하면 **모델이 순서를 사실상 못 맞히고 있음.**
- **[2026-07-04 공식 확정] 평가 지표 = Exact Match Accuracy** (Public 70% / Private 30%). 상세는 `docs/rules.md` §2.

## 3.5 Track A (동결 SigLIP2 + Set-to-Rank 헤드) — 시간 순서 신호는 존재한다 (2026-07-04)

- holdout(캡션 그룹 분리 90/10): **val EM 0.209** (identity 허용 + 게이트 τ 캘리브), 순수 순열 정확도 **perm_acc 0.208 = 랜덤(0.043)의 ~5배**, Kendall τ 0.23.
- **No_ordering "판별"은 요약 피처로 불가능(AUC 0.49)했지만, "순서" 자체는 동결 임베딩에서 부분적으로 학습 가능** — 두 과제의 난이도가 다르다는 첫 실증.
- 게이트(UNORDERABLE 분류 헤드)는 여전히 무력 (precision ~0.21 ≈ 기저율 15.5%) — 기존 음성 결과와 정합.
- 상세 ablation은 `reports/preprocessing.md`.

## 4. 전략에 미치는 영향

- VLM SFT 타깃에 **UNORDERABLE 분기 통합** (CoT 마지막에 "정렬 불가 → [1,2,3,4]"): 앙상블 금지 규칙을 지키면서 No_ordering 판별을 단일 모델에 내장.
- 전처리 우선순위: No_ordering 판별 > 캡션 이벤트 분해 > 근접 중복/레터박스 > 정합성 스코어. 상세 설계는 [reports/preprocessing.md](../reports/preprocessing.md).
