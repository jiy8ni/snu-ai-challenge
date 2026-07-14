# Track B _0707 라운드 — 7B 진단·융합 검증·다음 레버 (2026-07-06)

전 라운드: `track_b_0705.md` (3B, LB 0.71) → _0706 (7B, **LB 0.81**, 당시 1등 0.92).
이 문서는 _0706 raw(생성 tta4 + LL tta1)를 로컬 CPU로 재분석한 결과와 _0707 실행 계획.

## 요약 (확정 사실)

| 항목 | 3B (_0705) | 7B (_0706) |
|---|---|---|
| Public LB | 0.71 | **0.81** |
| val EM (투표 정책) | 0.4439 | **0.5173** |
| val em_orderable / em_no_ordering | 0.458 / 0.046 | 0.5025 / **0.5960** |
| val 4/4 합의 비율 · EM | 23.5% · 0.79 | 37.6% · 0.8045 |
| test 4/4 합의 비율 | 38.0% | **67.5%** |
| projected_lb (합의도 전이) | 0.533 (실제 0.71) | 0.6668 (실제 0.81) |

1. **no_ordering은 7B가 사실상 해소했다** — UNORDERABLE 텍스트는 여전히 0회지만,
   무순서 샘플에서 identity "투표"가 자발적으로 다수를 이뤄 em_no_ordering 0.046 → 0.596.
   _0705에서 "최대 단일 개선 항목(+14pp)"이던 것이 이제 잔여 +6.4pp(151샘플 × 0.4) 규모.
2. **LL 스코어링(tta1)은 투표(tta4)에 전면 열세** — 단독 best 0.4753(margin -0.075),
   융합 전 구성이 투표 0.5173 미달. E2(LL test 제출)는 tta1 기준 **제출 가치 없음**.
3. **projected_lb가 실제 LB를 -14pp 과소추정하는 패턴 재현** (0.533/0.71 → 0.667/0.81).
   7B 기준 val+30pp ≈ LB. 판단 규칙(상대 비교만) 유지.

## Phase A — 융합 정책 val 스윕 (로컬, 재추론 없음)

`src/infer/fuse.py` sweep + 스크래치 변형 검증. 전부 953샘플, 기존 raw만 사용:

| 정책 | val EM | 비고 |
|---|---|---|
| **vote (현행 C2 정책)** | **0.5173** | disperse_gate on, ban off |
| h1 (합의<3 → LL 전수) | 0.5100 | LL이 저합의 구간에서도 투표보다 약함 |
| h2 (합의<3 → 투표후보 LL 재순위) | 0.5089 | |
| v1 (borda_tie 89건만 LL) | 0.5005 | tie-break도 LL이 Borda보다 나쁨 |
| v2 (disperse_gate 70건만 LL) | 0.5142~0.5163 | 게이트(identity 베팅)가 근소 우위 |
| v3 (가산: votes + λ·softmax(LL)) | ≤0.5089 | |
| ll 단독 (margin -0.075) | 0.4753 | 단 em_no_ordering 0.629로 판별력 자체는 실재 |

**결론: 결정 정책은 현행 투표 유지.** LL(tta1)은 어떤 결합에서도 기여 없음.
identity-ban 재검증(findings §1-2)도 종결 — no_ordering 정답의 다수가 identity 예측에서
나오므로 ban은 명백한 해악. 기본값(off) 유지.

주의: 스윕 ~300구성 중 최고를 고르는 방식은 val 과적합 위험 — 채택 기준은
"vote 대비 +2pp(≈1.5 표준오차) 이상"으로 두었고, 아무 구성도 미달.

## 남은 병목과 _0707 레버 (우선순위)

병목 = **em_orderable 0.5025**, 특히 저합의 구간 (val 합의≤2 = 39%, EM 0.21~0.26;
test 합의≤2 = 14.4%).

1. **G1. 생성 TTA 4→8** (GPU 소, 리스크 최소): 합의도가 잘 보정된 지표이므로 표본을
   늘리면 2-2 동률·저합의 구간이 정밀해진다. `tta_perms` 접두어 속성 확인 완료 —
   같은 `--out`에 `--tta 8`이면 기존 4뷰는 스킵, 신규 4뷰만 생성 (val ~40분, test ~35분 A100).
   val EM ≥ +1pp면 test 확장 후 제출.
2. **G2. 재학습 (Phase C)**: no_ordering 오버샘플링 ×2 + 3 epochs. 목표는 orderable
   정밀도와 UNORDERABLE 신호 강화 둘 다. (×3이 아닌 ×2인 이유: no_ordering이 이미 0.596이라
   identity 과베팅 부작용 경계.)
3. **G3. (선택) LL tta4 재평가**: E1은 identity 뷰 1개만 스코어했다 — 뷰 4개 합산이면
   투표와 대등해질 가능성. `ll_score score --tta 4` 재개로 3뷰만 추가 (val ~3h A100).
   G1·G2보다 후순위. 이기면 fuse sweep 재실행 (인프라 준비 완료).
4. **G4. 3090 24h 예산 실측** (`src/infer/budget_check.py`) — 규정 준수 확인의 유일한
   잔여 항목. A100 실측 × 보수 계수 3으로 문서화.

## 픽셀 캡 OOM 근본 원인 수정 (E2 차단 해제)

`apply_pixel_caps`의 `try: ip.min_pixels=... except AttributeError` 는 **silent no-op**
이었다 (속성 대입은 예외를 던지지 않음 — 신형 transformers는 size dict만 읽어 캡 무시).
val은 캡 초과 이미지 0장이라 통과, test 대형 이미지에서만 OOM 재발 — 증상과 정합.

수정 (2026-07-06, 테스트 `tests/test_pixel_caps.py`):
- `cap_pixels()` — PIL 수준 강제 다운스케일. processor 버전과 무관하게 예산 보증.
  `load_frames`에 통합되어 생성·LL 두 경로 모두 적용 (학습 캡과 해상도 정합).
- `apply_pixel_caps` / `train_unsloth.build_model` — 구·신형 설정을 둘 다 무조건 기록.
- `ll_score.score_view` — 프롬프트 3,000토큰 초과 시 즉시 에러 (OOM 전 fail-fast).
- `ll_score.cmd_score` — CUDA OOM 시 chunk 반감 재시도 (이후 샘플에도 유지).

train 이미지는 전부 ≤640×360(230,400px ≤ 캡)이라 학습 결과에는 영향 없음 — 재학습 불필요.

## 실행 순서 (Colab)

1. C1b: val 생성 `--tta 8` 확장 → aggregate → EM 비교 (기준 0.5173)
2. 이기면 C2b: test `--tta 8` 확장 → submission (제출 #1)
3. B1→B4: 오버샘플 ×2 + 3ep 재학습 → C1 val 비교 → 이기면 제출 (제출 #2)
4. (여유 시) E1b: LL `--tta 4` 확장 → 로컬 fuse sweep 재실행
