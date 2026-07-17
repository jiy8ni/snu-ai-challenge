# RunPod 실행 가이드

목표: 큰 파일을 전부 `/workspace/snuai` 아래에 두어 root/container disk의
`disk quota exceeded`를 피한다.

> **⚠️ §7 (_0716 8B plain 라운드)이 현재 최신이다.** 아래 §0~§6은 7B mid 기준의 원본
> 가이드다 — 8B plain 라운드(`colab/qwen_vl_colab_0716.ipynb`를 RunPod CLI로 옮긴 것)를
> 돌리려면 **§7을 따르고**, §0~§6은 배경/공통 설정 참고로만 본다.
> RunPod는 학습·병합·추론이 **각각 별도 프로세스**라 Colab의 "병합 후 GPU 잔류 OOM →
> 런타임 재시작"(노트북 B5 셀)이 **없다**. 그 셀은 CLI에서 무시한다.

## 0. 권장 디렉터리

```bash
/workspace/code              # 이 repo
/workspace/snuai/data         # 대회 데이터(train.csv, test.csv, train/, test/)
/workspace/snuai/cache        # HF/torch/triton/pip cache
/workspace/snuai/outputs      # split, sft jsonl, LoRA/checkpoints, raw jsonl
/workspace/snuai/models       # merged model
```

## 1. 환경 변수와 설치

```bash
cd /workspace/code
source cloud/runpod_env.sh

python -m pip install -U pip
python -m pip install --no-cache-dir \
  unsloth imagehash scikit-learn open_clip_torch \
  pandas pillow tqdm pyyaml qwen-vl-utils
python -m pip cache purge || true
```

`--no-cache-dir`를 쓰면 pip wheel cache가 남지 않는다. `runpod_env.sh`는
Hugging Face, torch, triton, tmp 경로를 `/workspace/snuai/cache`로 돌린다.
RunPod PyTorch 템플릿을 쓰면 torch/torchvision은 보통 이미 설치돼 있으므로 다시
설치하지 않는 편이 quota와 시간을 아낀다.

## 2. 데이터 경로 준비

데이터가 `/workspace/snuai/data`에 있으면:

```bash
python -m cloud.runpod_prepare --data-root /workspace/snuai/data
export SNUAI_PATHS_CONFIG=/workspace/snuai/paths_runpod.yaml
```

다른 위치라면 `--data-root`만 바꾼다.

## 3. split / SFT jsonl 생성

현재 코드 zip에는 `outputs/sft_train.jsonl`이 포함되지 않을 수 있으므로 RunPod에서 생성한다.

```bash
python -m src.data.split
python -m src.train.cot_target
```

생성 파일:

```bash
/workspace/snuai/outputs/split.csv
/workspace/snuai/outputs/sft_train.jsonl
/workspace/snuai/outputs/sft_val.jsonl
```

## 3.5 Hard-case-aware LLM caption augmentation

Rules update: LLM text augmentation is allowed when the external-generation
budget stays under 30,000 KRW. Keep the default guard at 25,000 KRW unless you
explicitly want to spend the full allowance.

To weight `sft_train.jsonl`, build predictions for the same train fold IDs.
Do not use test information. A validation-fold hard-case file is useful for
diagnosis, but it will not match `sft_train.jsonl` IDs.

```bash
python -m src.infer.predict \
  --model /workspace/snuai/models/qwen25vl7b_merged \
  --split train --fold train --tta 4 --batch 8 \
  --style mid --out /workspace/snuai/outputs/raw_train.jsonl

python -m src.infer.aggregate \
  --raw /workspace/snuai/outputs/raw_train.jsonl \
  --out /workspace/snuai/outputs/pred_train.csv
```

Turn missed/low-tau train samples into augmentation weights:

```bash
python -m src.preprocess.hard_cases \
  --pred /workspace/snuai/outputs/pred_train.csv \
  --fold train \
  --out /workspace/snuai/outputs/hard_train_cases.csv
```

Generate LLM paraphrases once, offline:

```bash
export OPENAI_API_KEY=...
python -m src.preprocess.llm_caption_augment \
  --input /workspace/snuai/outputs/sft_train.jsonl \
  --hard-cases /workspace/snuai/outputs/hard_train_cases.csv \
  --out /workspace/snuai/outputs/sft_train_llm_aug.jsonl \
  --max-cost-krw 25000
```

If you do not have a previous checkpoint yet, omit `--hard-cases`; every record
gets the base number of variants and no hard-case extra repeats.

For a cheap smoke test before spending API budget:

```bash
python -m src.preprocess.llm_caption_augment \
  --input /workspace/snuai/outputs/sft_train.jsonl \
  --out /workspace/snuai/outputs/sft_train_llm_aug_smoke.jsonl \
  --max-records 5 --dry-run
```

Train on the augmented JSONL:

```bash
python -m cloud.runpod_train \
  --sft-jsonl /workspace/snuai/outputs/sft_train_llm_aug.jsonl \
  --per-device-batch 1 --grad-accum 16
```

## 4. 학습

먼저 스모크:

```bash
python -m cloud.runpod_train --smoke
```

본 학습:

```bash
python -m cloud.runpod_train --per-device-batch 1 --grad-accum 16
```

A100 40GB 이상이면 아래처럼 시도 가능:

```bash
python -m cloud.runpod_train --per-device-batch 2 --grad-accum 8
```

중단 후 재개:

```bash
python -m cloud.runpod_train --resume --per-device-batch 1 --grad-accum 16
```

RunPod config는 `configs/sft_qwen_runpod.yaml`이다. 기본값은
`save_total_limit: 1`, `save_steps: 500`이라 오래된 `checkpoint-*`가 자동 삭제된다.

## 5. 병합과 검증

```bash
python -m cloud.runpod_merge

python -m src.infer.predict \
  --model /workspace/snuai/models/qwen25vl7b_merged \
  --split train --fold val --limit 32 --tta 1 --batch 4 \
  --style mid --out /workspace/snuai/outputs/raw_val_smoke.jsonl
```

전체 val:

```bash
python -m src.infer.predict \
  --model /workspace/snuai/models/qwen25vl7b_merged \
  --split train --fold val --tta 8 --batch 8 \
  --style mid --out /workspace/snuai/outputs/raw_val.jsonl

python -m src.infer.aggregate \
  --raw /workspace/snuai/outputs/raw_val.jsonl \
  --out /workspace/snuai/outputs/pred_val.csv

python -m src.eval.em --pred /workspace/snuai/outputs/pred_val.csv --fold val
```

test submission:

```bash
python -m src.infer.predict \
  --model /workspace/snuai/models/qwen25vl7b_merged \
  --split test --tta 8 --batch 8 \
  --style mid --out /workspace/snuai/outputs/raw_test.jsonl

python -m src.infer.aggregate \
  --raw /workspace/snuai/outputs/raw_test.jsonl \
  --submission /workspace/snuai/outputs/submission.csv
```

## 6. 용량 점검

```bash
df -h / /workspace
du -h -d 1 /workspace/snuai | sort -h
du -h -d 1 /workspace/snuai/cache | sort -h
```

스모크 학습 산출물이 필요 없으면 본 학습 전에 지워도 된다:

```bash
rm -rf /workspace/snuai/outputs/qwen25vl7b_smoke
```

본 학습 중 수동 삭제가 필요하면 `checkpoint-*`만 삭제하고 `lora/`는 지우지 않는다.
평소에는 `save_total_limit: 1`이 자동으로 오래된 checkpoint를 정리한다.

---

# 7. _0716 8B plain 라운드 — 노트북 → RunPod CLI 대응

`colab/qwen_vl_colab_0716.ipynb`의 셀 A1~C2를 RunPod 터미널 명령으로 옮긴 것.
config는 **`configs/sft_qwen8b_runpod.yaml`**(8B·plain·identity_prior 0.155·하드반복 off),
학습 데이터는 **팀원 LLM 증강본 `outputs/sft_train_llm_aug_hard_nogate.jsonl`**(zip에 포함).

## 7.0 GPU·디스크 사양

- **GPU**: 권장 **A100 40GB**(`--per-device-batch 2 --grad-accum 8`). 최소 24GB(L4/4090,
  `--per-device-batch 1 --grad-accum 16`, 1.5~2배 느림). **T4 불가**(8B).
- **볼륨 디스크 100GB+**: 이미지 데이터 + 8B 병합본(fp16 ~16GB) + 4bit 캐시(~6GB) + checkpoint.
- 예상: 학습 ~2–3h + 추론(C0+C1+C2) ~1.5h ≈ **4–5h**.

## 7.1 A1·A4) 코드 압축 해제 + 환경 + 설치

```bash
mkdir -p /workspace/code && cd /workspace/code
unzip -qo /workspace/snuai_code_0716.zip -d /workspace/code   # zip을 볼륨에 먼저 업로드
source cloud/runpod_env.sh                                    # 캐시/tmp를 /workspace/snuai로

python -m pip install -U pip
python -m pip install --no-cache-dir unsloth imagehash pandas pillow tqdm pyyaml qwen-vl-utils
python -m pip cache purge || true
# 신선도 확인 (plain 스타일이 실제로 풀렸는지)
python -c "import sys; sys.path.insert(0,'/workspace/code'); from src.train.targets import STYLES; assert 'plain' in STYLES; print('STYLES', STYLES)"
```

## 7.2 A2·A3) 데이터 배치 + 경로

전달본 `configs/paths_runpod.yaml`은 데이터를 **`/workspace/snuaichallenge_data`**에서
찾는다. 대회 데이터를 그 경로에 풀고(또는 Kaggle CLI로 받고), 경로 config를 export한다.

```bash
# 데이터: /workspace/snuaichallenge_data/{train.csv,test.csv,sample_submission.csv,train/,test/}
export SNUAI_PATHS_CONFIG=/workspace/code/configs/paths_runpod.yaml
# split.csv를 outputs_dir로 (val fold 추론이 참조). zip의 outputs/split.csv 사용.
mkdir -p /workspace/snuai/outputs
cp /workspace/code/outputs/split.csv /workspace/snuai/outputs/split.csv
python -c "import sys; sys.path.insert(0,'/workspace/code'); from src.data.loader import load_split; print('train', len(load_split('train')), 'test', len(load_split('test')))"
```

> 데이터를 다른 경로에 뒀다면 `python -m cloud.runpod_prepare --data-root <경로>`로
> `paths_runpod.yaml`을 재생성하고 그 경로를 export한다.

## 7.3 B0) 전달 데이터 스키마 검증 (학습 전 필수)

```bash
cd /workspace/code
python - << 'PY'
import json
recs=[json.loads(l) for l in open('outputs/sft_train_llm_aug_hard_nogate.jsonl',encoding='utf-8')]
n=len(recs); n_no=sum(bool(r['no_ordering']) for r in recs)
bad=[r['Id'] for r in recs if r['no_ordering'] and r['rank']!=[1,2,3,4]]
assert not bad, f'no_ordering인데 rank!=identity {len(bad)}건 — 진실 증강 전제 위반'
assert abs(n_no/n-0.155)<0.01, f'no_ordering 비율 {n_no/n:.4f}'
print(f'OK records={n} no_ordering={n_no} ({n_no/n:.4f})')
PY
```

## 7.4 B1·B2) 스모크 → 본 학습

```bash
SFT=/workspace/code/outputs/sft_train_llm_aug_hard_nogate.jsonl
CFG=configs/sft_qwen8b_runpod.yaml

# B1) 스모크 (32샘플)
python -m cloud.runpod_train --config $CFG --sft-jsonl $SFT --smoke

# B2) 본 학습 (A100 40GB). 끊기면 --resume 추가. output_dir은 config가 지정(/workspace/snuai/...)
python -m cloud.runpod_train --config $CFG --sft-jsonl $SFT --per-device-batch 2 --grad-accum 8
```

## 7.5 B3·B4) 병합 → 병합 검증 게이트

```bash
# B3) 병합 (unsloth#1352 회피: merge_and_unload)
python -m cloud.runpod_merge \
  --lora /workspace/snuai/outputs/qwen3vl8b_0716/lora \
  --out  /workspace/snuai/models/qwen3vl8b_merged_0716

# B4) 병합 검증 (--style plain, 4/8 이상이어야 통과). RunPod는 프로세스 분리라 재시작 불요.
MODEL=/workspace/snuai/models/qwen3vl8b_merged_0716
python -m src.infer.predict --model $MODEL --split train --fold train \
  --limit 8 --tta 1 --batch 8 --style plain --out /workspace/snuai/outputs/merge_check.jsonl
python - << 'PY'
import json, sys; sys.path.insert(0,'/workspace/code')
from src.utils.permutation import parse_permutation, unshuffle_rank_label, parse_answer_column
from src.data.loader import load_split
truth=load_split('train').set_index('Id'); hits=n=0
for r in map(json.loads, open('/workspace/snuai/outputs/merge_check.jsonl')):
    rank=parse_permutation(r['text']); n+=1
    hits += rank is not None and unshuffle_rank_label(rank,r['perm'])==parse_answer_column(truth.loc[r['Id'],'Answer'])
print(f'merge check: {hits}/{n}'); assert n>=8 and hits>=4, '병합 실패 의심 — B3 재확인'
print('병합 검증 통과')
PY
```

## 7.6 C0) 기준선 EM — **다중 델타의 유일한 판정자** (건너뛰지 말 것)

이 라운드는 8B는 그대로 두고 레시피(plain)+증강본을 동시에 바꾼다. C1이 이겼는지 판정하려면
**기존 8B(mid, LB 0.88)의 val EM**이 필요하다. 두 경로 중 하나:

```bash
# (A) 기존 raw_val을 RunPod로 옮겼다면 — GPU 불필요, 즉시:
python -m src.infer.aggregate --raw /workspace/snuai/outputs/raw_val_8b_mid_baseline.jsonl \
  --out /workspace/snuai/outputs/pred_val_base.csv
python -m src.eval.em --pred /workspace/snuai/outputs/pred_val_base.csv --fold val

# (B) 기존 8B 병합모델을 옮겼다면 — val tta8 1회 (~40분):
BASE=/workspace/snuai/models/qwen3vl8b_merged_0712     # ← 실제 경로로
python -m src.infer.predict --model $BASE --split train --fold val \
  --style mid --tta 8 --batch 16 --out /workspace/snuai/outputs/raw_val_base.jsonl
python -m src.infer.aggregate --raw /workspace/snuai/outputs/raw_val_base.jsonl \
  --out /workspace/snuai/outputs/pred_val_base.csv
python -m src.eval.em --pred /workspace/snuai/outputs/pred_val_base.csv --fold val
```

> 기존 8B 산출물(모델 또는 raw_val)을 Drive/HF에서 RunPod 볼륨으로 먼저 복사해야 한다.
> 둘 다 없으면 C0을 못 재므로 제출 판정이 불가능하다.

## 7.7 C1·C2) val 게이트 → test 제출

```bash
MODEL=/workspace/snuai/models/qwen3vl8b_merged_0716

# C1) val 게이트. 통과 조건: ① EM ≥ C0 기준선 ② identity_rate ≈ 0.155 ③ em_no_ordering·em_orderable 개선방향
python -m src.infer.predict --model $MODEL --split train --fold val \
  --style plain --tta 8 --batch 16 --out /workspace/snuai/outputs/raw_val_0716.jsonl
python -m src.infer.aggregate --raw /workspace/snuai/outputs/raw_val_0716.jsonl \
  --out /workspace/snuai/outputs/pred_val.csv
python -m src.eval.em --pred /workspace/snuai/outputs/pred_val.csv --fold val

# C2) test 제출 (C1이 기준선을 넘었을 때만)
python -m src.infer.predict --model $MODEL --split test \
  --style plain --tta 8 --batch 16 --out /workspace/snuai/outputs/raw_test_0716.jsonl
python -m src.infer.aggregate --raw /workspace/snuai/outputs/raw_test_0716.jsonl \
  --submission /workspace/snuai/outputs/submission_0716.csv
```

## 7.8 H1') Phase 1.5 재채굴 (C1 통과 후에만)

plain 모델 자신의 오답으로 하드케이스를 다시 만든다 (부스팅의 정의: 직전 모델의 오답에 가중).

```bash
MODEL=/workspace/snuai/models/qwen3vl8b_merged_0716
python -m src.infer.predict --model $MODEL --split train --fold train \
  --style plain --tta 4 --batch 16 --out /workspace/snuai/outputs/raw_trainfold_0716.jsonl
python -m src.infer.aggregate --raw /workspace/snuai/outputs/raw_trainfold_0716.jsonl \
  --out /workspace/snuai/outputs/pred_trainfold_0716.csv
python -m src.preprocess.hard_cases --pred /workspace/snuai/outputs/pred_trainfold_0716.csv \
  --fold train --out /workspace/snuai/outputs/hard_train_cases_0716.csv
# 이후 llm_caption_augment로 재증강 → sft_qwen8b_v2 config로 Phase 1.5 재학습
```
