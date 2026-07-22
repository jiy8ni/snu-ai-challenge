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
  unsloth trl datasets imagehash scikit-learn open_clip_torch \
  pandas pillow tqdm pyyaml qwen-vl-utils
python -m pip cache purge || true
```

`trl`·`datasets`는 Phase 2 GRPO(§8, `cloud/grpo_train.py`·`grpo_smoke.py`)에서 필요하다.
SFT만 돌린다면 없어도 되지만, 같은 파드에서 이어 GRPO를 하므로 함께 설치해 둔다.
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

## 3.5 Hard-case 재가중 (모델 예측 기반, 외부 API 무관)

> 외부 LLM 캡션 증강은 대회 규칙상 철회했다. 캡션 다양성은 학습 시점 규칙 기반
> 변형(`caption_augment.py`)만으로 확보한다. Hard-case 부스팅은 모델 자신의 오답을
> 물리 오버샘플하는 방식이며 외부 생성이 전혀 없다.

직전 체크포인트로 train fold를 재채점해, 자주 틀리는 샘플에 반복 가중치를 매긴다.
test 정보는 쓰지 않는다.

```bash
python -m src.infer.predict \
  --model /workspace/snuai/models/qwen3vl8b_merged \
  --split train --fold train --tta 4 --batch 8 \
  --style plain --out /workspace/snuai/outputs/raw_train.jsonl

python -m src.infer.aggregate \
  --raw /workspace/snuai/outputs/raw_train.jsonl \
  --out /workspace/snuai/outputs/pred_train.csv
```

오답/low-tau train 샘플을 재가중 CSV로 변환:

```bash
python -m src.preprocess.hard_cases \
  --pred /workspace/snuai/outputs/pred_train.csv \
  --fold train \
  --out /workspace/snuai/outputs/hard_train_cases_clean.csv
```

CSV를 학습 시점 오버레이로 지정하면(`data.hard_cases_path`), VLSFTDataset이
`caption_aug_repeats`만큼 레코드를 물리 복제하고 `caption_aug_prob`로 규칙 기반
캡션 변형 강도를 조정한다. 별도의 데이터 생성 단계는 없다.

```bash
python -m cloud.runpod_train \
  --config configs/sft_qwen8b_v2_runpod.yaml \
  --sft-jsonl /workspace/snuai/outputs/sft_train.jsonl \
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
학습 데이터는 **클린 SFT 데이터 `outputs/sft_train.jsonl`**(외부 LLM 증강 없음, zip에 포함).
캡션 다양성은 학습 시점 규칙 기반 변형(`caption_augment.py`)으로만 확보한다.

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
python -m pip install --no-cache-dir unsloth trl datasets imagehash pandas pillow tqdm pyyaml qwen-vl-utils
python -m pip cache purge || true   # trl·datasets는 Phase 2 GRPO(§8)에서 필요
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
recs=[json.loads(l) for l in open('outputs/sft_train.jsonl',encoding='utf-8')]
n=len(recs); n_no=sum(bool(r['no_ordering']) for r in recs)
bad=[r['Id'] for r in recs if r['no_ordering'] and r['rank']!=[1,2,3,4]]
assert not bad, f'no_ordering인데 rank!=identity {len(bad)}건 — 진실 증강 전제 위반'
assert abs(n_no/n-0.155)<0.01, f'no_ordering 비율 {n_no/n:.4f}'
print(f'OK records={n} no_ordering={n_no} ({n_no/n:.4f})')
PY
```

## 7.4 B1·B2) 스모크 → 본 학습

```bash
SFT=/workspace/code/outputs/sft_train.jsonl
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
python -m src.eval.em --pred /workspace/snuai/outputs/pred_trainfold_0716.csv --fold train
python -m src.preprocess.hard_cases --pred /workspace/snuai/outputs/pred_trainfold_0716.csv \
  --fold train --out /workspace/snuai/outputs/hard_train_cases_0716.csv
# em을 hard_cases보다 **먼저**: em.py는 예측 누락 시 assert로 죽어 완전성을 보증하지만,
# hard_cases는 누락 Id를 조용히 score 1.0 처리한다. 이어서 재증강·재학습 → §7.9.
```

재개 시 `--tta 4`를 절대 바꾸지 말 것 (tta8로 재개하면 표 수 혼합 → aggregate
`votes_per_sample_dist` 불균일 → `disperse_top`이 샘플마다 다른 의미가 된다, 07-15 사고 재현).
집계 후 sanity: `votes_per_sample_dist == {"4": 8582}`, `parse_fail_rate < 0.01`,
`identity_rate ∈ [0.10, 0.25]`. train EM 기대 0.75–0.95(3ep 암기). `identity_rate ≈ 1.0/0.0`
또는 parse_fail > 5% → style/모델 오류이므로 **raw 삭제 후 재실행**(오염 raw에 resume 금지).

## 7.9 H2') Phase 1.5 v2 재학습 → 게이트 → 제출

전제: §7.8이 `hard_train_cases_0716.csv`(8,582행)를 만들었다. **변경점은 데이터 가중치 하나**
(`hard_aug_max_repeats 1→5` + 이 모델 자신의 오답 재채굴); 그 외 레시피(plain, identity_prior
0.155, oversample 1, lr/epochs/seed)는 클린 SFT와 동일하다(단일변수 원칙).

경로는 하나뿐이다(zero-API): 클린 데이터 `sft_train.jsonl` 그대로 두고, v2 config의
`hard_cases_path`를 재채굴 CSV로 지정해 오버레이한다. 외부 생성 단계는 없다 — VLSFTDataset이
`caption_aug_repeats`만큼 레코드를 물리 복제하고 `caption_aug_prob`로 규칙 기반 캡션 변형
강도만 조정한다.

### (a) v2 config에 재채굴 CSV 오버레이 지정 [로컬]

`configs/sft_qwen8b_v2_runpod.yaml`의 `data.hard_cases_path` 주석을 해제해 §7.8 산출
CSV를 가리키게 한다.

```yaml
  hard_cases_path: /workspace/snuai/outputs/hard_train_cases_0716.csv
```

### (b) B0') 재가중 CSV 게이트 [로컬 + pod 양쪽 — 전송 무결성 겸검]

```bash
python - << 'PY'
import csv, collections
CSVP='outputs/hard_train_cases_0716.csv'
rows=list(csv.DictReader(open(CSVP,encoding='utf-8')))
assert len(rows)==8582, f'rows {len(rows)}'
assert all(('hard_score' in r and 'caption_aug_prob' in r and 'caption_aug_repeats' in r)
           for r in rows), '재가중 필드 누락'
rep=[min(max(int(float(r.get('caption_aug_repeats',1))),1),5) for r in rows]
print('OK repeats', dict(sorted(collections.Counter(rep).items())), '| expanded', sum(rep))
PY
```

### (c) 패킹 · 업로드 [로컬 → pod]

```bash
python -m pytest tests -q                                  # 회귀 (전부 CPU)
python -m cloud.pack_code && cp outputs/snuai_code.zip outputs/snuai_code_0717.zip
# JupyterLab으로 /workspace/에 업로드 후:
unzip -qo /workspace/snuai_code_0717.zip -d /workspace/code
python - << 'PY'
import yaml; c=yaml.safe_load(open('/workspace/code/configs/sft_qwen8b_v2_runpod.yaml'))
assert c['output_dir'].endswith('qwen3vl8b_0717') and c['data']['hard_aug_max_repeats']==5
assert c['data'].get('hard_cases_path'), 'hard_cases_path 오버레이 미지정'
print('v2 config OK')
PY
```

### (d) B1'/B2') 스모크 → 본 학습 [RunPod]

```bash
CFG=configs/sft_qwen8b_v2_runpod.yaml
SFT=/workspace/code/outputs/sft_train.jsonl   # 클린 데이터 + config의 hard_cases_path 오버레이
python -m cloud.runpod_train --config $CFG --sft-jsonl $SFT --smoke
python -m cloud.runpod_train --config $CFG --sft-jsonl $SFT --per-device-batch 2 --grad-accum 8
```

CHECKPOINT: B2' 개시 직후 트레이너 총 스텝 == `expanded_rows//16 × 3`(±5%). 0716값 ~1,608
그대로면 repeats 미반영(config/jsonl 오류) → **즉시 중단**. `expanded_rows`는 §7.8 hard CSV의
`caption_aug_repeats` 합(clip 1..5), 기대 10,500–14,000. OOM 시 `--per-device-batch 1 --grad-accum 16`.

| train EM | n_wrong | expanded rows | total steps(3ep) | B2' 시간 |
|---|---|---|---|---|
| 0.90 | 858 | ~10,700 | ~2,000 | ~3.1–3.7h |
| 0.85 | 1,287 | ~11,800 | ~2,210 | ~3.4–4.1h |
| 0.75 | 2,146 | ~13,900 | ~2,600 | ~4.0–4.9h |

### (e) B3'/B4' 병합·검증 → C1'/C2' 게이트·제출 [RunPod]

```bash
python -m cloud.runpod_merge --lora /workspace/snuai/outputs/qwen3vl8b_0717/lora \
  --out /workspace/snuai/models/qwen3vl8b_merged_0717
MODEL=/workspace/snuai/models/qwen3vl8b_merged_0717
# B4') 병합 검증: §7.5 스니펫을 0717 경로로 재사용 (8샘플 tta1, hits ≥ 4/8)

# C1') val 게이트 (~40분)
python -m src.infer.predict --model $MODEL --split train --fold val \
  --style plain --tta 8 --batch 16 --out /workspace/snuai/outputs/raw_val_0717.jsonl
python -m src.infer.aggregate --raw /workspace/snuai/outputs/raw_val_0717.jsonl \
  --out /workspace/snuai/outputs/pred_val_0717.csv
python -m src.eval.em --pred /workspace/snuai/outputs/pred_val_0717.csv --fold val
```

**C1' 판정** (기준선 0716: EM 0.5782 / em_no_ordering 0.5629 / em_orderable 0.5810 / identity_rate ≈ 0.155):
- **PASS**: EM ≥ 0.5832 (+0.5pp = 5문항↑) → C2' 진행.
- **경계** [0.5782, 0.5832): em_no_ordering ≥ 0.5629 **그리고** em_orderable ≥ 0.5810 **그리고**
  identity_rate ∈ [0.13, 0.19]일 때만 진행(제출 슬롯 여유 전제, 07-21 이전).
- **FAIL** < 0.5782: 0716 라인 freeze (이미 LB 0.90052). **merged_0716은 C2' 완료까지 삭제 금지.**

```bash
# C2') test → 제출 (C1' 통과 시에만, ~40분)
python -m src.infer.predict --model $MODEL --split test \
  --style plain --tta 8 --batch 16 --out /workspace/snuai/outputs/raw_test_0717.jsonl
python -m src.infer.aggregate --raw /workspace/snuai/outputs/raw_test_0717.jsonl \
  --submission /workspace/snuai/outputs/submission_0717.csv
```

LB > 0.90052면 0717을 최종 라인으로, 아니면 0716 유지(단일 체크포인트 규정 — 최종 선택은 하나).
종료 후 `qwen3vl8b_0717/lora` 백업(tar) 다운로드, raw/pred/영수증 회수, 결과를
README_no_ordering.md·reports에 기록.

---

# 8. Phase 2 — 클린 SFT 재학습 → GRPO (EM + pairwise 보상)

전제·배경: 외부 LLM 캡션 증강이 규칙 위반으로 철회됐고, 기존 최고 체크포인트(0716)는 LLM
증강 데이터로 학습된 오염 모델이다. 따라서 **클린 데이터(`sft_train.jsonl`)로 8B SFT를 다시
돌리고(G1)**, 그 위에 GRPO로 EM을 직접 최적화한다(G3~G4). GRPO 보상은 EM + pairwise shaping
(6쌍 일치 수를 우연 수준으로 센터링, λ=0.25)이며, 전멸 그룹에서도 학습 신호를 살린다.
로직·근거는 `src/train/rewards.py`·`src/eval/pairwise_stats.py`, 보상은 프레임워크 비의존.

## 8.0 게이트 요약

- **Gate A** (클린 SFT): merged 클린 모델 val EM이 0716 기록(EM 0.5782) 대비 **−2pt 이내**면
  GRPO 진행. 그 이상 하락 시 `caption_aug_prob`/epoch 조정 후 재시도(LLM 증강분 손실은 불가피 —
  GRPO가 회복 수단).
- **Gate B** (스모크): `grpo_smoke.py` 통과 + 로그에 `rewards/grpo_em_reward`와
  `rewards/grpo_pairwise_reward`가 **개별 표시** + 파싱 성공률 ≥ 95%.
- **Gate C** (GRPO): GRPO val EM ≥ 클린 SFT val EM일 때만 제출. `identity_rate ∈ [0.13, 0.19]`로
  붕괴 재발 없음 확인.

## 8.1 G1) 클린 SFT 재학습 [RunPod]

```bash
CFG=configs/sft_qwen8b_runpod.yaml
SFT=/workspace/code/outputs/sft_train.jsonl        # 클린 데이터 (외부 LLM 증강 없음)
python -m cloud.runpod_train --config $CFG --sft-jsonl $SFT --smoke
python -m cloud.runpod_train --config $CFG --sft-jsonl $SFT --per-device-batch 2 --grad-accum 8
# 병합
python -m cloud.runpod_merge \
  --lora /workspace/snuai/outputs/qwen3vl8b_0716/lora \
  --out  /workspace/snuai/models/qwen3vl8b_clean_merged
```

## 8.2 Gate A) 클린 SFT val 평가

```bash
MODEL=/workspace/snuai/models/qwen3vl8b_clean_merged
python -m src.infer.predict --model $MODEL --split train --fold val \
  --style plain --tta 8 --batch 16 --out /workspace/snuai/outputs/raw_val_clean.jsonl
python -m src.infer.aggregate --raw /workspace/snuai/outputs/raw_val_clean.jsonl \
  --out /workspace/snuai/outputs/pred_val_clean.csv
python -m src.eval.em --pred /workspace/snuai/outputs/pred_val_clean.csv --fold val
# 판정: EM ≥ 0.5582 (0716−2pt) → G2 진행. 이 val EM을 GRPO 후 Gate C 기준선으로 기록.
```

## 8.3 G2) 클린 모델로 hard-case 재채굴 (선택 — GRPO hard 오버샘플용)

```bash
python -m src.infer.predict --model $MODEL --split train --fold train \
  --style plain --tta 4 --batch 16 --out /workspace/snuai/outputs/raw_trainfold_clean.jsonl
python -m src.infer.aggregate --raw /workspace/snuai/outputs/raw_trainfold_clean.jsonl \
  --out /workspace/snuai/outputs/pred_trainfold_clean.csv
python -m src.eval.em --pred /workspace/snuai/outputs/pred_trainfold_clean.csv --fold train
python -m src.preprocess.hard_cases --pred /workspace/snuai/outputs/pred_trainfold_clean.csv \
  --fold train --out /workspace/snuai/outputs/hard_train_cases_clean.csv
# 로컬에서 λ 재확인 (오답 일치쌍 분포):
#   python -m src.eval.pairwise_stats --cases outputs/hard_train_cases_clean.csv
# 이 CSV를 configs/grpo_qwen8b_runpod.yaml의 data.hard_cases_path에 지정 (주석 해제).
```

## 8.4 Gate B) GRPO 스모크 [RunPod A100]

> 전제: `trl`·`datasets` 설치됨(§1). 없으면 `ModuleNotFoundError: trl`로 스모크가 즉시 죽는다.

```bash
CFG=configs/grpo_qwen8b_runpod.yaml         # model: qwen3vl8b_clean_merged
python -m cloud.grpo_smoke --config $CFG --limit 32
# 통과 후 직렬화 스케일 확인 (PIL 인라인 Arrow 병목 조기 노출):
python -m cloud.grpo_smoke --config $CFG --limit 256
# 판정: 루프 진입·생성 OOM 없음 + 로그에 두 보상 성분 개별 표시 + 파싱 성공률 ≥ 95%.
# 실패 유형(버전 비호환·vLLM 필수·PIL 직렬화)이 여기서 확정된다.
```

## 8.5 G4) GRPO 본 학습 [RunPod A100]

```bash
python -m cloud.grpo_train --config configs/grpo_qwen8b_runpod.yaml \
  --sft-jsonl /workspace/code/outputs/sft_train.jsonl
# 끊기면 --resume 추가. 산출: /workspace/snuai/outputs/qwen3vl8b_grpo/lora
python -m cloud.runpod_merge --lora /workspace/snuai/outputs/qwen3vl8b_grpo/lora \
  --out /workspace/snuai/models/qwen3vl8b_grpo_merged
```

학습 중 감시: `rewards/grpo_em_reward`가 상승하는가(EM 직접 최적화 성사), `rewards/grpo_pairwise_reward`가
너무 앞서면 λ 하향; `log_completions`로 identity 표류·파싱 실패 육안 확인.

## 8.6 Gate C) GRPO val 평가 → 제출

```bash
GM=/workspace/snuai/models/qwen3vl8b_grpo_merged
python -m src.infer.predict --model $GM --split train --fold val \
  --style plain --tta 8 --batch 16 --out /workspace/snuai/outputs/raw_val_grpo.jsonl
python -m src.infer.aggregate --raw /workspace/snuai/outputs/raw_val_grpo.jsonl \
  --out /workspace/snuai/outputs/pred_val_grpo.csv
python -m src.eval.em --pred /workspace/snuai/outputs/pred_val_grpo.csv --fold val
# 판정: GRPO val EM ≥ 클린 SFT val EM(§8.2) 그리고 identity_rate ∈ [0.13, 0.19] → C2 제출.
python -m src.infer.predict --model $GM --split test \
  --style plain --tta 8 --batch 16 --out /workspace/snuai/outputs/raw_test_grpo.jsonl
python -m src.infer.aggregate --raw /workspace/snuai/outputs/raw_test_grpo.jsonl \
  --submission /workspace/snuai/outputs/submission_grpo.csv
```

단일 체크포인트 규정: 최종 제출은 클린 SFT vs GRPO 중 val이 높은 **하나**만 선택한다.
