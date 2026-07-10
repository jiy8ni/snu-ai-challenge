# RunPod 실행 가이드

목표: 큰 파일을 전부 `/workspace/snuai` 아래에 두어 root/container disk의
`disk quota exceeded`를 피한다.

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
