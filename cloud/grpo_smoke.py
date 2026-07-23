"""Phase 2 GRPO 스모크 — unsloth FastVisionModel(8B 4bit) + TRL GRPOTrainer 호환성 확인.

**목적은 학습이 아니라 "루프 진입 가능 여부 확정"이다.** limit개 프롬프트로 1~2 옵티마이저
스텝만 돌려, 아래 세 가지가 성립하는지 본선 라운드 전에 값싸게 검증한다:
  1) vision messages(4장 이미지 + plain 지시문) 프롬프트를 GRPOTrainer가 수용하는가
  2) 학습 중 생성(num_generations=8, temperature 샘플링)이 A100에서 OOM 없이 도는가
  3) 보상 함수가 디코딩된 completion에서 parse_permutation으로 rank를 뽑아 EM 보상을 내는가

로컬 CPU에서는 import되지 않는다(unsloth/trl 미설치) — RunPod A100에서만 실행한다.
프레임워크 연결부가 여기 다 모여 있어, 실패하면 유형(버전 비호환·vLLM 필수·collator 충돌·
PIL 직렬화)이 이 스크립트 한 곳에서 드러난다. 실제 GRPO 라운드는 별도 계획으로 착수한다.

reward 로직은 src/train/rewards.py (프레임워크 비의존, tests/test_rewards.py로 고정).
prompt/이미지 파이프라인은 src/train/vl_dataset.py를 그대로 재사용한다 (SFT와 동일 전처리).
"""

import argparse
import os

from src.data.loader import load_paths
from src.train.rewards import make_grpo_pairwise_reward, make_grpo_reward
from src.utils.permutation import parse_permutation
from src.utils.runtime import configure_disk_cache


def _patch_trl_vllm_skew():
    """trl↔vLLM 버전 skew 우회 (trl import 전에 호출).

    일부 trl이 vllm.sampling_params.GuidedDecodingParams를 import하는데 최신 vLLM은
    structured-output 리팩터링으로 그 이름을 없앴다. guided decoding은 GRPO 보상
    (EM/pairwise)과 무관하게 우리가 쓰지 않으므로, 이름만 더미로 채워 import를 통과시킨다.
    """
    try:
        import vllm.sampling_params as _sp
    except Exception:
        return
    if not hasattr(_sp, "GuidedDecodingParams"):
        class GuidedDecodingParams:  # noqa: N801 (vLLM 원본 이름 유지)
            def __init__(self, *a, **k):
                pass
        _sp.GuidedDecodingParams = GuidedDecodingParams


def _patch_trl_sampling_params():
    """trl↔vLLM SamplingParams 인자 skew를 통째로 흡수 (trl import 후 호출).

    trl이 SamplingParams(**kwargs)에 넘기는 인자 중 설치된 vLLM 버전이 모르는 것
    (truncate_prompt_tokens 등)을 버리고 받는 것만 남긴다. 인자 하나씩 쫓지 않고
    이 arg-skew 클래스 전체를 한 번에 처리한다.
    """
    try:
        import inspect

        import trl.trainer.grpo_trainer as gt
    except Exception:
        return
    SP = getattr(gt, "SamplingParams", None)
    if SP is None:
        return
    try:
        valid = set(inspect.signature(SP).parameters)
    except (TypeError, ValueError):
        return
    if "kwargs" in valid:   # **kwargs를 받으면 필터 불필요(이미 관대)
        return

    def _filtered(*a, **k):
        return SP(*a, **{kk: vv for kk, vv in k.items() if kk in valid})

    gt.SamplingParams = _filtered


def _assistant_text(messages):
    """VLSFTDataset이 만든 messages의 assistant 타깃 텍스트를 뽑는다."""
    content = messages[-1]["content"]
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if p.get("type") == "text")
    return str(content)


def build_grpo_examples(cfg, jsonl_path, data_dir, limit, offset=0):
    """SFT 파이프라인으로 (prompt, true_rank) 쌍을 materialize.

    각 예제를 한 번만 뽑아 프롬프트(user 턴 = 이미지 4장 + plain 지시문)와 그 뷰의 진실
    rank(assistant 타깃의 Answer)를 **같은 호출에서** 고정한다 — 증강이 매 __getitem__마다
    재추출되므로 프롬프트와 라벨의 일관성을 위해 반드시 한 번에 분리한다.

    offset: jsonl 앞 N개 레코드를 건너뛴다 — 이전 run이 --limit N으로 학습한 구간을 피해
    "안 본 프롬프트"로 이어 학습할 때 사용 (예: offset=2000, limit=2500 → 레코드 2000~4499).
    증강은 인덱스-결정적이라 슬라이스와 무관하게 동일 뷰가 나온다.
    """
    from src.train.targets import IDENTITY_PRIOR
    from src.train.vl_dataset import VLSFTDataset

    d = cfg["data"]
    ds = VLSFTDataset(
        jsonl_path, data_dir,
        style=d["style"], augment=d["perm_augment"], crop=d["letterbox_crop"],
        seed=cfg["train"]["seed"], limit=None if limit is None else offset + limit,
        oversample_no_ordering=d.get("oversample_no_ordering", 1),
        caption_aug_prob=d.get("caption_aug_prob", 0.0),
        identity_prior=d.get("identity_prior", IDENTITY_PRIOR),
    )
    assert offset < len(ds), f"offset {offset} >= 데이터 {len(ds)}개"
    examples = []
    for i in range(offset, len(ds)):
        messages = ds[i]["messages"]
        true_rank = parse_permutation(_assistant_text(messages))
        assert true_rank is not None, f"타깃에서 rank 파싱 실패 (i={i})"
        examples.append({"prompt": messages[:-1], "true_rank": true_rank})
    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo_qwen8b_runpod.yaml")
    ap.add_argument("--sft-jsonl", default=None, help="override training JSONL path")
    ap.add_argument("--limit", type=int, default=32)
    ap.add_argument("--num-generations", type=int, default=8, help="프롬프트당 샘플 수")
    ap.add_argument("--max-steps", type=int, default=2, help="옵티마이저 스텝 (스모크)")
    ap.add_argument("--max-completion-length", type=int, default=48, help="plain ~30tok + 여유")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--use-vllm", action="store_true",
                    help="vLLM 생성 가속(별도 호환 축). 기본 off = 순수 HF 생성으로 호환성부터 확인")
    args = ap.parse_args()

    import datasets

    _patch_trl_vllm_skew()
    from trl import GRPOConfig, GRPOTrainer

    _patch_trl_sampling_params()
    from cloud.train_unsloth import build_model, load_cfg

    configure_disk_cache()
    cfg = load_cfg(args.config)
    paths = load_paths()
    sft = args.sft_jsonl or os.path.join(paths["outputs_dir"], "sft_train.jsonl")
    assert os.path.exists(sft), f"학습 jsonl 없음: {sft}"

    model, processor = build_model(cfg)
    # trl↔transformers 5.x 호환 shim: trl은 model.warnings_issued(구 transformers 속성)를
    # 기대하지만 transformers 5.5에선 없어 GRPOTrainer 생성이 터진다. 빈 dict로 채운다.
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    examples = build_grpo_examples(cfg, sft, paths["data_dir"], args.limit)
    # PIL 이미지가 messages 안에 인라인돼 있다. datasets가 Arrow로 직렬화하며 이를 못 다루면
    # 여기서 터진다 — 그 경우가 곧 "compatibility 실패 유형: PIL 직렬화"의 확정 지점이다
    # (폴백: 커스텀 collator로 torch 데이터셋을 직접 넘기는 경로. 별도 계획에서 처리).
    train_dataset = datasets.Dataset.from_list(examples)
    print(f"GRPO 스모크 데이터셋: {len(train_dataset)} 프롬프트 (limit={args.limit})")

    # 본 라운드와 동일한 2-보상(EM + pairwise) 설정을 그대로 스모크한다 — Gate B에서
    # reward_weights·성분 개별 로깅까지 검증하려면 실제 구성과 일치해야 한다.
    reward_weights = cfg.get("grpo", {}).get("reward_weights", [1.0, 0.25])

    # num_generations가 per_device_train_batch_size를 나눠야 한다 (TRL 제약).
    grpo_cfg = GRPOConfig(
        output_dir=cfg["output_dir"] + "_grpo_smoke",
        per_device_train_batch_size=args.num_generations,
        gradient_accumulation_steps=1,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_length,
        temperature=args.temperature,
        max_steps=args.max_steps,
        learning_rate=cfg["train"]["lr"],
        reward_weights=reward_weights,
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        seed=cfg["train"]["seed"],
        use_vllm=args.use_vllm,
        vllm_mode="colocate",   # unsloth fast_inference = 같은 프로세스 내 vLLM (별도 server 아님)
        log_completions=True,
        remove_unused_columns=False,   # true_rank 컬럼을 보상 함수까지 살려 보낸다
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=[make_grpo_reward("true_rank"), make_grpo_pairwise_reward("true_rank")],
        args=grpo_cfg,
        train_dataset=train_dataset,
    )
    trainer.train()
    print("GRPO 스모크 통과 — FastVisionModel + GRPOTrainer 루프 진입·생성·EM+pairwise 보상 OK. "
          "본 라운드는 cloud/grpo_train.py로 착수 (착수 조건 ③ 충족).")


if __name__ == "__main__":
    main()
