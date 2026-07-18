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
from src.train.rewards import make_grpo_reward
from src.utils.permutation import parse_permutation
from src.utils.runtime import configure_disk_cache


def _assistant_text(messages):
    """VLSFTDataset이 만든 messages의 assistant 타깃 텍스트를 뽑는다."""
    content = messages[-1]["content"]
    if isinstance(content, list):
        return " ".join(p.get("text", "") for p in content if p.get("type") == "text")
    return str(content)


def build_grpo_examples(cfg, jsonl_path, data_dir, limit):
    """SFT 파이프라인으로 (prompt, true_rank) 쌍을 materialize.

    각 예제를 한 번만 뽑아 프롬프트(user 턴 = 이미지 4장 + plain 지시문)와 그 뷰의 진실
    rank(assistant 타깃의 Answer)를 **같은 호출에서** 고정한다 — 증강이 매 __getitem__마다
    재추출되므로 프롬프트와 라벨의 일관성을 위해 반드시 한 번에 분리한다.
    """
    from src.train.targets import IDENTITY_PRIOR
    from src.train.vl_dataset import VLSFTDataset

    d = cfg["data"]
    ds = VLSFTDataset(
        jsonl_path, data_dir,
        style=d["style"], augment=d["perm_augment"], crop=d["letterbox_crop"],
        seed=cfg["train"]["seed"], limit=limit,
        oversample_no_ordering=d.get("oversample_no_ordering", 1),
        caption_aug_prob=d.get("caption_aug_prob", 0.0),
        caption_aug_source=d.get("caption_aug_source", "rule"),
        llm_caption_field=d.get("llm_caption_field", "caption_llm_variants"),
        identity_prior=d.get("identity_prior", IDENTITY_PRIOR),
    )
    examples = []
    for i in range(len(ds)):
        messages = ds[i]["messages"]
        true_rank = parse_permutation(_assistant_text(messages))
        assert true_rank is not None, f"타깃에서 rank 파싱 실패 (i={i})"
        examples.append({"prompt": messages[:-1], "true_rank": true_rank})
    return examples


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/sft_qwen8b_v2_runpod.yaml")
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
    from trl import GRPOConfig, GRPOTrainer

    from cloud.train_unsloth import build_model, load_cfg

    configure_disk_cache()
    cfg = load_cfg(args.config)
    paths = load_paths()
    sft = args.sft_jsonl or os.path.join(paths["outputs_dir"], "sft_train_llm_aug_hard_0717.jsonl")
    assert os.path.exists(sft), f"학습 jsonl 없음: {sft}"

    model, processor = build_model(cfg)

    examples = build_grpo_examples(cfg, sft, paths["data_dir"], args.limit)
    # PIL 이미지가 messages 안에 인라인돼 있다. datasets가 Arrow로 직렬화하며 이를 못 다루면
    # 여기서 터진다 — 그 경우가 곧 "compatibility 실패 유형: PIL 직렬화"의 확정 지점이다
    # (폴백: 커스텀 collator로 torch 데이터셋을 직접 넘기는 경로. 별도 계획에서 처리).
    train_dataset = datasets.Dataset.from_list(examples)
    print(f"GRPO 스모크 데이터셋: {len(train_dataset)} 프롬프트 (limit={args.limit})")

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
        logging_steps=1,
        save_strategy="no",
        report_to="none",
        seed=cfg["train"]["seed"],
        use_vllm=args.use_vllm,
        remove_unused_columns=False,   # true_rank 컬럼을 보상 함수까지 살려 보낸다
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=[make_grpo_reward("true_rank")],
        args=grpo_cfg,
        train_dataset=train_dataset,
    )
    trainer.train()
    print("GRPO 스모크 통과 — FastVisionModel + GRPOTrainer 루프 진입·생성·보상 OK. "
          "본 라운드는 별도 계획으로 착수 가능 (착수 조건 ③ 충족).")


if __name__ == "__main__":
    main()
