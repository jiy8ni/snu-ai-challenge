"""Phase 2 GRPO 본 학습 — 클린 SFT 병합 모델 위에 EM + pairwise 보상으로 정책 최적화.

스모크(cloud/grpo_smoke.py)가 착수 조건 ③(루프 진입·생성·보상)을 확정한 뒤 이 스크립트로
본 라운드를 돌린다. 스모크와의 차이:
  - 전체 train을 materialize (limit=None), num_train_epochs 학습.
  - 보상 = EM(make_grpo_reward) + pairwise shaping(make_grpo_pairwise_reward) 두 함수.
    GRPOConfig(reward_weights=[1.0, λ])가 λ를 준다 — TRL이 두 성분을 개별 로깅한다.
  - 체크포인트 저장(save_steps)·resume, log_completions로 identity 표류 육안 감시.

base 모델은 클린 SFT를 **병합한 모델**(cfg["model"])이며, build_model이 그 위에 새 LoRA를 붙인다
(기존 0716 플로우·src/infer/predict.py와 동일 구조). 로컬 CPU에서는 unsloth/trl 미설치라
import되지 않는다 — RunPod A100 전용. 보상 로직은 src/train/rewards.py(프레임워크 비의존).
"""

import argparse
import os


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo_qwen8b_runpod.yaml")
    ap.add_argument("--sft-jsonl", default=None, help="GRPO 프롬프트 소스 JSONL (기본 sft_train.jsonl)")
    ap.add_argument("--limit", type=int, default=None, help="프롬프트 수 제한 (스모크·디버그용)")
    ap.add_argument("--resume", action="store_true", help="output_dir의 최신 checkpoint에서 재개")
    ap.add_argument("--output-dir", default=None, help="config output_dir 오버라이드")
    args = ap.parse_args()

    import datasets

    from cloud.grpo_smoke import (
        _patch_trl_sampling_params,
        _patch_trl_vllm_skew,
        build_grpo_examples,
    )

    _patch_trl_vllm_skew()
    from trl import GRPOConfig, GRPOTrainer

    _patch_trl_sampling_params()
    from cloud.train_unsloth import build_model, load_cfg, make_nan_guard
    from src.data.loader import load_paths
    from src.train.rewards import make_grpo_pairwise_reward, make_grpo_reward
    from src.utils.runtime import configure_disk_cache

    cfg = load_cfg(args.config)
    if args.output_dir:
        cfg["output_dir"] = args.output_dir
    g = cfg.get("grpo", {})

    configure_disk_cache(cfg.get("runpod", {}).get("base_dir"))
    os.makedirs(cfg["output_dir"], exist_ok=True)

    paths = load_paths()
    sft = args.sft_jsonl or os.path.join(paths["outputs_dir"], "sft_train.jsonl")
    assert os.path.exists(sft), f"학습 jsonl 없음: {sft}"

    model, processor = build_model(cfg)
    # trl↔transformers 5.x 호환 shim: trl은 model.warnings_issued(구 transformers 속성)를
    # 기대하지만 transformers 5.5에선 없어 GRPOTrainer 생성이 터진다. 빈 dict로 채운다.
    if not hasattr(model, "warnings_issued"):
        model.warnings_issued = {}

    examples = build_grpo_examples(cfg, sft, paths["data_dir"], args.limit)
    train_dataset = datasets.Dataset.from_list(examples)
    print(f"GRPO 데이터셋: {len(train_dataset)} 프롬프트 (limit={args.limit})")

    num_generations = g.get("num_generations", 8)
    reward_weights = g.get("reward_weights", [1.0, 0.25])
    assert len(reward_weights) == 2, "reward_weights는 [EM, pairwise] 2개여야 한다"

    # num_generations가 per_device_train_batch_size를 나눠야 한다 (TRL 제약).
    grpo_cfg = GRPOConfig(
        output_dir=cfg["output_dir"],
        per_device_train_batch_size=num_generations,
        gradient_accumulation_steps=cfg["train"].get("grad_accum", 1),
        num_generations=num_generations,
        max_completion_length=g.get("max_completion_length", 48),
        temperature=g.get("temperature", 1.0),
        beta=g.get("beta", 0.04),
        num_train_epochs=g.get("epochs", 1),
        learning_rate=cfg["train"]["lr"],
        reward_weights=reward_weights,
        logging_steps=cfg["train"].get("logging_steps", 1),
        save_strategy="steps",
        save_steps=g.get("save_steps", 100),
        save_total_limit=g.get("save_total_limit", 2),
        report_to="none",
        seed=cfg["train"]["seed"],
        use_vllm=g.get("use_vllm", False),
        vllm_mode="colocate",           # unsloth fast_inference = 같은 프로세스 내 vLLM (별도 server 아님)
        log_completions=True,           # identity 표류·파싱 실패 육안 감시
        remove_unused_columns=False,    # true_rank 컬럼을 보상 함수까지 살려 보낸다
    )

    trainer = GRPOTrainer(
        model=model,
        processing_class=processor,
        reward_funcs=[make_grpo_reward("true_rank"), make_grpo_pairwise_reward("true_rank")],
        args=grpo_cfg,
        train_dataset=train_dataset,
    )
    trainer.add_callback(make_nan_guard())
    trainer.train(resume_from_checkpoint=args.resume)

    lora_dir = cfg["output_dir"] + "/lora"
    model.save_pretrained(lora_dir)
    processor.save_pretrained(lora_dir)
    print(f"GRPO 완료 — LoRA saved: {lora_dir}")


if __name__ == "__main__":
    main()
