"""Track B 학습 로직 (Kaggle T4에서 실행, Unsloth FastVisionModel + TRL SFT).

노트북은 이 스크립트를 얇게 감싸기만 한다. 로컬 CPU에서는 import되지 않는다
(unsloth/bitsandbytes 미설치). 로컬에서는 vl_dataset/targets만 단위 테스트한다.

핵심 안전장치 (docs/rules.md, 계획 리스크 §1):
  - fp16 안정화: max_grad_norm 0.3 + NaN-loss 배치 스킵 콜백
  - train_on_responses_only (응답부만 loss)
  - 순열 증강은 VLSFTDataset이 담당 (no_ordering rank 고정 규칙 포함)
"""

import os

import yaml

from src.utils.runtime import configure_disk_cache


def load_cfg(path):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(cfg):
    from unsloth import FastVisionModel

    model, processor = FastVisionModel.from_pretrained(
        cfg["model"],
        load_in_4bit=cfg["load_in_4bit"],
        use_gradient_checkpointing="unsloth",
    )
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=cfg["lora"]["finetune_vision_layers"],
        finetune_language_layers=cfg["lora"]["finetune_language_layers"],
        finetune_attention_modules=cfg["lora"]["finetune_attention_modules"],
        finetune_mlp_modules=cfg["lora"]["finetune_mlp_modules"],
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        random_state=cfg["train"]["seed"],
    )
    # 작은 프레임의 강제 업스케일 방지 + 이상치 캡 (계획 §1.2)
    # 구형(min/max_pixels 속성)·신형(size dict, 픽셀 "개수" 예산) 어느 쪽이 유효한지
    # 버전마다 다르고 무시되는 쪽도 대입은 예외 없이 성공하므로 둘 다 설정한다
    # (predict.apply_pixel_caps와 동일 규칙 — 2026-07-06 silent no-op 교훈).
    ip = getattr(processor, "image_processor", None)
    if ip is not None:
        for name, val in (("min_pixels", cfg["pixels"]["min"]), ("max_pixels", cfg["pixels"]["max"])):
            try:
                setattr(ip, name, val)
            except AttributeError:
                pass
        try:
            ip.size = {"shortest_edge": cfg["pixels"]["min"], "longest_edge": cfg["pixels"]["max"]}
        except AttributeError:
            pass
    return model, processor


def make_nan_guard():
    """fp16 NaN/Inf loss 감시 콜백 (T4 안정화). TrainerCallback 상속 필수 —
    Trainer가 모든 이벤트 메서드를 호출하므로 no-op 기본 구현이 필요하다."""
    import math

    from transformers import TrainerCallback

    class NanGuardCallback(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            loss = (logs or {}).get("loss")
            if loss is not None and not math.isfinite(loss):
                print(f"[NanGuard] step {state.global_step}: non-finite loss {loss}")
            return control

    return NanGuardCallback()


def make_collator(model, processor, cfg):
    """응답부만 loss 계산하는 collator. 우리 데이터셋은 매 호출 증강을 새로 뽑는
    동적 방식이라 HF .map 기반 train_on_responses_only는 쓸 수 없다 (lists 에러).
    신형 unsloth는 collator 인자로 지원 -> 시도 후 미지원 버전이면 전체 시퀀스 학습."""
    from unsloth.trainer import UnslothVisionDataCollator

    if cfg["train"].get("train_on_responses_only"):
        try:
            return UnslothVisionDataCollator(
                model, processor,
                train_on_responses_only=True,
                instruction_part="<|im_start|>user\n",
                response_part="<|im_start|>assistant\n",
            )
        except TypeError:
            print("[경고] 이 unsloth collator는 응답부 마스킹 미지원 -> 전체 시퀀스 학습으로 진행")
    return UnslothVisionDataCollator(model, processor)


def make_trainer(model, processor, dataset, cfg):
    from trl import SFTConfig, SFTTrainer
    from unsloth import FastVisionModel, is_bf16_supported

    FastVisionModel.for_training(model)
    bf16 = is_bf16_supported()  # T4=False(fp16 경로), L4/G4/A100=True
    t = cfg["train"]
    steps_per_epoch = max(1, len(dataset) // (t["per_device_batch"] * t["grad_accum"]))
    warmup_steps = max(1, int(t["warmup_ratio"] * steps_per_epoch * t["epochs"]))
    sft_cfg = SFTConfig(
        per_device_train_batch_size=t["per_device_batch"],
        gradient_accumulation_steps=t["grad_accum"],
        warmup_steps=warmup_steps,  # warmup_ratio는 transformers v5.2에서 제거 예정
        num_train_epochs=t["epochs"],
        learning_rate=t["lr"],
        lr_scheduler_type=t["scheduler"],
        weight_decay=t["weight_decay"],
        max_grad_norm=t["max_grad_norm"],
        fp16=not bf16,
        bf16=bf16,
        logging_steps=t["logging_steps"],
        save_steps=t["save_steps"],
        save_total_limit=t.get("save_total_limit", 2),  # RunPod/root disk quota 보호
        optim="paged_adamw_8bit",
        seed=t["seed"],
        output_dir=cfg["output_dir"],
        report_to="none",
        remove_unused_columns=False,
        dataset_kwargs={"skip_prepare_dataset": True},
        max_seq_length=t["max_seq_len"],
        dataset_num_proc=1,  # 순열 증강 rng 재현성
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=processor,
        data_collator=make_collator(model, processor, cfg),
        train_dataset=dataset,
        args=sft_cfg,
    )
    return trainer


def run(cfg_path, jsonl_path, data_dir, resume=False, limit=None, output_dir=None,
        train_overrides=None):
    """output_dir: Colab에서는 Drive 경로로 지정 (세션 휘발 대비).
    train_overrides: GPU별 배치 조정 등 train 섹션 오버라이드.
      예) A100: train_overrides={"per_device_batch": 4, "grad_accum": 4}  # 유효 16 유지
    """
    from src.train.vl_dataset import VLSFTDataset

    cfg = load_cfg(cfg_path)
    if output_dir:
        cfg["output_dir"] = output_dir
    if train_overrides:
        cfg["train"].update(train_overrides)
    configure_disk_cache(cfg.get("runpod", {}).get("base_dir"))
    os.makedirs(cfg["output_dir"], exist_ok=True)
    model, processor = build_model(cfg)
    dataset = VLSFTDataset(
        jsonl_path, data_dir,
        style=cfg["data"]["style"], augment=cfg["data"]["perm_augment"],
        crop=cfg["data"]["letterbox_crop"], seed=cfg["train"]["seed"], limit=limit,
        oversample_no_ordering=cfg["data"].get("oversample_no_ordering", 1),
        caption_aug_prob=cfg["data"].get("caption_aug_prob", 0.0),
        caption_aug_source=cfg["data"].get("caption_aug_source", "rule"),
        llm_caption_field=cfg["data"].get("llm_caption_field", "caption_llm_variants"),
        hard_cases_path=cfg["data"].get("hard_cases_path"),
        hard_aug_repeats_field=cfg["data"].get("hard_aug_repeats_field", "caption_aug_repeats"),
        hard_aug_max_repeats=cfg["data"].get("hard_aug_max_repeats", 5),
    )
    trainer = make_trainer(model, processor, dataset, cfg)
    trainer.add_callback(make_nan_guard())
    stats = trainer.train(resume_from_checkpoint=resume)
    model.save_pretrained(cfg["output_dir"] + "/lora")
    processor.save_pretrained(cfg["output_dir"] + "/lora")
    print(f"train done: {stats.metrics if hasattr(stats, 'metrics') else stats}")
    return cfg["output_dir"] + "/lora"
