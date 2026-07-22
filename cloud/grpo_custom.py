"""Phase 2 GRPO — 커스텀 루프 (TRL/vLLM 비의존).

배경: TRL+unsloth+vLLM(및 HF) 비전 GRPO는 생성 시 우리 멀티모달 프롬프트를 모델에
제대로 전달하지 못해(2026-07-21 확정: 완성물이 프롬프트와 무관한 랜덤 텍스트, 전 보상 0)
전량 폐기했다. 대신 이 루프는 predict.py의 **검증된 생성 경로**(chat template + 4프레임을
그대로 먹여 정확한 `Order:/Answer:`를 내는 그것)를 샘플링으로 바꿔 재사용한다. 프레임워크
생성/보상 배선이 없으므로 버전 벽이 없다.

알고리즘 (프롬프트 하나당):
  1) K개 completion 샘플 (do_sample, temperature)  ← predict의 generate와 동일 입력
  2) 각 completion 보상 = w_em·sequence_reward + w_pw·pairwise_reward (src/train/rewards.py)
  3) 그룹-상대 어드밴티지 A_i = (r_i - mean)/(std+eps)
  4) completion 토큰 log-prob를 정책으로 다시 forward해서 구함 (grad on)
  5) loss = -mean(A_i · mean_token_logprob_i), 역전파 → LoRA(언어층) 업데이트
KL(참조모델)은 v1에서 생략(β=0). 그룹 베이스라인만으로도 학습 신호는 성립한다.

로컬 CPU에선 unsloth 미설치라 import 불가 — RunPod A100 전용. 4bit QLoRA(vLLM 안 쓰므로
merge 문제 없음). 실행: docs/runpod.md §8. 보상 로직은 프레임워크 비의존(tests/test_rewards.py).
"""

import argparse
import os
import random

import torch


def _load_model(cfg, device, grad_ckpt=True):
    """4bit QLoRA 모델 로드 + 언어층 LoRA. predict.load_model과 동일한 4bit 경로.

    grad_ckpt=False면 gradient checkpointing 없이 로드 — backward에서 activation 재계산이
    빠지므로 30~40% 가속. A100 80GB + 8B 4bit + 짧은 completion에선 VRAM이 감당한다
    (OOM이 나면 --no-grad-ckpt를 빼서 원복).
    """
    from unsloth import FastVisionModel

    from src.infer.predict import apply_pixel_caps

    model, processor = FastVisionModel.from_pretrained(
        cfg["model"], load_in_4bit=True,
        use_gradient_checkpointing="unsloth" if grad_ckpt else False,
    )
    lo = cfg["lora"]
    model = FastVisionModel.get_peft_model(
        model,
        finetune_vision_layers=lo.get("finetune_vision_layers", False),
        finetune_language_layers=lo.get("finetune_language_layers", True),
        finetune_attention_modules=lo.get("finetune_attention_modules", True),
        finetune_mlp_modules=lo.get("finetune_mlp_modules", True),
        r=lo["r"], lora_alpha=lo["alpha"], lora_dropout=lo["dropout"],
        random_state=cfg["train"]["seed"],
    )
    apply_pixel_caps(processor)
    return model, processor


def _generate_k(model, processor, prompt_msgs, images, k, max_new, temperature, top_p, device):
    """프롬프트 하나에 대해 K개 completion 텍스트 샘플 (predict.generate_batch의 샘플링 버전)."""
    from unsloth import FastVisionModel

    FastVisionModel.for_inference(model)
    text = processor.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    inputs = processor(text=[text], images=[images], return_tensors="pt")
    inputs = {kk: vv.to(device) for kk, vv in inputs.items()}
    with torch.inference_mode():
        out = model.generate(
            **inputs, max_new_tokens=max_new, do_sample=True,
            temperature=temperature, top_p=top_p, num_return_sequences=k,
        )
    plen = inputs["input_ids"].shape[1]
    return processor.batch_decode(out[:, plen:], skip_special_tokens=True)


def _completion_logprobs(model, processor, prompt_msgs, completions, images, device):
    """각 completion의 (정책 하) 토큰 평균 log-prob. grad 살아있음. 반환 [K].

    prompt+completion 전체를 오른쪽 패딩으로 배치 forward → next-token log-prob를
    completion 구간에서만 모은다. Lp = 프롬프트 토큰 길이(이미지 확장 포함)로 경계를 잡는다.
    """
    K = len(completions)
    prompt_text = processor.apply_chat_template(prompt_msgs, tokenize=False, add_generation_prompt=True)
    p_in = processor(text=[prompt_text], images=[images], return_tensors="pt")
    Lp = p_in["input_ids"].shape[1]

    full_texts = []
    for c in completions:
        msgs = list(prompt_msgs) + [{"role": "assistant", "content": [{"type": "text", "text": c}]}]
        full_texts.append(processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False))

    old_side = processor.tokenizer.padding_side
    processor.tokenizer.padding_side = "right"   # 실제 토큰이 앞, 패딩이 뒤 → 경계 계산이 단순
    inputs = processor(text=full_texts, images=[images] * K, return_tensors="pt", padding=True)
    processor.tokenizer.padding_side = old_side
    inputs = {kk: vv.to(device) for kk, vv in inputs.items()}

    logits = model(**inputs).logits[:, :-1, :]        # [K, L-1, V] (t번째가 t+1 토큰 예측)
    targets = inputs["input_ids"][:, 1:]              # [K, L-1]
    logp = torch.log_softmax(logits.float(), dim=-1).gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    attn = inputs["attention_mask"][:, 1:].bool()     # [K, L-1] 실토큰=1
    pos = torch.arange(targets.shape[1], device=device).unsqueeze(0)  # [1, L-1]
    comp_mask = (pos >= (Lp - 1)) & attn              # completion 토큰(=input_ids[Lp:])만
    comp_len = comp_mask.sum(dim=1).clamp(min=1)
    return (logp * comp_mask).sum(dim=1) / comp_len   # [K] 토큰 평균 log-prob


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/grpo_qwen8b_runpod.yaml")
    ap.add_argument("--sft-jsonl", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--num-generations", type=int, default=8, help="프롬프트당 샘플 수 K")
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top-p", type=float, default=1.0)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--accum", type=int, default=4, help="옵티마이저 스텝당 프롬프트 수")
    ap.add_argument("--save-steps", type=int, default=100)
    ap.add_argument("--output-dir", default=None)
    ap.add_argument("--max-steps", type=int, default=None, help="스모크용 옵티마이저 스텝 상한")
    ap.add_argument("--no-grad-ckpt", action="store_true",
                    help="gradient checkpointing 끔 — backward 가속. OOM 시 이 플래그 제거")
    ap.add_argument("--resume-lora", default=None,
                    help="저장된 LoRA 디렉토리(adapter_model.safetensors)에서 이어서 학습")
    ap.add_argument("--skip-prompts", type=int, default=0,
                    help="첫 에폭에서 셔플 후 앞 N개 프롬프트 건너뜀 — resume 시 이미 학습한 "
                         "구간 재방문 방지 (N = 마지막 step × accum, seed 동일 전제)")
    args = ap.parse_args()

    from cloud.grpo_smoke import build_grpo_examples
    from cloud.train_unsloth import load_cfg
    from src.data.loader import load_paths
    from src.train.rewards import pairwise_reward, sequence_reward
    from src.utils.permutation import parse_permutation
    from src.utils.runtime import configure_disk_cache

    cfg = load_cfg(args.config)
    out_dir = args.output_dir or cfg["output_dir"]
    g = cfg.get("grpo", {})
    w_em, w_pw = g.get("reward_weights", [1.0, 0.25])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    configure_disk_cache(cfg.get("runpod", {}).get("base_dir"))
    os.makedirs(out_dir, exist_ok=True)
    paths = load_paths()
    sft = args.sft_jsonl or os.path.join(paths["outputs_dir"], "sft_train.jsonl")
    assert os.path.exists(sft), f"학습 jsonl 없음: {sft}"

    model, processor = _load_model(cfg, device, grad_ckpt=not args.no_grad_ckpt)
    processor.tokenizer.padding_side = "left"   # 생성 기본(로그확률 함수가 필요시 right로 임시전환)

    if args.resume_lora:
        from safetensors.torch import load_file

        from peft import set_peft_model_state_dict

        sd_path = os.path.join(args.resume_lora, "adapter_model.safetensors")
        assert os.path.exists(sd_path), f"어댑터 없음: {sd_path}"
        set_peft_model_state_dict(model, load_file(sd_path))
        print(f"LoRA 이어서 학습: {args.resume_lora} (옵티마이저 상태는 새로 시작)")

    examples = build_grpo_examples(cfg, sft, paths["data_dir"], args.limit)
    print(f"GRPO(custom) 데이터셋: {len(examples)} 프롬프트 | K={args.num_generations} | "
          f"reward_weights=[{w_em}, {w_pw}]")

    from unsloth import FastVisionModel

    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=cfg["train"]["lr"])
    rng = random.Random(cfg["train"]["seed"])
    step = 0
    running_r, running_em = [], []

    for epoch in range(args.epochs):
        order = list(range(len(examples)))
        rng.shuffle(order)   # seed 고정 → run마다 동일 순서 (skip-prompts 이어달리기의 전제)
        if epoch == 0 and args.skip_prompts:
            order = order[args.skip_prompts:]
            print(f"skip: 앞 {args.skip_prompts}개 프롬프트 건너뜀 → 남은 {len(order)}개"
                  f" (~{len(order) // args.accum} 스텝)")
        opt.zero_grad()
        for i, ei in enumerate(order):
            ex = examples[ei]
            prompt, true_rank = ex["prompt"], ex["true_rank"]
            images = [c["image"] for c in prompt[0]["content"] if c.get("type") == "image"]

            comps = _generate_k(model, processor, prompt, images, args.num_generations,
                                args.max_new_tokens, args.temperature, args.top_p, device)
            rewards = [w_em * sequence_reward(c, true_rank) + w_pw * pairwise_reward(c, true_rank)
                       for c in comps]
            rt = torch.tensor(rewards, device=device, dtype=torch.float32)
            adv = (rt - rt.mean()) / (rt.std() + 1e-6)

            running_r.append(rt.mean().item())
            running_em.append(sum(parse_permutation(c) == true_rank for c in comps) / len(comps))

            FastVisionModel.for_training(model)
            if args.no_grad_ckpt:
                model.gradient_checkpointing_disable()   # for_training이 되켤 수 있어 매번 확실히 끔
            logp = _completion_logprobs(model, processor, prompt, comps, images, device)  # [K] grad
            loss = -(adv.detach() * logp).mean() / args.accum
            loss.backward()

            if (i + 1) % args.accum == 0:
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                opt.zero_grad()
                step += 1
                mr = sum(running_r) / len(running_r)
                me = sum(running_em) / len(running_em)
                print(f"[epoch {epoch} step {step}] mean_reward={mr:.3f} EM={me:.3f} "
                      f"| sample: {comps[0][:120]!r}")
                running_r, running_em = [], []
                if step % args.save_steps == 0:
                    model.save_pretrained(os.path.join(out_dir, "lora"))
                    processor.save_pretrained(os.path.join(out_dir, "lora"))
                    print(f"[checkpoint] step {step} → {os.path.join(out_dir, 'lora')}")
                if args.max_steps and step >= args.max_steps:
                    break
        if args.max_steps and step >= args.max_steps:
            break

    lora_dir = os.path.join(out_dir, "lora")
    model.save_pretrained(lora_dir)
    processor.save_pretrained(lora_dir)
    print(f"GRPO(custom) 완료 — LoRA saved: {lora_dir}")


if __name__ == "__main__":
    main()
