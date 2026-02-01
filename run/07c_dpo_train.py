#!/usr/bin/env python3
# 07c_dpo_train.py
# DPO training on (prompt, chosen, rejected) pairs.
# Output adapter: results/dpo_lora_qwen

import os
import argparse
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, TrainingArguments
from peft import PeftModel

def print_gpu_info():
    print(f"[INFO] torch: {torch.__version__}")
    print(f"[INFO] cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        n = torch.cuda.device_count()
        print(f"[INFO] gpu_count: {n}")
        for i in range(n):
            name = torch.cuda.get_device_name(i)
            cap = torch.cuda.get_device_capability(i)
            props = torch.cuda.get_device_properties(i)
            vram_gb = props.total_memory / (1024**3)
            print(f"[INFO] gpu{i}: {name} cap={cap} vram={vram_gb:.1f}GB")

def load_4bit_base(base_model_dir: str):
    bnb = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
    )
    model = AutoModelForCausalLM.from_pretrained(
        base_model_dir,
        quantization_config=bnb,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj_root", type=str, required=True)

    ap.add_argument("--train_jsonl", type=str, default="results/dpo_pairs.jsonl")
    ap.add_argument("--base_model_dir", type=str, required=True)
    ap.add_argument("--sft_adapter_dir", type=str, required=True)

    ap.add_argument("--out_dir", type=str, default="results/dpo_lora_qwen")

    ap.add_argument("--max_steps", type=int, default=800)
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup_steps", type=int, default=50)
    ap.add_argument("--max_prompt_len", type=int, default=1024)
    ap.add_argument("--max_completion_len", type=int, default=512)
    ap.add_argument("--beta", type=float, default=0.1)  # DPO beta
    args = ap.parse_args()

    proj = args.proj_root
    train_path = os.path.join(proj, args.train_jsonl)
    out_dir = os.path.join(proj, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(train_path):
        raise FileNotFoundError(train_path)
    if not os.path.exists(args.base_model_dir):
        raise FileNotFoundError(args.base_model_dir)
    if not os.path.exists(args.sft_adapter_dir):
        raise FileNotFoundError(args.sft_adapter_dir)

    print_gpu_info()
    print(f"[INFO] train_jsonl: {train_path}")
    print(f"[INFO] base_model_dir: {args.base_model_dir}")
    print(f"[INFO] sft_adapter_dir: {args.sft_adapter_dir}")
    print(f"[INFO] out_dir: {out_dir}")

    # dataset
    ds = load_dataset("json", data_files=train_path, split="train")
    need_cols = {"prompt", "chosen", "rejected"}
    if not need_cols.issubset(set(ds.column_names)):
        raise ValueError(f"DPO jsonl must have columns {need_cols}, got {ds.column_names}")

    # tokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # model (trainable): base 4bit + SFT adapter
    base = load_4bit_base(args.base_model_dir)
    model = PeftModel.from_pretrained(base, args.sft_adapter_dir, is_trainable=True)
    model.train()

    # ref model (frozen): base 4bit + same SFT adapter
    base_ref = load_4bit_base(args.base_model_dir)
    ref_model = PeftModel.from_pretrained(base_ref, args.sft_adapter_dir, is_trainable=False)
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False

    # TrainingArguments
    ta = TrainingArguments(
        output_dir=out_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        logging_steps=20,
        save_steps=200,
        save_total_limit=3,
        fp16=False,
        bf16=torch.cuda.is_available(),
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        report_to=[],
        remove_unused_columns=False,
    )

    # DPOTrainer import (compat)
    try:
        from trl import DPOTrainer
        dpo_trainer_cls = DPOTrainer
        print("[INFO] Using trl.DPOTrainer")
    except Exception as e:
        raise ImportError(f"Cannot import DPOTrainer from trl. Please ensure trl is installed. err={e}")

    # Build trainer (handle API differences)
    kwargs = dict(
        model=model,
        ref_model=ref_model,
        args=ta,
        train_dataset=ds,
        tokenizer=tok,
        beta=float(args.beta),
        max_prompt_length=int(args.max_prompt_len),
        max_length=int(args.max_prompt_len + args.max_completion_len),
    )

    # Some TRL versions want: max_completion_length
    # We'll try a safe fallback if init fails.
    try:
        trainer = dpo_trainer_cls(**kwargs)
    except TypeError:
        kwargs.pop("max_length", None)
        kwargs["max_completion_length"] = int(args.max_completion_len)
        trainer = dpo_trainer_cls(**kwargs)

    trainer.train()

    # save adapter + tokenizer
    trainer.model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    print(f"[OK] DPO done. Saved adapter to: {out_dir}")
    print("[NEXT] Inference: load base + this DPO adapter (instead of SFT adapter).")

if __name__ == "__main__":
    main()