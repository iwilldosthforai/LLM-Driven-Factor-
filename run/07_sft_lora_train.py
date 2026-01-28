#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import argparse
import json
import inspect
import subprocess
import platform
from typing import Dict, Any

import torch


def print_gpu_info():
    print(f"[INFO] platform: {platform.platform()}")
    print(f"[INFO] python: {platform.python_version()}")
    print(f"[INFO] torch: {torch.__version__}")
    print(f"[INFO] cuda available: {torch.cuda.is_available()}")
    print(f"[INFO] cuda device count: {torch.cuda.device_count()}")
    print(f"[INFO] CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES')}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"[INFO] gpu{i}: {torch.cuda.get_device_name(i)}")
        # 尝试打印 nvidia-smi（不保证节点有权限，但通常有）
        try:
            out = subprocess.check_output(["nvidia-smi"], stderr=subprocess.STDOUT, text=True)
            print("[INFO] nvidia-smi:\n" + out)
        except Exception as e:
            print(f"[WARN] nvidia-smi failed: {e}")


def build_text(example: Dict[str, Any]) -> str:
    # 训练目标：给定 input，让模型输出 output（JSON）
    inp = str(example.get("input", "")).strip()
    out = str(example.get("output", "")).strip()
    # 尽量简单通用，不依赖 chat template
    return f"{inp}\n\n### ANSWER (JSON)\n{out}\n"


def filter_kwargs(callable_obj, kwargs: Dict[str, Any]) -> Dict[str, Any]:
    """根据函数签名过滤掉不支持的 kwargs，避免版本不兼容报错"""
    sig = inspect.signature(callable_obj)
    allowed = set(sig.parameters.keys())
    return {k: v for k, v in kwargs.items() if k in allowed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proj_root", type=str, required=True)
    ap.add_argument("--train_jsonl", type=str, default="results/sft_train.jsonl")
    ap.add_argument("--base_model_dir", type=str, required=True)
    ap.add_argument("--out_dir", type=str, default="results/sft_lora_qwen")

    ap.add_argument("--max_steps", type=int, default=1200)
    ap.add_argument("--bsz", type=int, default=1)
    ap.add_argument("--grad_accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--warmup_steps", type=int, default=50)
    ap.add_argument("--max_seq_len", type=int, default=2048)

    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--save_steps", type=int, default=200)
    ap.add_argument("--logging_steps", type=int, default=20)
    ap.add_argument("--use_4bit", action="store_true", help="use QLoRA 4bit (recommended)")
    ap.add_argument("--bf16", action="store_true", help="force bf16 if available")
    ap.add_argument("--fp16", action="store_true", help="force fp16")
    args = ap.parse_args()

    print_gpu_info()

    proj = args.proj_root
    train_path = os.path.join(proj, args.train_jsonl)
    out_dir = os.path.join(proj, args.out_dir)
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(train_path):
        raise FileNotFoundError(train_path)
    if not os.path.exists(args.base_model_dir):
        raise FileNotFoundError(args.base_model_dir)

    # lazy imports
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
    from peft import LoraConfig

    # TRL: API 经常变，所以我们用“探测 + 兼容”
    from trl import SFTTrainer

    # dataset
    ds = load_dataset("json", data_files=train_path, split="train")
    if not {"input", "output"}.issubset(set(ds.column_names)):
        raise ValueError(f"[FATAL] jsonl must contain columns: input, output. got={ds.column_names}")

    # 统一生成 text 字段，避免不同版本 SFTTrainer 对 formatting_func 支持差异
    ds_text = ds.map(
        lambda ex: {"text": build_text(ex)},
        remove_columns=[c for c in ds.column_names if c != "text"],
    )

    # tokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model_dir, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"

    # model load
    quant_cfg = None
    if args.use_4bit:
        from transformers import BitsAndBytesConfig
        compute_dtype = torch.bfloat16 if (torch.cuda.is_available() and (args.bf16 or torch.cuda.is_bf16_supported())) else torch.float16
        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model_dir,
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quant_cfg,
    )
    model.config.use_cache = False
    # 省显存建议
    try:
        model.gradient_checkpointing_enable()
        print("[INFO] gradient_checkpointing enabled")
    except Exception as e:
        print(f"[WARN] gradient_checkpointing_enable failed: {e}")

    # LoRA config
    lora = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )

    # precision
    bf16 = False
    fp16 = False
    if torch.cuda.is_available():
        if args.bf16 or torch.cuda.is_bf16_supported():
            bf16 = True
        elif args.fp16:
            fp16 = True

    # training args
    ta = TrainingArguments(
        output_dir=out_dir,
        max_steps=args.max_steps,
        per_device_train_batch_size=args.bsz,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        warmup_steps=args.warmup_steps,
        logging_steps=args.logging_steps,
        save_steps=args.save_steps,
        save_total_limit=3,
        bf16=bf16,
        fp16=fp16,
        optim="paged_adamw_8bit" if args.use_4bit else "adamw_torch",
        lr_scheduler_type="cosine",
        report_to=[],
        remove_unused_columns=False,
        seed=args.seed,
    )

    # --- build trainer kwargs compatibly ---
    trainer_kwargs = dict(
        model=model,
        args=ta,
        train_dataset=ds_text,
        peft_config=lora,
    )

    # 有的版本是 dataset_text_field，有的默认就是 text
    # 有的版本要 processing_class(=tokenizer)，有的版本叫 tokenizer
    sft_sig = inspect.signature(SFTTrainer.__init__).parameters
    if "dataset_text_field" in sft_sig:
        trainer_kwargs["dataset_text_field"] = "text"

    if "processing_class" in sft_sig:
        trainer_kwargs["processing_class"] = tok
    elif "tokenizer" in sft_sig:
        trainer_kwargs["tokenizer"] = tok
    else:
        # 老/怪版本：不接 tokenizer，至少保证数据里有 text
        pass

    if "max_seq_length" in sft_sig:
        trainer_kwargs["max_seq_length"] = args.max_seq_len
    elif "max_length" in sft_sig:
        trainer_kwargs["max_length"] = args.max_seq_len

    # 如果版本支持 packing，可选打开（默认关，避免行为变化）
    # if "packing" in sft_sig:
    #     trainer_kwargs["packing"] = False

    trainer_kwargs = filter_kwargs(SFTTrainer.__init__, trainer_kwargs)

    print("[INFO] SFTTrainer kwargs keys:", sorted(trainer_kwargs.keys()))

    trainer = SFTTrainer(**trainer_kwargs)

    print("[INFO] Run SFT LoRA...")
    trainer.train()

    # 保存
    trainer.save_model(out_dir)
    tok.save_pretrained(out_dir)

    # 额外保存训练信息
    meta = {
        "base_model_dir": args.base_model_dir,
        "train_jsonl": train_path,
        "out_dir": out_dir,
        "max_steps": args.max_steps,
        "bsz": args.bsz,
        "grad_accum": args.grad_accum,
        "lr": args.lr,
        "max_seq_len": args.max_seq_len,
        "use_4bit": args.use_4bit,
        "bf16": bf16,
        "fp16": fp16,
        "torch": torch.__version__,
    }
    with open(os.path.join(out_dir, "run_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[OK] SFT done. Saved adapter to: {out_dir}")
    print("[NEXT] inference: load base + peft adapter, or generate DPO pairs.")


if __name__ == "__main__":
    main()