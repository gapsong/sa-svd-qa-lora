"""
Training / quantization / evaluation helpers for the reproduction script.

These wrap the standard PEFT + GPTQModel + TRL APIs. They are deliberately thin:
the novel part of this repo is SA-SVD initialization (src/sa_svd/), and these
helpers exist only so reproduce.py reads cleanly. Hyperparameters match the
thesis (AdamW, cosine schedule, 2-bit GPTQ, bf16 compute).
"""

from __future__ import annotations

import torch


def attach_qalora_2bit(model, tokenizer, *, rank, group_size, seed):
    """Quantize ``model`` to 2-bit with GPTQ and attach a QA-LoRA adapter.

    Returns a PEFT model ready for fine-tuning. For SA-SVD, the caller then
    overwrites the adapter A/B via ``write_adapter_weights``.
    """
    from gptqmodel import GPTQModel, QuantizeConfig
    from peft import LoraConfig, get_peft_model

    # 2-bit GPTQ quantization of the (residual-weighted, if SA-SVD) base.
    quant_config = QuantizeConfig(bits=2, group_size=group_size)
    quantized = GPTQModel.from_pretrained(model, quant_config)
    quantized.quantize(_calibration_samples(tokenizer))

    # QA-LoRA adapter. use_qalora + qalora_group_size enable the group-wise
    # operator that makes the adapter merge into the quantization zero-point.
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,                 # alpha = r, per the thesis
        lora_dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
        use_qalora=True,
        qalora_group_size=group_size,
        init_lora_weights=True,          # SA-SVD overwrites this afterwards
    )
    return get_peft_model(quantized.model, lora_config)


def train(peft_model, tokenizer, dataset, args):
    """Fine-tune on an instruction dataset with TRL's SFTTrainer."""
    from transformers import TrainingArguments
    from trl import SFTTrainer

    def format_example(ex):
        instr, inp, out = ex["instruction"], ex.get("input", ""), ex["output"]
        prompt = f"### Instruction:\n{instr}\n"
        if inp:
            prompt += f"### Input:\n{inp}\n"
        prompt += f"### Response:\n{out}"
        return {"text": prompt}

    dataset = dataset.map(format_example)

    targs = TrainingArguments(
        output_dir="./_ckpt",
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        max_steps=args.max_steps,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        bf16=True,
        logging_steps=10,
        save_strategy="no",
        seed=args.seed,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=peft_model,
        args=targs,
        train_dataset=dataset,
        processing_class=tokenizer,
    )
    trainer.train()


@torch.no_grad()
def evaluate_wikitext(peft_model, tokenizer, *, stride: int = 512,
                      max_length: int = 1024) -> float:
    """WikiText-2 perplexity via sliding-window negative log-likelihood.

    This is the repair signal: it cannot be faked by learning the instruction
    format, because WikiText is never seen during fine-tuning.
    """
    from datasets import load_dataset

    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    text = "\n\n".join(data["text"])
    enc = tokenizer(text, return_tensors="pt").input_ids.to(peft_model.device)

    nlls, n_tokens = [], 0
    peft_model.eval()
    for begin in range(0, enc.size(1), stride):
        end = min(begin + max_length, enc.size(1))
        trg_len = end - begin
        input_ids = enc[:, begin:end]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100
        out = peft_model(input_ids, labels=target_ids)
        nlls.append(out.loss * trg_len)
        n_tokens += trg_len
        if end == enc.size(1):
            break

    return torch.exp(torch.stack(nlls).sum() / n_tokens).item()


def _calibration_samples(tokenizer, n: int = 128):
    """Small calibration set for GPTQ from WikiText train split."""
    from datasets import load_dataset

    data = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    samples = [t for t in data["text"] if len(t.strip()) > 0][:n]
    return [tokenizer(s, return_tensors="pt").input_ids for s in samples]
