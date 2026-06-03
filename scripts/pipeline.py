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
    import tempfile

    from gptqmodel import GPTQModel, QuantizeConfig
    from peft import LoraConfig, get_peft_model

    # 2-bit GPTQ quantization of the (residual-weighted, if SA-SVD) base.
    # gptqmodel loads the base from a path, so the in-memory model (which for
    # SA-SVD already carries the residual weights) is written to a temp dir and
    # quantized. The freshly quantized in-memory model is NOT inference-ready (its
    # quant-kernel buffers are populated only on the load path), so it is saved
    # and reloaded via from_quantized, which initializes those buffers, the bias,
    # and device placement correctly. offload_to_disk=False keeps the quantized
    # layers materialized (the default disk offload leaves meta tensors).
    quant_config = QuantizeConfig(bits=2, group_size=group_size, offload_to_disk=False)
    with tempfile.TemporaryDirectory() as src, tempfile.TemporaryDirectory() as qdir:
        model.save_pretrained(src)
        tokenizer.save_pretrained(src)
        q = GPTQModel.from_pretrained(src, quant_config)
        q.quantize(_calibration_samples(tokenizer))
        (getattr(q, "save", None) or q.save_quantized)(qdir)
        quantized = GPTQModel.from_quantized(qdir, device="cuda:0")

    # QA-LoRA adapter. use_qalora + qalora_group_size enable the group-wise
    # operator that makes the adapter merge into the quantization zero-point.
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=rank,                 # alpha = r, per the thesis
        lora_dropout=0.0,
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

    # SmolLM2's tokenizer ships without a pad token; the SFT collator needs one
    # to batch variable-length sequences. Reuse EOS (standard for causal LMs).
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

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

    data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
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
    """Small calibration set for GPTQ from WikiText train split.

    gptqmodel expects each sample as a dict of tokenized ids, so the raw text is
    tokenized into ``input_ids`` / ``attention_mask`` lists (truncated to keep
    GPTQ calibration memory bounded at 2-bit).
    """
    from datasets import load_dataset

    data = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    texts = [t for t in data["text"] if len(t.strip()) > 0][:n]
    return [dict(tokenizer(t, truncation=True, max_length=1024)) for t in texts]
