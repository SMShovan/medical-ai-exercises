"""Time MMedPO (LLaVA-Med-1.5 7B + MMedPO LoRA) inference on one GPU.

Prompt construction and generation settings mirror the authors' scripts
(MMedPO/inference/llava-med-1.5_{vqa,report}.py): vicuna_v1 template, batch size 1,
fp16, temperature 0.2 sampling, max_new_tokens 64 (VQA) / 1024 (report).
Only timing and bookkeeping are added.
"""
import argparse
import json
import os
import random
import re
import sys
import time

import numpy as np
import torch
from PIL import Image
from transformers import set_seed
from transformers.generation.streamers import BaseStreamer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "MMedPO", "train", "dpo"))
from llava.constants import DEFAULT_IMAGE_TOKEN, IMAGE_TOKEN_INDEX
from llava.conversation import conv_templates
from llava.mm_utils import get_model_name_from_path, process_images, tokenizer_image_token
from llava.model.builder import load_pretrained_model
from llava.utils import disable_torch_init

REPORT_PROMPTS = [
    "<image>\nWalk through the important details of the image",
    "<image>\nClarify the contents of the displayed image with great detail",
    "<image>\nExamine the image closely and share its details",
    "<image>\nClarify the contents of the displayed image with great detail",
    "<image>\nAnalyze the image in a comprehensive and detailed manner",
]


class TimingStreamer(BaseStreamer):
    """generate() calls put() once before the first decode step, then once per new token."""

    def __init__(self):
        self.times = []

    def put(self, value):
        self.times.append(time.perf_counter())

    def end(self):
        pass


def pct(xs, q):
    return float(np.percentile(xs, q)) if xs else float("nan")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", choices=["vqa", "report"], required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--model-base", required=True)
    p.add_argument("--question-file", required=True)
    p.add_argument("--image-folder", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--temperature", type=float, default=0.2)
    args = p.parse_args()
    max_new_tokens = 64 if args.task == "vqa" else 1024
    os.makedirs(args.out_dir, exist_ok=True)

    set_seed(0)
    random.seed(42)
    disable_torch_init()
    t0 = time.perf_counter()
    model_name = get_model_name_from_path(args.model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(
        args.model_path, args.model_base, model_name, device="cuda:0"
    )
    torch.cuda.synchronize()
    load_s = time.perf_counter() - t0
    weights_gb = torch.cuda.memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()

    questions = [json.loads(l) for l in open(args.question_file)]
    if args.limit:
        questions = questions[: args.limit]

    rows = []
    for i, q in enumerate(questions):
        t_start = time.perf_counter()
        if args.task == "vqa":
            qs = q["question"].replace(DEFAULT_IMAGE_TOKEN, "").strip()
        else:
            qs = random.choice(REPORT_PROMPTS).replace(DEFAULT_IMAGE_TOKEN, "").strip()
        prompt_text = qs
        conv = conv_templates["vicuna_v1"].copy()
        conv.append_message(conv.roles[0], DEFAULT_IMAGE_TOKEN + "\n" + qs)
        conv.append_message(conv.roles[1], None)
        input_ids = tokenizer_image_token(
            conv.get_prompt(), tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        ).unsqueeze(0).to("cuda:0")
        image = Image.open(os.path.join(args.image_folder, q["image"])).convert("RGB")
        image_tensor = process_images([image], image_processor, model.config)[0]
        images = image_tensor.unsqueeze(0).half().to("cuda:0")
        torch.cuda.synchronize()
        t_gen = time.perf_counter()

        streamer = TimingStreamer()
        with torch.inference_mode():
            output_ids = model.generate(
                input_ids,
                images=images,
                do_sample=args.temperature > 0,
                temperature=args.temperature,
                top_p=None,
                num_beams=1,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                pad_token_id=tokenizer.eos_token_id,  # HF's default anyway; silences a per-call warning
                streamer=streamer,
            )
        torch.cuda.synchronize()
        t_end = time.perf_counter()

        # LLaVA passes inputs_embeds to HF generate, so output_ids holds only new tokens.
        n_new = int(output_ids.shape[1])
        st = streamer.times
        answer = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
        rows.append({
            "question_id": q["question_id"],
            "image": q["image"],
            "prompt": prompt_text,
            "answer": answer,
            "gt_answer": q.get("answer", q.get("gt_report")),
            "answer_type": q.get("answer_type"),
            "warmup": i < args.warmup,
            "new_tokens": n_new,
            "prompt_tokens": int(input_ids.shape[1]) - 1 + 576,  # 576 CLIP-336 patch tokens replace <image>
            "preprocess_s": t_gen - t_start,
            "vision_encode_s": st[0] - t_gen,  # CLIP encode + projector + embedding merge
            "ttft_s": st[1] - t_gen if len(st) > 1 else float("nan"),
            "decode_s": t_end - st[1] if len(st) > 1 else 0.0,
            "generate_s": t_end - t_gen,
            "end_to_end_s": t_end - t_start,
        })
        if i < 3 or (i + 1) % 100 == 0:
            r = rows[-1]
            print(f"[{i+1}/{len(questions)}] {r['generate_s']:.3f}s {n_new} tok | {answer[:90]!r}", flush=True)

    timed = [r for r in rows if not r["warmup"]]
    gen = [r["generate_s"] for r in timed]
    e2e = [r["end_to_end_s"] for r in timed]
    toks = [r["new_tokens"] for r in timed]
    dec_rates = [(r["new_tokens"] - 1) / r["decode_s"] for r in timed if r["new_tokens"] > 1 and r["decode_s"] > 0]
    summary = {
        "tag": args.tag,
        "task": args.task,
        "model_path": args.model_path,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "max_new_tokens": max_new_tokens,
        "n_samples_total": len(rows),
        "n_samples_timed": len(timed),
        "model_load_s": load_s,
        "weights_gpu_mem_gb": weights_gb,
        "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
        "peak_gpu_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
        "generate_s_mean": float(np.mean(gen)),
        "generate_s_median": pct(gen, 50),
        "generate_s_p90": pct(gen, 90),
        "generate_s_max": float(np.max(gen)),
        "end_to_end_s_mean": float(np.mean(e2e)),
        "ttft_s_mean": float(np.nanmean([r["ttft_s"] for r in timed])),
        "vision_encode_s_mean": float(np.mean([r["vision_encode_s"] for r in timed])),
        "new_tokens_mean": float(np.mean(toks)),
        "new_tokens_max": int(np.max(toks)),
        "decode_tok_per_s_mean": float(np.mean(dec_rates)) if dec_rates else float("nan"),
        "total_inference_s": float(np.sum([r["end_to_end_s"] for r in rows])),
        "throughput_samples_per_s": len(timed) / float(np.sum(e2e)),
    }
    if args.task == "vqa":
        closed = [r for r in timed if r["answer_type"] == "CLOSED"]
        if closed:
            summary["sanity_closed_acc"] = float(np.mean(
                [r["gt_answer"].strip().lower() in re.findall(r"[a-z]+", r["answer"].lower())[:3] for r in closed]))

    with open(os.path.join(args.out_dir, f"{args.tag}_answers.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(args.out_dir, f"{args.tag}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
