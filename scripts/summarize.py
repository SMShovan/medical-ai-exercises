"""Collect *_summary.json files into one markdown table."""
import glob
import json
import os
import sys

out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "full")
rows = [json.load(open(f)) for f in sorted(glob.glob(f"{out_dir}/*_summary.json"))]
cols = [
    ("Run", lambda s: s["tag"]),
    ("Task", lambda s: s["task"]),
    ("Samples", lambda s: s["n_samples_total"]),
    ("Load (s)", lambda s: f"{s['model_load_s']:.1f}"),
    ("GPU mem peak (GB)", lambda s: f"{s['peak_gpu_mem_gb']:.1f}"),
    ("Per-sample mean (s)", lambda s: f"{s['end_to_end_s_mean']:.3f}"),
    ("Median (s)", lambda s: f"{s['generate_s_median']:.3f}"),
    ("p90 (s)", lambda s: f"{s['generate_s_p90']:.3f}"),
    ("Max (s)", lambda s: f"{s['generate_s_max']:.2f}"),
    ("TTFT (s)", lambda s: f"{s['ttft_s_mean']:.3f}"),
    ("Tokens out (mean)", lambda s: f"{s['new_tokens_mean']:.0f}"),
    ("Decode tok/s", lambda s: f"{s['decode_tok_per_s_mean']:.0f}"),
    ("Whole test set (min)", lambda s: f"{s['total_inference_s'] / 60:.1f}"),
]
print("| " + " | ".join(c for c, _ in cols) + " |")
print("|" + "---|" * len(cols))
for s in rows:
    print("| " + " | ".join(str(f(s)) for _, f in cols) + " |")
for s in rows:
    if "sanity_closed_acc" in s:
        print(f"{s['tag']}: closed-ended accuracy (rough check) = {100 * s['sanity_closed_acc']:.1f}%")
