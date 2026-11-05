"""Print the inference and training benchmark results as markdown tables.

usage: summarize.py [results_dir]   (default: results/full)
"""
import csv
import glob
import json
import os
import statistics
import sys

out_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results", "full")


def table(cols, rows):
    print("| " + " | ".join(c for c, _ in cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(str(f(r)) for _, f in cols) + " |")


infer = [json.load(open(f)) for f in sorted(glob.glob(f"{out_dir}/*_summary.json"))]
if infer:
    print("Inference, per scan (first post-contrast phase, fold-0 validation patients)\n")
    table([
        ("Setting", lambda s: s["tag"]),
        ("Models", lambda s: len(s["folds"])),
        ("Mirroring", lambda s: "yes" if s["mirroring"] else "no"),
        ("Scans", lambda s: s["n_cases"]),
        ("Load (s)", lambda s: f"{s['model_load_s']:.1f}"),
        ("Peak GPU mem (GB)", lambda s: f"{s['peak_gpu_mem_gb']:.1f}"),
        ("Read (s)", lambda s: f"{s['read_s_mean']:.2f}"),
        ("Preprocess (s)", lambda s: f"{s['preprocess_s_mean']:.2f}"),
        ("Predict (s)", lambda s: f"{s['predict_s_mean']:.2f}"),
        ("Export (s)", lambda s: f"{s['export_s_mean']:.2f}"),
        ("Total mean (s)", lambda s: f"{s['total_s_mean']:.2f}"),
        ("Total p90 (s)", lambda s: f"{s['total_s_p90']:.2f}"),
        ("Dice", lambda s: f"{s['dice_mean']:.3f}"),
        ("Authors' Dice", lambda s: f"{s['dice_reported_mean']:.3f}"),
    ], infer)
    print()


def gpu_util(tag):
    f = f"{out_dir}/train_{tag}_gpu_util.csv"
    if not os.path.isfile(f):
        return float("nan")
    vals = [float(r[2].strip().rstrip("%").strip()) for r in csv.reader(open(f)) if len(r) >= 3]
    return statistics.mean(vals[len(vals) // 4:]) if vals else float("nan")  # skip the start-up quarter


train = []
for f in sorted(glob.glob(f"{out_dir}/train_*.json")):
    s = json.load(open(f))
    s["tag"] = os.path.basename(f)[len("train_"):-len(".json")]
    steady = s["train_iter_s"][1:] or s["train_iter_s"]   # epoch 1 includes torch.compile
    s["iter_s"] = statistics.median(steady)
    s["epoch_s_steady"] = statistics.median(s["epoch_s"][1:] or s["epoch_s"])
    s["patches_per_s"] = s["global_batch_size"] / s["iter_s"]
    s["util"] = gpu_util(s["tag"])
    train.append(s)
if train:
    base = next((s for s in train if s["gpus"] == 1 and s["global_batch_size"] == 2), train[0])
    order = sorted(train, key=lambda s: (s["gpus"], s["global_batch_size"]))
    print("Training, 3d_fullres 128^3 patches (median of epochs 2 onwards; epoch = 250 training + 50 validation batches)\n")
    table([
        ("Run", lambda s: s["tag"]),
        ("GPUs", lambda s: s["gpus"]),
        ("Batch (total)", lambda s: s["global_batch_size"]),
        ("Batch per GPU", lambda s: s["batch_size_per_gpu"]),
        ("Epoch (s)", lambda s: f"{s['epoch_s_steady']:.1f}"),
        ("Train step (s)", lambda s: f"{s['iter_s']:.3f}"),
        ("Patches/s", lambda s: f"{s['patches_per_s']:.1f}"),
        ("Speed-up vs 1 GPU, batch 2", lambda s: f"{s['patches_per_s'] / base['patches_per_s']:.2f}x"),
        ("Peak GPU mem (GB)", lambda s: f"{s['peak_mem_gb_per_gpu']:.1f}"),
        ("Mean GPU util (%)", lambda s: f"{s['util']:.0f}"),
    ], order)
    full = 1000 * base["epoch_s_steady"] / 3600
    print(f"\nA full 1,000-epoch training of one fold at the authors' setting would take about {full:.1f} h on one GPU.")

conc = f"{out_dir}/concurrency/concurrency.csv"
if os.path.isfile(conc):
    rows = list(csv.DictReader(open(conc)))
    one = float(rows[0]["scans_per_hour"])
    print("\nInference throughput on ONE GPU with K copies running at once (one model, mirroring, model load included)\n")
    table([
        ("Copies on the GPU", lambda r: r["K"]),
        ("Scans", lambda r: r["scans"]),
        ("Wall time (s)", lambda r: r["wall_s"]),
        ("Scans per hour", lambda r: r["scans_per_hour"]),
        ("vs 1 copy", lambda r: f"{float(r['scans_per_hour']) / one:.2f}x"),
    ], rows)
