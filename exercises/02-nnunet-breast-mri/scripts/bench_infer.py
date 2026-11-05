"""Time the MAMA-MIA pretrained nnU-Net on breast DCE-MRI, one scan at a time, on one GPU.

For each scan it times the four stages nnU-Net goes through and checks the result against the
expert segmentation:
  read        load the NIfTI file from disk
  preprocess  crop, resample to 1 mm, z-score (CPU)
  predict     sliding-window inference with 128^3 patches, optionally mirrored (GPU)
  export      resample the prediction back to the original grid (CPU)

The model is run on its own fold's validation patients (listed in the authors' per-fold metrics
file), so the Dice is measured on scans that fold never trained on.
"""
import argparse
import csv
import json
import os
import time

import numpy as np
import torch
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.inference.export_prediction import convert_predicted_logits_to_segmentation_with_correct_shape
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor


def sync():
    torch.cuda.synchronize()
    return time.perf_counter()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--model-dir", required=True)
    p.add_argument("--folds", default="0", help="comma-separated folds to ensemble, e.g. 0 or 0,1,2,3,4")
    p.add_argument("--cases-from-fold", type=int, default=0, help="use this fold's validation patients")
    p.add_argument("--no-mirroring", action="store_true", help="disable test-time mirroring (8x fewer passes)")
    p.add_argument("--phase", type=int, default=1, help="DCE phase to segment; 1 = first post-contrast")
    p.add_argument("--data-dir", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--offset", type=int, default=0, help="skip this many patients first (to split work between runs)")
    p.add_argument("--warmup", type=int, default=2)
    a = p.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    folds = tuple(int(f) for f in a.folds.split(","))

    with open(os.path.join(a.model_dir, f"fold_{a.cases_from_fold}", f"nnunet_metrics_fold_{a.cases_from_fold}.csv")) as f:
        cases = [(r["patient_id"], float(r["dice"])) for r in csv.DictReader(f)]
    cases = cases[a.offset:]
    if a.limit:
        cases = cases[: a.limit]

    t0 = time.perf_counter()
    predictor = nnUNetPredictor(tile_step_size=0.5, use_gaussian=True, use_mirroring=not a.no_mirroring,
                                perform_everything_on_device=True, device=torch.device("cuda", 0),
                                verbose=False, verbose_preprocessing=False, allow_tqdm=False)
    predictor.initialize_from_trained_model_folder(a.model_dir, use_folds=folds, checkpoint_name="checkpoint_final.pth")
    load_s = sync() - t0
    weights_gb = torch.cuda.memory_allocated() / 1e9
    torch.cuda.reset_peak_memory_stats()

    pm, cm = predictor.plans_manager, predictor.configuration_manager
    preprocessor = cm.preprocessor_class(verbose=False)
    io = SimpleITKIO()
    rows = []
    for i, (pid, reported_dice) in enumerate(cases):
        img_file = os.path.join(a.data_dir, "images", pid, f"{pid}_{a.phase:04d}.nii.gz")
        gt_file = os.path.join(a.data_dir, "segmentations", "expert", f"{pid}.nii.gz")
        if not os.path.isfile(img_file):
            print(f"skip {pid}: no phase {a.phase} image", flush=True)
            continue
        t_start = time.perf_counter()
        img, props = io.read_images([img_file])
        t_read = time.perf_counter()
        data, _, props = preprocessor.run_case_npy(img, None, props, pm, cm, predictor.dataset_json)
        data = torch.from_numpy(data)
        t_pre = sync()
        logits = predictor.predict_logits_from_preprocessed_data(data)
        t_pred = sync()
        seg = convert_predicted_logits_to_segmentation_with_correct_shape(
            logits, pm, cm, predictor.label_manager, props, return_probabilities=False)
        t_exp = time.perf_counter()

        gt, _ = io.read_seg(gt_file)
        gt, pr = gt[0] > 0, seg > 0
        denom = gt.sum() + pr.sum()
        dice = float(2 * np.logical_and(gt, pr).sum() / denom) if denom else 1.0
        padded = [max(s, ps) for s, ps in zip(data.shape[1:], cm.patch_size)]
        rows.append({
            "patient_id": pid,
            "warmup": i < a.warmup,
            "original_shape": list(img.shape[1:]),
            "spacing_mm": [float(s) for s in props["spacing"]],
            "preprocessed_shape": list(data.shape[1:]),
            "n_patches": len(predictor._internal_get_sliding_window_slicers(padded)),
            "read_s": t_read - t_start,
            "preprocess_s": t_pre - t_read,
            "predict_s": t_pred - t_pre,
            "export_s": t_exp - t_pred,
            "total_s": t_exp - t_start,
            "dice": dice,
            "dice_reported_by_authors": reported_dice,
        })
        if i < 3 or (i + 1) % 25 == 0:
            r = rows[-1]
            print(f"[{i+1}/{len(cases)}] {pid} {r['total_s']:.2f}s (predict {r['predict_s']:.2f}s, "
                  f"{r['n_patches']} patches) dice {dice:.3f} (authors {reported_dice:.3f})", flush=True)

    timed = [r for r in rows if not r["warmup"]]
    def mean(k):
        return float(np.mean([r[k] for r in timed]))
    summary = {
        "tag": a.tag,
        "folds": list(folds),
        "mirroring": not a.no_mirroring,
        "phase": a.phase,
        "gpu": torch.cuda.get_device_name(0),
        "torch": torch.__version__,
        "patch_size": list(cm.patch_size),
        "n_cases": len(rows),
        "n_timed": len(timed),
        "model_load_s": load_s,
        "weights_gpu_mem_gb": weights_gb,
        "peak_gpu_mem_gb": torch.cuda.max_memory_allocated() / 1e9,
        "peak_gpu_reserved_gb": torch.cuda.max_memory_reserved() / 1e9,
        "read_s_mean": mean("read_s"),
        "preprocess_s_mean": mean("preprocess_s"),
        "predict_s_mean": mean("predict_s"),
        "export_s_mean": mean("export_s"),
        "total_s_mean": mean("total_s"),
        "total_s_median": float(np.median([r["total_s"] for r in timed])),
        "total_s_p90": float(np.percentile([r["total_s"] for r in timed], 90)),
        "n_patches_mean": mean("n_patches"),
        "dice_mean": float(np.mean([r["dice"] for r in rows])),
        "dice_reported_mean": float(np.mean([r["dice_reported_by_authors"] for r in rows])),
        "total_inference_s": float(np.sum([r["total_s"] for r in rows])),
    }
    with open(os.path.join(a.out_dir, f"{a.tag}_cases.jsonl"), "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(os.path.join(a.out_dir, f"{a.tag}_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
