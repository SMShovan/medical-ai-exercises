"""Set up an nnU-Net training dataset from MAMA-MIA for the training-speed benchmark.

- nnUNet_raw/Dataset105_full_image: links to the first post-contrast image and expert segmentation
  of N patients (no copies). The name matches the authors' plans so their settings apply unchanged.
- nnUNet_preprocessed/Dataset105_full_image: the authors' plans (1 mm spacing, 128^3 patches, batch 2)
  plus copies with batch size 4 and 8. All share one preprocessed-data folder.

The training speed of nnU-Net does not depend on the number of training cases (an epoch is a fixed
250 batches), so a few hundred patients are enough; preprocessing all 6,941 phase images would take hours.
"""
import argparse
import copy
import json
import os
import random
import shutil

DATASET = "Dataset105_full_image"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-cases", type=int, default=300)
    p.add_argument("--batch-sizes", default="4,8", help="extra plans with these batch sizes")
    a = p.parse_args()
    data_dir = os.environ["MAMAMIA_DIR"]
    raw = os.path.join(os.environ["nnUNet_raw"], DATASET)
    pre = os.path.join(os.environ["nnUNet_preprocessed"], DATASET)
    model = os.path.join(os.environ["nnUNet_results"], "full_image_dce_mri_tumor_segmentation")

    patients = sorted(d for d in os.listdir(os.path.join(data_dir, "images"))
                      if os.path.isfile(os.path.join(data_dir, "images", d, f"{d}_0001.nii.gz")))
    random.seed(0)
    chosen = sorted(random.sample(patients, min(a.n_cases, len(patients))))

    for sub in ("imagesTr", "labelsTr"):
        shutil.rmtree(os.path.join(raw, sub), ignore_errors=True)
        os.makedirs(os.path.join(raw, sub))
    for pid in chosen:
        os.symlink(os.path.join(data_dir, "images", pid, f"{pid}_0001.nii.gz"),
                   os.path.join(raw, "imagesTr", f"{pid}_0000.nii.gz"))
        os.symlink(os.path.join(data_dir, "segmentations", "expert", f"{pid}.nii.gz"),
                   os.path.join(raw, "labelsTr", f"{pid}.nii.gz"))
    dataset_json = {"channel_names": {"0": "T1"}, "labels": {"background": 0, "tumor": 1},
                    "numTraining": len(chosen), "file_ending": ".nii.gz"}
    with open(os.path.join(raw, "dataset.json"), "w") as f:
        json.dump(dataset_json, f, indent=2)

    os.makedirs(pre, exist_ok=True)
    shutil.copy(os.path.join(raw, "dataset.json"), pre)
    shutil.copy(os.path.join(model, "dataset_fingerprint.json"), pre)
    plans = json.load(open(os.path.join(model, "plans.json")))
    with open(os.path.join(pre, "nnUNetPlans.json"), "w") as f:
        json.dump(plans, f, indent=2)
    for bs in (int(b) for b in a.batch_sizes.split(",") if b):
        q = copy.deepcopy(plans)
        q["plans_name"] = f"nnUNetPlans_bs{bs}"
        q["configurations"]["3d_fullres"]["batch_size"] = bs
        # keep data_identifier, so every variant reads the same preprocessed files
        with open(os.path.join(pre, f"nnUNetPlans_bs{bs}.json"), "w") as f:
            json.dump(q, f, indent=2)
    print(f"{DATASET}: {len(chosen)} patients linked; plans: nnUNetPlans + batch sizes {a.batch_sizes}")


if __name__ == "__main__":
    main()
