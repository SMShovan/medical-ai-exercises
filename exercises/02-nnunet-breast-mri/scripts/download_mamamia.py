"""Download the MAMA-MIA dataset and the authors' pretrained nnU-Net weights from Synapse.

Needs a Synapse account that has accepted the dataset's terms, and a personal access token
saved in ~/.synapseConfig (see the README). The token is read by synapseclient itself.

usage: download_mamamia.py --what weights|data
  weights  the pretrained nnU-Net (1.2 GB zip), unzipped into $nnUNet_results
  data     everything else in the project: images, segmentations, clinical files (~100 GB)
Files already downloaded with a matching checksum are skipped, so both are safe to re-run.
"""
import argparse
import glob
import os
import zipfile

import synapseclient
import synapseutils

PROJECT_ID = "syn60868042"   # MAMA-MIA
WEIGHTS_ID = "syn61247992"   # its nnUNet_pretrained_weights folder (Dataset101, 3d_fullres)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--what", choices=["weights", "data"], required=True)
    a = p.parse_args()

    data_dir = os.environ["MAMAMIA_DIR"]
    results_dir = os.environ["nnUNet_results"]
    weights_dir = os.path.join(data_dir, "nnUNet_pretrained_weights")
    syn = synapseclient.Synapse(skip_checks=True)
    syn.max_threads = 8     # parallel transfers per folder; the default (36) is more than we need
    syn.login(silent=True)  # reads the token from ~/.synapseConfig
    print("logged in to Synapse as", syn.getUserProfile()["userName"])

    if a.what == "weights":
        synapseutils.syncFromSynapse(syn, WEIGHTS_ID, path=weights_dir)
        for zf in glob.glob(os.path.join(weights_dir, "*.zip")):
            with zipfile.ZipFile(zf) as z:
                z.extractall(results_dir)
            print("unzipped", zf, "into", results_dir)
    else:
        for child in syn.getChildren(PROJECT_ID):
            if child["id"] == WEIGHTS_ID:
                continue  # downloaded by --what weights
            if child["type"].endswith("Folder"):
                n = sync_folder(syn, child["id"], os.path.join(data_dir, child["name"]))
            else:
                n = len(synapseutils.syncFromSynapse(syn, child["id"], path=data_dir))
            print(f"{child['name']}: {n} files", flush=True)
    print("DOWNLOAD_DONE")


def sync_folder(syn, folder_id, path):
    """Download a folder one subfolder at a time.

    Syncing the whole project at once starts all 1,506 case folders together, which used more than
    the job's memory limit (1.9 GB per CPU on our cluster) and got the job killed.
    """
    subfolders = [c for c in syn.getChildren(folder_id) if c["type"].endswith("Folder")]
    if not subfolders:
        return len(synapseutils.syncFromSynapse(syn, folder_id, path=path))
    total = 0
    for i, sub in enumerate(subfolders, 1):
        total += sync_folder(syn, sub["id"], os.path.join(path, sub["name"]))
        if i % 100 == 0:
            print(f"  {os.path.basename(path)}: {i}/{len(subfolders)} folders", flush=True)
    return total


if __name__ == "__main__":
    main()
