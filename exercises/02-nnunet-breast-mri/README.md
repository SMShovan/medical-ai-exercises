# Exercise 2: nnU-Net on breast cancer MRI

Part of [medical-ai-exercises](../../README.md).

[nnU-Net](https://www.nature.com/articles/s41592-020-01008-z) (Isensee et al., *Nature Methods* 2021, from the German Cancer Research Center in Heidelberg) is the standard baseline for 3D medical image segmentation. It configures itself for each new dataset. This exercise measures how it performs on our B200 GPUs on public breast cancer MRI:

- **Inference:** the time, GPU memory and throughput to outline the tumour in one scan, using the pretrained nnU-Net released with the MAMA-MIA dataset.
- **Training:** the time per epoch, memory and scaling on 1, 2 and 4 GPUs, using the same network and settings.

The focus is speed and resources. Dice, the overlap with the expert's tumour outline, is reported only to check that we run the model the way its authors did.

**Data:** [MAMA-MIA](https://doi.org/10.1038/s41597-025-04707-4) (*Scientific Data* 2025), 1,506 pre-treatment breast DCE-MRI scans from four TCIA collections (Duke, I-SPY1, I-SPY2, NACT-Pilot), with expert tumour segmentations and pCR labels. This is the public dataset for the breast MRI pCR project.

## Results

hgx-01, NVIDIA B200 (183 GB), nnU-Net 2.8.1, PyTorch 2.8 (CUDA 12.8), 5 November 2026. The raw files are in [`results/reference/`](results/reference). The model is nnU-Net `3d_fullres` (a 3D U-Net with 31.2M parameters, 128 × 128 × 128 patches at 1 mm), applied to the first post-contrast phase.

### Inference, one scan at a time

Each setting ran on its own GPU over the same 100 patients. These are the fold-0 validation patients, so the fold-0 model never trained on them.

| Setting | Models | Mirroring | Read | Preprocess (CPU) | Predict (GPU) | Export (CPU) | **Total per scan** | Peak GPU memory | Dice | Authors' Dice |
|---|---|---|---|---|---|---|---|---|---|---|
| One model, fast | 1 | no | 0.10 s | 4.72 s | 0.81 s | 1.59 s | **7.2 s** | 1.6 GB | 0.646 | 0.684 |
| One model (authors' setting) | 1 | yes | 0.10 s | 4.71 s | 6.17 s | 1.60 s | **12.6 s** | 1.6 GB | **0.695** | 0.684 |
| Five-model ensemble (nnU-Net default) | 5 | yes | 0.10 s | 4.67 s | 30.91 s | 1.58 s | **37.3 s** | 1.6 GB | 0.777* | 0.684 |

- **We reproduce the authors' model.** With mirroring, our Dice of 0.695 matches the authors' published 0.684 for the same patients. Mirroring is nnU-Net's default and repeats each prediction on 8 flipped copies of the scan.
- **On a single scan, the GPU is mostly idle.** Resampling the scan to 1 mm and back runs on the CPU and takes 6.3 s per scan. The GPU part is 0.8 s without mirroring and 6.2 s with it.
- **Memory is not a constraint.** Peak GPU memory is 1.6 GB, under 1% of a B200.
- \*The ensemble's Dice is optimistic, because four of its five models trained on these patients.

### Inference throughput on one GPU

Several copies of the one-model setting with mirroring ran on **the same** B200 at once, each on its own 24 patients. Model loading is included.

| Copies on one GPU | Scans | Wall time | **Scans per hour** | vs 1 copy |
|---|---|---|---|---|
| 1 | 24 | 405 s | 213 | 1.00x |
| 2 | 48 | 514 s | 336 | 1.58x |
| 4 | 96 | 719 s | 481 | 2.26x |
| 8 | 192 | 926 s | **746** | 3.50x |

Because a single run leaves the GPU idle while the CPU resamples, running 8 copies gives 3.5x the throughput from the same GPU. Throughput was still rising at 8 copies. For clinical-scale inference, the CPU cores per GPU and the number of parallel runs matter more than GPU memory.

### Training speed

These runs use the authors' plans unchanged (`3d_fullres`, 128³ patches, batch 2, 1 mm), on 300 patients. Each epoch is 250 training plus 50 validation batches, whatever the dataset size. Each run trained for 4 epochs; the table shows the median of epochs 2–4, because epoch 1 includes `torch.compile`. "Batch per GPU" is 1–8 patches; multi-GPU runs use PyTorch DDP.

| GPUs | Batch (total) | Batch per GPU | Epoch | Train step | Patches/s | **Speed-up** | Peak memory per GPU | GPU utilisation |
|---|---|---|---|---|---|---|---|---|
| 1 | 2 | 2 | 9.1 s | 34 ms | 58.7 | **1.00x** (authors' setup) | 4.7 GB | 57% |
| 1 | 4 | 4 | 15.5 s | 58 ms | 68.5 | 1.17x | 9.2 GB | 77% |
| 1 | 8 | 8 | 29.5 s | 111 ms | 72.0 | 1.23x | 18.3 GB | 93% |
| 2 | 2 | 1 | 5.9 s | 22 ms | 90.5 | 1.54x | 2.5 GB | 41% |
| 2 | 4 | 2 | 9.5 s | 36 ms | 111.6 | **1.90x** | 5.0 GB | 54% |
| 4 | 8 | 2 | 11.7 s | 44 ms | 183.6 | **3.13x** | 5.0 GB | 66% |

- **How long a full training takes.** One fold at the authors' setting takes about **2.5 h for 1,000 epochs** (nnU-Net's default) on one B200. The released model was trained with `nnUNetTrainer_4000epochs`, so reproducing it takes about 10 h per fold. That is longer than our 6-hour job limit, so it needs nnU-Net's resume option (`--c`) across jobs.
- **More GPUs scale well.** With 2 patches per GPU, 2 GPUs reach 1.90x (95% efficiency) and 4 GPUs reach 3.13x (78%).
- **A bigger batch on one GPU barely helps.** Batch 8 gives only 1.23x: the GPU is already 93% busy. It uses 18 GB, so the B200's memory is not what limits nnU-Net training.
- **For choosing GPUs,** nnU-Net needs less than 20 GB even at batch 8. Many smaller GPUs would serve it as well as a few large ones; the large memory of a B200 matters for big models such as the vision-language models in exercise 1.

## What you need

- Linux with NVIDIA GPUs and a driver for CUDA 12.8. The inference benchmark uses 3 GPUs and the training benchmark 4. Each needs only a few GB.
- About **110 GB** of disk: 89 GB of images, 8 GB preprocessed, 5 GB Python environment, 1.2 GB model.
- A free [Synapse](https://www.synapse.org) account. On the [MAMA-MIA page](https://www.synapse.org/Synapse:syn60868042), accept the terms, then create a personal access token with View and Download permission.
- `git` and [`uv`](https://docs.astral.sh/uv/). Slurm is optional.

## Run it

Run every command from this folder:

```bash
cd medical-ai-exercises/exercises/02-nnunet-breast-mri
```

**1. Build the environment** (CPU, about a minute):

```bash
sbatch scripts/setup_env.sbatch
```

**2. Save your Synapse token** in `~/.synapseConfig`. The download reads it from there.

```
[authentication]
authtoken = YOUR_TOKEN
```

Then run `chmod 600 ~/.synapseConfig`.

**3. Download the model, then the data.** The model is 1.2 GB; the data is about 89 GB and takes roughly 40 minutes. Re-running either one skips files that are already there.

```bash
WHAT=weights sbatch scripts/download_data.sbatch
WHAT=data sbatch scripts/download_data.sbatch
```

**4. Inference benchmarks.** `LIMIT=6` gives a quick test.

```bash
sbatch scripts/bench_infer.sbatch                # three settings, 100 patients, about 1 h
sbatch scripts/bench_infer_concurrency.sbatch    # 1-8 copies on one GPU, about 45 min
```

**5. Training benchmark.** Preprocessing takes a few minutes and the training runs about 15 minutes.

```bash
sbatch scripts/preprocess.sbatch
sbatch scripts/bench_train.sbatch
```

**6. Results.**

```bash
.venv/bin/python scripts/summarize.py                     # your run (results/full)
.venv/bin/python scripts/summarize.py results/reference   # ours
```

> **On our cluster, submit one job at a time and wait for it to finish.** On hgx-01, when one of your jobs ends, every other job you have running there is killed (shown as `FAILED 0:9`). This includes interactive `srun` sessions. The scripts keep their parallel work inside a single job for this reason.

## What the scripts do

| File | What it does |
|---|---|
| `scripts/setup_env.sbatch` | Creates `.venv` with PyTorch 2.8 (CUDA 12.8), nnU-Net 2.8.1 and the Synapse client, and installs `nnUNetTrainerBenchmark` into nnU-Net. |
| `scripts/env.sh` | Sets the folders nnU-Net uses (`nnUNet_raw`, `nnUNet_preprocessed`, `nnUNet_results`), all inside this exercise. |
| `scripts/download_mamamia.py`, `download_data.sbatch` | Download from Synapse one folder at a time, which keeps memory low, and unzip the model. |
| `scripts/bench_infer.py`, `bench_infer.sbatch` | Time read, preprocess, predict and export for each scan, and compute Dice against the expert outline. Three settings run in parallel, one GPU each. |
| `scripts/bench_infer_concurrency.sbatch` | Run 1, 2, 4 and 8 copies of the inference on one GPU and measure scans per hour. |
| `scripts/prepare_dataset.py`, `preprocess.sbatch` | Link 300 patients into nnU-Net's training layout, copy the authors' plans (plus batch-4 and batch-8 copies), and run `nnUNetv2_preprocess`. |
| `scripts/nnUNetTrainerBenchmark.py` | A trainer that runs a few epochs, skips checkpoints and the final validation, and writes epoch times and peak memory to JSON. |
| `scripts/bench_train.sbatch` | Run each GPU-count and batch-size combination in turn, and log GPU utilisation every 2 s. |
| `scripts/summarize.py` | Print the tables above from the result files. |

## Notes

- **Which image is segmented.** Each MAMA-MIA case has 3–6 DCE phases. The experts outlined the tumour on the first post-contrast phase (`_0001`), and our Dice matches the authors' only on that phase. The released model was trained on all phases as separate images (6,941 in total), as its `dataset.json` shows.
- **The training benchmark does not produce a usable model.** It runs a few epochs to measure speed and keeps no checkpoints.
- **Speed does not depend on dataset size,** because an nnU-Net epoch is a fixed number of batches. That's why 300 patients are enough for the benchmark.
- **GPU utilisation** is the mean of `nvidia-smi` samples every 2 s, leaving out the first quarter of each run (start-up and compilation). It includes the validation part of each epoch.

## Credits

- nnU-Net: Isensee et al., *nnU-Net: a self-configuring method for deep learning-based biomedical image segmentation*, Nature Methods 18, 203–211 (2021). Code at [MIC-DKFZ/nnUNet](https://github.com/MIC-DKFZ/nnUNet) (Apache-2.0).
- MAMA-MIA: Garrucho et al., *A large-scale multicenter breast cancer DCE-MRI benchmark dataset with expert segmentations*, Scientific Data 12, 453 (2025). The data and pretrained model are under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/); see [the MAMA-MIA repository](https://github.com/LidiaGarrucho/MAMA-MIA) for the image collections' own licences. This repository contains no images.
