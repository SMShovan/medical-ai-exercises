# MMedPO inference benchmark

Scripts to measure how fast the released [MMedPO](https://arxiv.org/abs/2412.06141) models (ICML 2025) answer medical questions and write chest X-ray reports, and how much GPU memory they need. They download everything, run the four released models on public test sets, and save every answer with its timings.

This covers **inference only**. Nothing is trained.

## Results from our run

These are from Slurm job 266 on our cluster node hgx-01, with one NVIDIA B200 (183 GB) per run, fp16 and batch size 1. The raw files are in [`results/reference/`](results/reference).

| Run | Test set | Items | Per item | Whole test set | Peak GPU memory |
|---|---|---|---|---|---|
| vqa-rad | VQA-RAD questions | 451 | 0.19 s | 1.7 min | 16.6 GB |
| slake | SLAKE questions (English) | 1,061 | 0.12 s | 2.4 min | 16.6 GB |
| iuxray | IU-Xray reports | 590 | 0.73 s | 7.5 min | 16.6 GB |
| mimic-ckpt_on_iuxray | IU-Xray images, MIMIC-CXR model | 590 | 1.00 s | 10.0 min | 16.8 GB |

Loading the model takes 36–45 s per run. The four runs go in parallel, so the whole job took about 11 minutes.

## What you need

- A Linux machine with NVIDIA GPUs and a driver that supports CUDA 12.8. The full run uses **4 GPUs**, each needing about 17 GB of memory. We have only tested NVIDIA B200s. Other GPUs that PyTorch 2.8 supports should work, but your timings will differ from ours.
- About **30 GB** of disk for the model weights, test sets and Python packages.
- Internet access during setup (Hugging Face and PyPI). No Hugging Face login is needed.
- `git` and [`uv`](https://docs.astral.sh/uv/). To install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`
- Slurm is optional. The `.sbatch` files are ordinary bash scripts.

## Run it

**1. Clone the repository, including the authors' MMedPO code:**

```bash
git clone --recursive https://github.com/SMShovan/mmedpo-inference-benchmark.git
cd mmedpo-inference-benchmark
```

Run every command below from this folder.

**2. Set up once.** This builds the Python environment, downloads the models and test sets, and prepares the test files. It uses no GPU. Most of the time goes on downloading about 20 GB.

```bash
sbatch scripts/setup_env.sbatch
```

Wait until `logs/mmedpo-setup-<jobid>.out` ends with `SETUP_DONE`. You can check progress with `squeue -u $USER`.

**3. Run the benchmark.** This starts four runs at once, one per GPU.

```bash
sbatch scripts/run_bench.sbatch
```

The job log `logs/mmedpo-bench-<jobid>.out` ends with `BENCH_DONE`. Each run also writes its own log, `logs/bench_<run>.log`.

To try it quickly first, answer only the first 20 items of each test set:

```bash
LIMIT=20 sbatch scripts/run_bench.sbatch
```

**4. Read the results.** This prints a table for your run. Point it at `results/reference` to see ours.

```bash
python3 scripts/summarize.py
python3 scripts/summarize.py results/reference
```

Each run writes two files to `results/full/`:

- `<run>_answers.jsonl` has one line per test item: the question, the model's answer, the answer key, and the timings.
- `<run>_summary.json` has the averages, the peak GPU memory and the model load time.

## Running on a different cluster or machine

- **Another Slurm partition.** Options on the command line override the ones in the script, for example `sbatch -p gpu scripts/run_bench.sbatch`.
- **Slurm that hands out GPUs** (most clusters do). Our cluster doesn't, so the script picks idle GPUs itself. Elsewhere, ask for them explicitly: `sbatch -p gpu --gres=gpu:4 scripts/run_bench.sbatch`. On clusters that hide the GPUs a job wasn't given, the script then sees and uses only those four.
- **No Slurm.** Run the same files with bash, `bash scripts/setup_env.sbatch`, then `bash scripts/run_bench.sbatch`.
- **Fewer than 4 GPUs.** Run one test set at a time instead:

  ```bash
  HF_HOME=$PWD/hf_cache HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 .venv/bin/python scripts/bench_inference.py \
    --tag vqa-rad --task vqa \
    --model-path checkpoints/llava-med-1.5_vqa-rad_mmedpo_lora --model-base checkpoints/llava-med-v1.5-mistral-7b \
    --question-file data/vqa-rad/test.jsonl --image-folder data/vqa-rad/images --out-dir results/full
  ```

  For the other runs, copy the matching line from `scripts/run_bench.sbatch`.

## What the scripts do

| File | What it does |
|---|---|
| `scripts/setup_env.sbatch` | Creates `.venv` with PyTorch 2.8 (CUDA 12.8) and transformers 4.37.2. Downloads LLaVA-Med v1.5, the CLIP image encoder, the four MMedPO adapters and the three test sets from Hugging Face, then runs `prepare_testsets.py`. |
| `scripts/prepare_testsets.py` | Rewrites each test set as one JSON line per item: image file, question and answer key. |
| `scripts/run_bench.sbatch` | Finds 4 idle GPUs and starts one `bench_inference.py` run per adapter, in parallel. |
| `scripts/bench_inference.py` | Loads LLaVA-Med v1.5 and merges one MMedPO adapter into it. Then, for every test item, it builds the prompt, prepares the image, generates the answer and records the timings. The first 3 items are warm-up and are left out of the averages. |
| `scripts/summarize.py` | Collects the `*_summary.json` files into one table. |

The prompts and generation settings match the authors' inference scripts in `MMedPO/inference/`: vicuna_v1 chat template, temperature 0.2, at most 64 new tokens for questions and 1,024 for reports.

## Notes

- **MIMIC-CXR** needs credentialed PhysioNet access, so it isn't downloaded. The MIMIC-CXR adapter is timed on the IU-Xray images instead. That measures its speed, not its report quality.
- **PyTorch version.** The paper used PyTorch 2.1.2, which has no kernels for Blackwell GPUs such as the B200. PyTorch 2.8 works on both older and newer GPUs.
- **The MMedPO code** is included as a git submodule pinned to commit `ec5ea44` of [aiming-lab/MMedPO](https://github.com/aiming-lab/MMedPO). If you cloned without `--recursive`, the setup script fetches it for you.
- **Accuracy.** The summary includes a rough yes/no accuracy check, but this repo doesn't compute the paper's scores. For those, use `MMedPO/eval/eval_vqa.py` and `MMedPO/eval/eval_report.py` on the answer files.

## Background reading

- [`guides/MMedPO_Guide1_Understanding_the_Paper.pdf`](guides/MMedPO_Guide1_Understanding_the_Paper.pdf) explains the paper from the ground up, assuming no recent-AI or medical background.
- [`guides/MMedPO_Guide2_Running_on_the_Cluster.pdf`](guides/MMedPO_Guide2_Running_on_the_Cluster.pdf) walks through this pipeline and Slurm for someone who has never run a model.
- [`report/mmedpo_inference_report.html`](report/mmedpo_inference_report.html) is a one-page summary of the timing results.

## Credits

- MMedPO: Zhu et al., *MMedPO: Aligning Medical Vision-Language Models with Clinical-Aware Multimodal Preference Optimization*, ICML 2025. Code at [aiming-lab/MMedPO](https://github.com/aiming-lab/MMedPO) (Apache-2.0). Adapters from [zky11235/mmedpo_checkpoints](https://huggingface.co/zky11235/mmedpo_checkpoints).
- Base model: [microsoft/llava-med-v1.5-mistral-7b](https://huggingface.co/microsoft/llava-med-v1.5-mistral-7b).
- Test sets: [VQA-RAD](https://huggingface.co/datasets/flaviagiammarino/vqa-rad), [SLAKE](https://huggingface.co/datasets/BoKelvin/SLAKE) and [IU-Xray](https://huggingface.co/datasets/dz-osamu/IU-Xray). Each dataset has its own terms of use.
