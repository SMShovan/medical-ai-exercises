# Medical AI exercises

Hands-on exercises from the OSU AI Hub MRI project. Each exercise downloads a published model and public data, runs it on GPUs, and records how it performs: how long it takes, how much GPU memory it needs, and how it scales. Everything was run on hgx-01, a Slurm node with 8 NVIDIA B200 GPUs.

| # | Exercise | Model | Data | What it measures |
|---|---|---|---|---|
| 01 | [MMedPO inference](exercises/01-mmedpo-inference/) | LLaVA-Med v1.5 + MMedPO adapters (vision-language) | VQA-RAD, SLAKE, IU-Xray | Time per answer and GPU memory for the four released models |
| 02 | [nnU-Net on breast MRI](exercises/02-nnunet-breast-mri/) | nnU-Net (segmentation) | MAMA-MIA breast DCE-MRI | Inference time and memory per scan; training speed on 1, 2 and 4 GPUs |

## How the repository is organised

```
medical-ai-exercises/
├── README.md            this page
├── common/              small helpers shared by the exercises (e.g. picking idle GPUs)
└── exercises/
    ├── 01-mmedpo-inference/
    └── 02-nnunet-breast-mri/
```

Every exercise folder is self-contained and follows the same pattern:

- `README.md`: what it does, our results, and how to reproduce them.
- `scripts/setup_env.sbatch`: builds the exercise's own Python environment (`.venv/`) and downloads what it needs.
- One or more benchmark scripts, plus `scripts/summarize.py` to print a results table.
- `results/reference/`: the numbers from our runs, for comparison.
- `logs/`: where Slurm writes job logs.

Downloaded models, datasets and environments stay inside the exercise folder and are not committed (see `.gitignore`). Run each exercise's commands from inside its own folder.

## Getting started

```bash
git clone --recursive https://github.com/SMShovan/medical-ai-exercises.git
cd medical-ai-exercises/exercises/01-mmedpo-inference    # or another exercise
```

Then follow that exercise's README. You need Linux, NVIDIA GPUs with a driver for CUDA 12.8, `git`, and [`uv`](https://docs.astral.sh/uv/). Slurm is optional: the `.sbatch` files are plain bash scripts.

## Adding an exercise

Create `exercises/NN-short-name/` with the same layout. Use `common/gpu.sh` to pick GPUs, keep large files inside the folder under names `.gitignore` already covers (`.venv/`, `data/`, `models/`, `checkpoints/`, `results/full/`), and add a row to the table above.
