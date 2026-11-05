#!/bin/bash
#SBATCH -J nnunet-setup
#SBATCH -p hgx
#SBATCH -c 32
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
# Builds the Python environment for nnU-Net (no GPU work).
# Run from this exercise's folder: sbatch scripts/setup_env.sbatch  (without Slurm: bash scripts/setup_env.sbatch)
set -euo pipefail
W=${SLURM_SUBMIT_DIR:-$(pwd)}
UV=${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}
cd $W
[ -f scripts/env.sh ] || { echo "run this from exercises/02-nnunet-breast-mri"; exit 1; }
[ -x "$UV" ] || { echo "uv not found; install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }
export UV_CACHE_DIR=$W/.uv_cache
# A cold cache downloads ~3.5 GB of wheels; fewer parallel downloads keep the job under its memory limit.
export UV_CONCURRENT_DOWNLOADS=4

# PyTorch 2.8 + CUDA 12.8 has kernels for Blackwell GPUs (B200); nnU-Net 2.8.1 supports torch 2.1-2.8.
$UV venv --python 3.11 .venv
source .venv/bin/activate
$UV pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0
# torch is pinned again so the second install keeps the CUDA 12.8 build.
$UV pip install torch==2.8.0 nnunetv2==2.8.1 synapseclient pandas openpyxl
python -c "import torch, nnunetv2, importlib.metadata as m; print('torch', torch.__version__, 'cuda', torch.version.cuda, '| nnunetv2', m.version('nnunetv2'))"
# nnU-Net only finds trainer classes inside its own package, so install the benchmark trainer there.
NNUNET_DIR=$(python -c "import nnunetv2, os; print(os.path.dirname(nnunetv2.__file__))")
cp scripts/nnUNetTrainerBenchmark.py $NNUNET_DIR/training/nnUNetTrainer/variants/
echo SETUP_DONE
