#!/bin/bash
#SBATCH -J mmedpo-setup
#SBATCH -p hgx
#SBATCH -c 16
#SBATCH -t 01:30:00
#SBATCH -o logs/%x-%j.out
# Builds the Python env and downloads model weights + test sets (no GPU work).
# Run from this exercise's folder: sbatch scripts/setup_env.sbatch  (without Slurm: bash scripts/setup_env.sbatch)
set -euo pipefail
W=${SLURM_SUBMIT_DIR:-$(pwd)}
UV=${UV:-$(command -v uv || echo "$HOME/.local/bin/uv")}
export W HF_HOME=$W/hf_cache UV_CACHE_DIR=$W/.uv_cache HF_HUB_ENABLE_HF_TRANSFER=1
cd $W
[ -f scripts/bench_inference.py ] || { echo "run this from exercises/01-mmedpo-inference"; exit 1; }
[ -x "$UV" ] || { echo "uv not found; install it with: curl -LsSf https://astral.sh/uv/install.sh | sh"; exit 1; }

# --- MMedPO code (git submodule, pinned to the commit we ran) -------------------
[ -f MMedPO/train/dpo/llava/model/builder.py ] || git submodule update --init

# --- Python env ---------------------------------------------------------------
# Paper used torch 2.1.2, which has no Blackwell (B200, sm_100) kernels; torch 2.8 + cu128 does.
# transformers 4.37.2 is the last release the LLaVA-1.5 model code runs on unmodified.
$UV venv --python 3.10 .venv
source .venv/bin/activate
$UV pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.8.0 torchvision==0.23.0
$UV pip install "transformers==4.37.2" "tokenizers==0.15.2" "peft==0.11.1" "accelerate==0.28.0" \
    "numpy<2" sentencepiece protobuf shortuuid einops einops-exts pillow pandas pyarrow tqdm \
    "huggingface_hub[hf_transfer]<0.26"
python -c "import torch, transformers, peft; print('torch', torch.__version__, 'cuda', torch.version.cuda, '| transformers', transformers.__version__, '| peft', peft.__version__)"

# --- Weights ------------------------------------------------------------------
python - <<'EOF'
import os
from huggingface_hub import snapshot_download
W = os.environ["W"]
snapshot_download("microsoft/llava-med-v1.5-mistral-7b", local_dir=f"{W}/checkpoints/llava-med-v1.5-mistral-7b")
snapshot_download("openai/clip-vit-large-patch14-336", allow_patterns=["*.json", "*.txt", "pytorch_model.bin"])
snapshot_download("zky11235/mmedpo_checkpoints", local_dir=f"{W}/checkpoints/mmedpo_checkpoints")
EOF
# LLaVA's loader only takes the LoRA-merge path when the folder name contains "lora".
for ds in iuxray mimic slake vqa-rad; do
  ln -sfn mmedpo_checkpoints/llava-med-1.5_${ds}_mmedpo $W/checkpoints/llava-med-1.5_${ds}_mmedpo_lora
done

# --- Test sets ----------------------------------------------------------------
python - <<'EOF'
import os
from huggingface_hub import hf_hub_download
D = os.environ["W"] + "/data"
hf_hub_download("flaviagiammarino/vqa-rad", "data/test-00000-of-00001-e5bc3d208bb4deeb.parquet", repo_type="dataset", local_dir=f"{D}/vqa-rad")
for f in ["test.json", "imgs.zip"]:
    hf_hub_download("BoKelvin/SLAKE", f, repo_type="dataset", local_dir=f"{D}/slake")
for f in ["test.jsonl", "image.zip"]:
    hf_hub_download("dz-osamu/IU-Xray", f, repo_type="dataset", local_dir=f"{D}/iuxray")
EOF
cd $W/data/slake && unzip -qo imgs.zip
cd $W/data/iuxray && unzip -qo image.zip
# The zip keeps its authors' full path; point images/ at the folder inside it.
ln -sfn mnt/bn/haiyang-dataset-lq/medical/home/yisiyang/ysy/medical_dataset/iu_xray/iu_xray/images images
cd $W && python scripts/prepare_testsets.py
du -sh $W/checkpoints $W/hf_cache $W/data
echo SETUP_DONE
