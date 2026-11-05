#!/bin/bash
#SBATCH -J nnunet-infer
#SBATCH -p hgx
#SBATCH -c 48
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%j.out
# Inference timing for the MAMA-MIA pretrained nnU-Net: three settings at once, one GPU each,
# on the first 100 fold-0 validation patients (first post-contrast phase; LIMIT=0 for all ~300).
#   fold0_nomirror  one model, no test-time mirroring     (fastest)
#   fold0_mirror    one model, 8 mirrored passes          (nnU-Net's default for one model)
#   ensemble5_mirror all 5 fold models, mirrored          (what `nnUNetv2_predict` does by default)
# Run from this exercise's folder: sbatch scripts/bench_infer.sbatch   (LIMIT=6 for a quick test)
set -euo pipefail
W=${SLURM_SUBMIT_DIR:-$(pwd)}
cd $W
[ -f scripts/env.sh ] || { echo "run this from exercises/02-nnunet-breast-mri"; exit 1; }
ROOT=$(cd "$W/../.." && pwd)
. "$ROOT/common/gpu.sh"
source .venv/bin/activate
. scripts/env.sh
MODEL=$nnUNet_results/full_image_dce_mri_tumor_segmentation
OUT=${OUT:-$W/results/full}
EXTRA="--limit ${LIMIT:-100}"
mkdir -p $OUT

nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
pick_idle_gpus 3

run() {  # gpu tag extra-args...
  local gpu=$1 tag=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu python scripts/bench_infer.py --tag $tag --model-dir $MODEL \
    --data-dir $MAMAMIA_DIR --out-dir $OUT $EXTRA "$@" > $W/logs/infer_$tag.log 2>&1
}
run ${GPUS[0]} fold0_nomirror   --folds 0 --no-mirroring &
run ${GPUS[1]} fold0_mirror     --folds 0 &
run ${GPUS[2]} ensemble5_mirror --folds 0,1,2,3,4 &
wait
echo INFER_DONE
