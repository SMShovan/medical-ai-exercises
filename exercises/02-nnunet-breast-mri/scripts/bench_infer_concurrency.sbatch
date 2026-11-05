#!/bin/bash
#SBATCH -J nnunet-infer-conc
#SBATCH -p hgx
#SBATCH -c 64
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
# How many scans per hour can ONE GPU segment? nnU-Net inference uses ~1.6 GB of GPU memory and spends
# much of each scan on CPU work, so several runs can share a GPU. This runs K = 1, 2, 4, 8 copies of the
# one-model, mirrored setting on the same GPU, each on its own 24 patients, and records the wall time.
# Run from this exercise's folder: sbatch scripts/bench_infer_concurrency.sbatch
set -euo pipefail
W=${SLURM_SUBMIT_DIR:-$(pwd)}
cd $W
[ -f scripts/env.sh ] || { echo "run this from exercises/02-nnunet-breast-mri"; exit 1; }
ROOT=$(cd "$W/../.." && pwd)
. "$ROOT/common/gpu.sh"
source .venv/bin/activate
. scripts/env.sh
MODEL=$nnUNet_results/full_image_dce_mri_tumor_segmentation
OUT=${OUT:-$W/results/full}/concurrency
PER_RUN=${PER_RUN:-24}
mkdir -p $OUT
pick_idle_gpus 1
echo "K,scans,wall_s,scans_per_hour" > $OUT/concurrency.csv
for K in ${KS:-1 2 4 8}; do
  t0=$(date +%s.%N)
  for k in $(seq 0 $((K - 1))); do
    CUDA_VISIBLE_DEVICES=${GPUS[0]} python scripts/bench_infer.py --tag k${K}_run$k --model-dir $MODEL --folds 0 \
      --data-dir $MAMAMIA_DIR --out-dir $OUT --offset $((k * PER_RUN)) --limit $PER_RUN \
      > $W/logs/infer_conc_k${K}_run$k.log 2>&1 &
  done
  wait
  t1=$(date +%s.%N)
  python - "$K" "$PER_RUN" "$t0" "$t1" >> $OUT/concurrency.csv <<'EOF'
import sys
k, n, t0, t1 = int(sys.argv[1]), int(sys.argv[2]), float(sys.argv[3]), float(sys.argv[4])
wall = t1 - t0
print(f"{k},{k * n},{wall:.1f},{3600 * k * n / wall:.0f}")
EOF
  tail -1 $OUT/concurrency.csv
done
echo CONCURRENCY_DONE
