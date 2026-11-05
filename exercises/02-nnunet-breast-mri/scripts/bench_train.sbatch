#!/bin/bash
#SBATCH -J nnunet-train-bench
#SBATCH -p hgx
#SBATCH -c 64
#SBATCH -t 06:00:00
#SBATCH -o logs/%x-%j.out
# Training-speed benchmark for nnU-Net 3d_fullres on breast DCE-MRI (the authors' exact plans).
# Each run trains fold 0 for a few epochs with nnUNetTrainerBenchmark and records the time per epoch,
# the peak GPU memory and the GPU utilisation. Runs go one after another so they don't compete for CPUs,
# which nnU-Net uses heavily for data augmentation.
#   GPUS:PLANS        1:nnUNetPlans is the authors' setup (1 GPU, batch 2)
#   1:nnUNetPlans_bs8 one GPU, 4x the batch (needs the memory a B200 has)
#   4:nnUNetPlans_bs8 four GPUs, 2 per GPU (data-parallel, DDP)
# Run from this exercise's folder after preprocessing: sbatch scripts/bench_train.sbatch
# Optional: EPOCHS=4 (per run), RUNS="1:nnUNetPlans 2:nnUNetPlans_bs4" to choose runs.
set -euo pipefail
W=${SLURM_SUBMIT_DIR:-$(pwd)}
cd $W
[ -f scripts/env.sh ] || { echo "run this from exercises/02-nnunet-breast-mri"; exit 1; }
ROOT=$(cd "$W/../.." && pwd)
. "$ROOT/common/gpu.sh"
source .venv/bin/activate
. scripts/env.sh
OUT=${OUT:-$W/results/full}
mkdir -p $OUT
export NNUNET_BENCH_EPOCHS=${EPOCHS:-4}
export nnUNet_n_proc_DA=${N_PROC_DA:-12}   # data-augmentation worker processes per GPU (nnU-Net's default)
RUNS=${RUNS:-"1:nnUNetPlans 1:nnUNetPlans_bs4 1:nnUNetPlans_bs8 2:nnUNetPlans 2:nnUNetPlans_bs4 4:nnUNetPlans_bs8"}

nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
pick_idle_gpus 4

for r in $RUNS; do
  g=${r%%:*}
  plans=${r#*:}
  tag=g${g}_${plans}
  gpus=$(IFS=,; echo "${GPUS[*]:0:$g}")
  echo "== $tag on GPUs $gpus ($(date +%T))"
  nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader -l 2 -i $gpus \
    > $OUT/train_${tag}_gpu_util.csv &
  smi=$!
  CUDA_VISIBLE_DEVICES=$gpus NNUNET_BENCH_OUT=$OUT/train_${tag}.json \
    nnUNetv2_train 105 3d_fullres 0 -tr nnUNetTrainerBenchmark -p $plans -num_gpus $g \
    > $W/logs/train_${tag}.log 2>&1 || echo "   $tag failed, see logs/train_${tag}.log"
  kill $smi
done
echo TRAIN_BENCH_DONE
