#!/bin/bash
#SBATCH -J mmedpo-bench
#SBATCH -p hgx
#SBATCH -c 48
#SBATCH -t 02:30:00
#SBATCH -o logs/%x-%j.out
# Full-test-set inference timing for the released MMedPO checkpoints, one GPU per run.
# Slurm on this cluster does not manage GPUs (no GRES), so pick idle ones ourselves.
# Run from this exercise's folder: sbatch scripts/run_bench.sbatch  (without Slurm: bash scripts/run_bench.sbatch)
# Optional: LIMIT=20 sbatch scripts/run_bench.sbatch  answers only the first 20 items of each test set.
set -euo pipefail
W=${SLURM_SUBMIT_DIR:-$(pwd)}
cd $W
[ -f scripts/bench_inference.py ] || { echo "run this from exercises/01-mmedpo-inference"; exit 1; }
ROOT=$(cd "$W/../.." && pwd)
. "$ROOT/common/gpu.sh"
export HF_HOME=$W/hf_cache HF_HUB_OFFLINE=1 PYTHONUNBUFFERED=1
BASE=$W/checkpoints/llava-med-v1.5-mistral-7b
OUT=${OUT:-$W/results/full}
EXTRA=${LIMIT:+--limit $LIMIT}
mkdir -p $OUT

nvidia-smi --query-gpu=index,memory.used,utilization.gpu --format=csv
pick_idle_gpus 4

run() {  # gpu tag task lora question_file image_folder
  CUDA_VISIBLE_DEVICES=$1 $W/.venv/bin/python $W/scripts/bench_inference.py --tag $2 --task $3 \
    --model-path $W/checkpoints/llava-med-1.5_$4_mmedpo_lora --model-base $BASE \
    --question-file $5 --image-folder $6 --out-dir $OUT $EXTRA > $W/logs/bench_$2.log 2>&1
}
run ${GPUS[0]} vqa-rad vqa    vqa-rad $W/data/vqa-rad/test.jsonl        $W/data/vqa-rad/images &
run ${GPUS[1]} slake   vqa    slake   $W/data/slake/test.jsonl          $W/data/slake/imgs &
run ${GPUS[2]} iuxray  report iuxray  $W/data/iuxray/test_mmedpo.jsonl  $W/data/iuxray/images &
# MIMIC-CXR needs PhysioNet credentials; time the MIMIC checkpoint on IU-Xray chest X-rays instead.
run ${GPUS[3]} mimic-ckpt_on_iuxray report mimic $W/data/iuxray/test_mmedpo.jsonl $W/data/iuxray/images &
wait
echo BENCH_DONE
