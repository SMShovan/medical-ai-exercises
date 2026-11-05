# Shared helpers for the exercises' Slurm scripts. Load with:  . "$ROOT/common/gpu.sh"

# pick_idle_gpus N: set the array GPUS to the indices of N GPUs that have almost no memory in use.
# Slurm on our cluster does not hand out GPUs (no GRES), so jobs pick idle ones themselves.
# On clusters that hide the GPUs a job wasn't given, this simply picks from the job's own GPUs.
pick_idle_gpus() {
  local n=$1 idle
  # No `head` here: under `set -o pipefail` it can kill nvidia-smi with SIGPIPE and fail the job.
  mapfile -t idle < <(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' '$2<1000{print $1}')
  if [ ${#idle[@]} -lt "$n" ]; then
    echo "need $n idle GPUs, found ${#idle[@]}" >&2
    return 1
  fi
  GPUS=("${idle[@]:0:$n}")
  echo "using GPUs ${GPUS[*]}"
}
