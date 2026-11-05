#!/bin/bash
#SBATCH -J nnunet-preprocess
#SBATCH -p hgx
#SBATCH -c 32
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
# Builds the training dataset for the speed benchmark and runs nnU-Net preprocessing on it (CPU only).
# Run from this exercise's folder after the data download: sbatch scripts/preprocess.sbatch  (N_CASES=300 by default)
set -euo pipefail
W=${SLURM_SUBMIT_DIR:-$(pwd)}
cd $W
[ -f scripts/env.sh ] || { echo "run this from exercises/02-nnunet-breast-mri"; exit 1; }
source .venv/bin/activate
. scripts/env.sh
python scripts/prepare_dataset.py --n-cases ${N_CASES:-300}
# Resample to 1 mm, crop, z-score, and store as compressed arrays; -np = parallel worker processes.
nnUNetv2_preprocess -d 105 -plans_name nnUNetPlans -c 3d_fullres -np 24
du -sh $nnUNet_preprocessed/Dataset105_full_image
echo PREPROCESS_DONE
