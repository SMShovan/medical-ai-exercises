#!/bin/bash
#SBATCH -J mamamia-download
#SBATCH -p hgx
#SBATCH -c 16
#SBATCH -t 06:00:00
#SBATCH -o logs/%x-%j.out
# Downloads MAMA-MIA and the pretrained nnU-Net weights from Synapse (no GPU work).
# Run from this exercise's folder:  WHAT=weights sbatch scripts/download_data.sbatch   then   WHAT=data sbatch scripts/download_data.sbatch
# (re-run either one if it stops early: finished files are skipped)
set -euo pipefail
W=${SLURM_SUBMIT_DIR:-$(pwd)}
cd $W
[ -f scripts/env.sh ] || { echo "run this from exercises/02-nnunet-breast-mri"; exit 1; }
[ -f ~/.synapseConfig ] || { echo "no ~/.synapseConfig: create a Synapse token first (README, step 2)"; exit 1; }
source .venv/bin/activate
. scripts/env.sh
python scripts/download_mamamia.py --what ${WHAT:-data}
du -sh "$MAMAMIA_DIR" "$nnUNet_results"
