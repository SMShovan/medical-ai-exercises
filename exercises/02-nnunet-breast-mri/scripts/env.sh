# Paths shared by this exercise's scripts. Source it after setting W to the exercise folder.
# nnU-Net finds its data and models through these three variables.
export nnUNet_raw=$W/data/nnUNet_raw
export nnUNet_preprocessed=$W/data/nnUNet_preprocessed
export nnUNet_results=$W/models/nnUNet_results
export MAMAMIA_DIR=$W/data/mamamia       # the dataset as downloaded from Synapse
export PYTHONUNBUFFERED=1
mkdir -p "$nnUNet_raw" "$nnUNet_preprocessed" "$nnUNet_results" "$MAMAMIA_DIR"
