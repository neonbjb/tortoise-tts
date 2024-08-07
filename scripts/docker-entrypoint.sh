#!/bin/bash
# Source the conda environment
source /root/miniconda/etc/profile.d/conda.sh
conda activate tortoise

# Execute the Python script with passed arguments
python /app/tortoise/do_tts.py "$@"