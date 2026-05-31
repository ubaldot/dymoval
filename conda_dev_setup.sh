#!/bin/bash

set -e

# initialize conda for non-interactive shell
source "$(conda info --base)/etc/profile.d/conda.sh"

if ! command -v conda &> /dev/null && ! command -v mamba &> /dev/null; then
    echo "Error: Neither conda nor mamba is installed."
    exit 1
fi

if [ ! -f environment.yml ]; then
    echo "Error: environment.yml not found."
    exit 1
fi

# create env (optional: avoid error if exists)
conda env update -f environment.yml --prune
conda activate dymoval_dev

pip install -e .

ln -sf ../../.githooks/pre-commit .git/hooks/pre-commit
