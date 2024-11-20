#!/bin/bash

if ! command -v conda &> /dev/null && ! command -v mamba &> /dev/null; then
    echo "Error: Neither conda nor mamba is installed."
    exit 1
fi

if [ ! -f environment.yml ]; then
    echo "Error: environment.yml not found."
    exit 1
fi

# Setup conda environment
conda env create --file=environment.yml
conda activate dymoval_dev

# Editable install
pip install -e .

# symlink pre-commit hook
ln -sf ../../.githooks/pre-commit .git/hooks/pre-commit
