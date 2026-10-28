#!/bin/bash

# Download the Nurikabe 10x10 JSONL splits from the Hugging Face Hub and
# convert them to Bagz. Usage: bash prepare_nurikabe_dataset.sh [OUT_DIR]
# OUT_DIR defaults to this directory, where the configs expect the files.
# Requires the `hf` CLI (huggingface_hub); the dataset is public.

# Exit on error
set -e

# Use the directory of the script to find the relative paths
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" &> /dev/null && pwd)"

OUT_DIR="${1:-$SCRIPT_DIR}"

mkdir -p "$OUT_DIR"

echo "Downloading Nurikabe dataset from the Hugging Face Hub..."
hf download isometricneko/nurikabe10x10 \
    nurikabe_10x10.train.jsonl \
    nurikabe_10x10.validation.jsonl \
    nurikabe_10x10.test.jsonl \
    --repo-type dataset --local-dir "$OUT_DIR"

echo "Running conversion script..."
# Use the currently active Python environment
python3 "${SCRIPT_DIR}/convert_nurikabe.py" \
    --input_dir="$OUT_DIR" \
    --output_dir="$OUT_DIR"

echo "Conversion complete! Output files are in $OUT_DIR"
