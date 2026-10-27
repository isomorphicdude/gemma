# Data Pipelines for Hackable Diffusion (HD) Text

This package contains all data loading, preprocessing, and dataset-specific
pipelines for the Hackable Diffusion text SFT project.

## Package Structure

*   `data.py`: Common data utilities, transforms (like `CanvasChunker` and
    `SequenceTargetShift`), and shared pipeline helpers.
*   `pubmedqa/`: Dataset pipeline for the PubMedQA medical Q&A task.
*   `sudoku/`: Dataset pipeline for the Sudoku puzzle solving task.
*   `nurikabe/`: JSONL-to-Bagz conversion for the Nurikabe 10x10 dataset.

## Key Components

### Shared Transforms (`data.py`)

*   **`CanvasChunker`**: Splices a long response token sequence into multiple,
    fixed-size "canvas" chunks. This is required for the localized diffusion
    process, allowing the model to denoise specific parts of the text
    iteratively.
*   **`SequenceTargetShift`**: Shifts sequences to prepare targets for the
    encoder loss.

### Dataset-Specific Pipelines

Each subfolder contains:

1.  A data loader or custom `DataSource` (e.g., `JsonlDataSource` for
    PubMedQA, `Bagz` wrapper for Sudoku).
2.  A conversion script (`convert_*.py`) to preprocess raw data into the
    format expected by the pipeline.
3.  A `README.md` explaining the dataset structure.

## Converting Nurikabe

Download the JSONL files from
[`isometricneko/nurikabe10x10`](https://huggingface.co/datasets/isometricneko/nurikabe10x10)
into `gemma/diffusion/hackable_diffusion_adapter/data/nurikabe/`, then run from
the repository root:

```sh
.venv/bin/python gemma/diffusion/hackable_diffusion_adapter/data/nurikabe/convert_nurikabe.py \
  --output_dir=gemma/diffusion/hackable_diffusion_adapter/data/nurikabe
```

The converter defaults to reading the JSONL files alongside the script. Use
`--input_dir` to read from another directory, and `--max_records=N` to limit the
number of examples **per split** (`-1` converts all; `0` writes empty splits).
It preserves the upstream split boundaries:

| Input | Output |
| --- | --- |
| `nurikabe_10x10.train.jsonl` | `nurikabe_train.bagz` |
| `nurikabe_10x10.validation.jsonl` | `nurikabe_eval.bagz` |
| `nurikabe_10x10.test.jsonl` | `nurikabe_test.bagz` |

Each Bagz record is a serialized `tf.train.Example` with scalar UTF-8 byte
features `puzzle` and `solution`, matching the existing puzzle loader's schema.
The grids remain 100 space-separated cells (`?` for unknown cells, `#` for sea,
and island sizes `1` through `10`); multi-digit numbers are not split into
digits. Grid dimensions and clue alignment are validated, while source metadata
(`height`, `width`, `layout_key`) is not stored in the output.
