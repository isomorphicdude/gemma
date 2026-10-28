# Evaluation Metrics and Helpers for HD Text

This package contains all evaluation pipelines, custom metrics, and helpers
used during the SFT training of Hackable Diffusion (HD) text models.

Using custom JAX-compatible and host-side metrics, it tracks performance
across different SFT tasks (Sudoku, PubMedQA).

## Key Components

### 1. Base Metric Classes (`base_metric.py`)

Shared base classes (`BaseTokenizerMetric`, `BaseTextMetric`,
`BaseSimpleTextMetric`) that consolidate tokenizer access, 3D→2D squeezing,
and host-side `io_callback` wiring into a reusable hierarchy.

### 2. PubMedQA Evaluation (`pubmedqa_eval.py`)

Contains evaluation helpers for the PubMedQA dataset (both short yes/no/maybe
answers and long explanations).

*   **`extract_pubmedqa_answer`**: Extracts the structured answer from the
    model's response by looking for the `"The answer is: {yes|no|maybe}"`
    marker.
*   **`PubMedQAAccuracy`**: Computes accuracy of the extracted answers
    against the ground truth.
*   **`BLEUScore`**: Computes smoothed sentence-level BLEU between generated and
    ground-truth text using `sacrebleu`.

### 3. Sudoku Evaluation (`sudoku_eval.py`)

A comprehensive evaluation suite for Sudoku puzzle solving.

*   **`SudokuAllMetrics`**: A unified host-side metric that computes **20
    different stratified accuracy metrics** (e.g., overall cell accuracy,
    puzzle-level correctness, accuracy stratified by difficulty
    (easy/medium/hard), and count of completely correct puzzles).
    Grouping these under a single Kauldron metric key avoids TPU HBM OOM.

*  **Hardness Stratification**: Difficulty is categorized dynamically on-device
    according to the number of non-zero clue digits in the initial puzzle
    context:
    *   **All**: Every puzzle in the eval dataset.
    *   **Easy**: Puzzles with $\ge 40$ clues.
    *   **Medium**: Puzzles with $\ge 30$ and $< 40$ clues.
    *   **Hard**: Puzzles with $< 30$ clues.
*  **Monitored Metrics Modes**:
    *   **`accuracy`**: Exact grid match correctness (requires all 81 digits to
        match reference).
    *   **`cell_accuracy`**: Fraction of correctly infilled digits.
    *   **`exact_mask_accuracy`**: Exact preservation correctness of all
        initial clues.
    *   **`partial_mask_accuracy`**: Fraction of initial clues preserved.
    *   **`count`**: Tracks the quantity of evaluated instances in each
        difficulty category.

### 3b. Nurikabe Evaluation (`nurikabe_eval.py`)

Scores 10x10 Nurikabe outputs (100 cells: `#` for sea, island size for island
cells). The first 100 cell tokens (`#`, `?` or a whole number, so `10` is one
cell) are taken from the decoded sample.

*   **`NurikabeAllMetrics`**: host-side metric reporting
    *   **`nurikabe_accuracy`**: the shading passes `nurikabe_verify.verify`
        (clues white, no 2x2 sea, connected sea, one clue per island of that
        size) and every cell is `#` or a number. Solutions are unique, so this
        equals matching the ground-truth shading.
    *   **`nurikabe_exact_accuracy`**: all 100 cells, island sizes included,
        match the ground truth.
    *   **`nurikabe_cell_accuracy`**: fraction of unknown (`?`) cells shaded
        as in the ground truth.
    *   **`nurikabe_clue_accuracy`**: every clue cell keeps its number.
    *   **`nurikabe_well_formed`**: exactly 100 cells, each `#` or a number.
    *   **`nurikabe_count`**: number of evaluated examples.
*   **`NurikabeBoardImages`**: summary rendering the puzzle next to the
    model's board with rule violations highlighted
    (`nurikabe_viz.render_nurikabe`).
*   **`nurikabe_verify.py`**: standalone verifier, also usable from the
    command line on a board file.

### 4. Text Metrics (`text_metric.py`)

A utility metric to decode and log generated samples alongside prompts during
training.

*   **`DetokenizePromptAndResponse`**: A custom Kauldron metric that collects
    prompt and response tokens, decodes them using the SentencePiece tokenizer,
    and logs the detokenized text strings.

### 5. Autoregressive Evaluation (`ar_eval.py`)

Contains autoregressive evaluation helpers for DiffusionGemma SFT.

*   **`make_ar_evals`**: Creates autoregressive evaluation configurations.
*   **`AR_DENOISING_STEPS`**: Defines the denoising step counts for AR evaluation.

