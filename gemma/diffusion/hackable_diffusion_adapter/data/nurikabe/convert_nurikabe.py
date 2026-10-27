# Copyright 2026 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Convert the Nurikabe 10x10 JSONL splits to Bagz tf.train.Example files.

Preserves the upstream train/validation/test splits, naming validation "eval" to
match the adapter's convention. Each example contains scalar UTF-8 byte features
"puzzle" and "solution", with one space-separated token per cell. In particular,
"10" stays a single cell; "?" (unknown) and "#" (sea) are preserved.
"""

import json
import os

from absl import app
from absl import flags
from absl import logging
import bagz
import tensorflow as tf

################################################################################
# MARK: Flags
################################################################################

_INPUT_DIR = flags.DEFINE_string(
    "input_dir",
    os.path.dirname(__file__),
    "Directory containing nurikabe_10x10.{train,validation,test}.jsonl.",
)
_OUTPUT_DIR = flags.DEFINE_string(
    "output_dir",
    "",
    "Directory to write the output Bagz files.",
)
_MAX_RECORDS = flags.DEFINE_integer(
    "max_records",
    -1,
    "Maximum number of records to process per split. Use -1 for all.",
    lower_bound=-1,
)

_SPLITS = (("train", "train"), ("validation", "eval"), ("test", "test"))
_ISLAND_SIZES = frozenset(str(size) for size in range(1, 11))


################################################################################
# MARK: Helpers
################################################################################


def make_tf_example(features_dict: dict[str, bytes]) -> tf.train.Example:
  """Create a tf.train.Example proto from a dictionary of byte features."""
  tf_features = {
      k: tf.train.Feature(bytes_list=tf.train.BytesList(value=[v]))
      for k, v in features_dict.items()
  }
  return tf.train.Example(features=tf.train.Features(feature=tf_features))


def _parse_record(record: object) -> dict[str, bytes]:
  """Validate and encode a JSONL record without splitting multi-digit cells."""
  if not isinstance(record, dict):
    raise ValueError("Expected a JSON object.")
  if record.get("height") != 10 or record.get("width") != 10:
    raise ValueError("Expected height=10 and width=10.")

  grids = {}
  for field, placeholder in (("puzzle", "?"), ("solution", "#")):
    text = record.get(field)
    if not isinstance(text, str):
      raise ValueError(f"{field} must be a string.")
    tokens = text.split()
    if len(tokens) != 100:
      raise ValueError(f"{field} must contain 100 cells; got {len(tokens)}.")
    for token in tokens:
      if token != placeholder and token not in _ISLAND_SIZES:
        raise ValueError(f"{field} contains invalid cell {token!r}.")
    grids[field] = tokens

  for index, (clue, cell) in enumerate(
      zip(grids["puzzle"], grids["solution"], strict=True)
  ):
    if clue != "?" and clue != cell:
      raise ValueError(f"solution does not preserve the clue at cell {index}.")

  return {
      field: " ".join(tokens).encode("utf-8")
      for field, tokens in grids.items()
  }


def _convert_split(
    input_path: str, output_path: str, max_records: int = -1
) -> int:
  """Stream a JSONL split to Bagz and return the number of examples written."""
  if max_records < -1:
    raise ValueError("max_records must be -1 or non-negative.")

  num_records = 0
  with open(input_path, encoding="utf-8") as source, bagz.Writer(
      output_path
  ) as writer:
    for line_number, line in enumerate(source, start=1):
      if max_records >= 0 and num_records >= max_records:
        break
      if not line.strip():
        continue
      try:
        features = _parse_record(json.loads(line))
      except ValueError as error:
        raise ValueError(f"{input_path}:{line_number}: {error}") from error
      writer.write(make_tf_example(features).SerializeToString())
      num_records += 1
  return num_records


################################################################################
# MARK: Main
################################################################################


def main(argv):
  """Convert downloaded Nurikabe train, validation, and test JSONL files."""
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  input_dir = _INPUT_DIR.value
  output_dir = _OUTPUT_DIR.value
  if not output_dir:
    raise app.UsageError("--output_dir must be specified.")

  paths = [
      (
          os.path.join(input_dir, f"nurikabe_10x10.{source_split}.jsonl"),
          os.path.join(output_dir, f"nurikabe_{output_split}.bagz"),
      )
      for source_split, output_split in _SPLITS
  ]
  for input_path, _ in paths:
    if not os.path.isfile(input_path):
      raise FileNotFoundError(f"Missing Nurikabe JSONL split: {input_path}")

  os.makedirs(output_dir, exist_ok=True)
  for input_path, output_path in paths:
    logging.info("Converting %s to %s", input_path, output_path)
    num_records = _convert_split(input_path, output_path, _MAX_RECORDS.value)
    logging.info("Wrote %d records to %s", num_records, output_path)

  logging.info("Conversion complete.")


if __name__ == "__main__":
  app.run(main)
