"""Sampling and neutral rendering shared by the puzzle trajectory notebooks.

Model/data dependencies are imported inside the loading functions: a saved run
can be inspected and exported on CPU without initializing JAX or TensorFlow.
"""

from __future__ import annotations

import base64
import dataclasses
import functools
import html
import importlib
import io
import json
import math
import os
from pathlib import Path
import re
import uuid

import numpy as np
from PIL import Image, ImageDraw, ImageFont


TASK_SIZES = {"sudoku": 9, "nurikabe": 10}
# Optional root for trained/ and <task>_data/ outside the checkout.
PROJECT_STORAGE = Path(os.environ.get(
    "GEMMA_PROJECT_STORAGE", Path(__file__).resolve().parents[3])).expanduser()


def latest_checkpoint(task: str, trained_root=PROJECT_STORAGE / "trained",
                      checkpoint=None) -> Path:
  """Find a completed Kauldron checkpoint, excluding temporary/partial saves.

  Select the largest step in each run, then the most recently committed run.
  An explicit path may name a run, a ckpt_<step> directory, or its default item.
  Never fall back to the pretrained model when trained weights are missing.
  """
  TASK_SIZES[task]
  root = Path(checkpoint or trained_root).expanduser().resolve()
  if checkpoint and root.name == "default":
    root = root.parent
  if checkpoint and re.fullmatch(r"ckpt_\d+", root.name):
    candidates = [root]
  elif checkpoint:
    candidates = list(root.glob("checkpoints/ckpt_*"))
  else:
    candidates = list(root.glob(f"{task}_*/checkpoints/ckpt_*"))
  runs = {}
  for path in candidates:
    match = re.fullmatch(r"ckpt_(\d+)", path.name)
    if (not match or not (path / "commit_success.txt").is_file()
        or not (path / "default" / "_METADATA").is_file()):
      continue
    step = int(match[1])
    previous = runs.get(path.parent)
    if previous is None or step > int(previous.name.removeprefix("ckpt_")):
      runs[path.parent] = path
  if not runs:
    raise FileNotFoundError(f"No completed {task} training checkpoint in {root}")
  return max(runs.values(), key=lambda p: (
      (p / "commit_success.txt").stat().st_mtime_ns,
      int(p.name.removeprefix("ckpt_")), str(p)))


def load_example(task: str, project_dir, *, example_index=0, bagz_path=None):
  """Use the config's held-out eval split and its complete preprocessing pipeline.

  Only the file location, slice, batch size and worker count are changed. The
  prompt template, BOS, right padding and target transforms come from the config.
  Returns (config, batch, raw puzzle/solution metadata).
  """
  TASK_SIZES[task]
  if example_index < 0:
    raise ValueError("example_index must be nonnegative")
  import jax  # pylint: disable=g-import-not-at-top
  import tensorflow as tf  # pylint: disable=g-import-not-at-top
  from kauldron import konfig  # pylint: disable=g-import-not-at-top
  from gemma.diffusion.hackable_diffusion_adapter.data.sudoku import sudoku_data

  jax.devices()
  tf.config.set_visible_devices([], "GPU")
  module = importlib.import_module(
      f"gemma.diffusion.hackable_diffusion_adapter.configs.sft_{task}")
  cfg = module.get_config()
  if bagz_path is not None:
    path = Path(bagz_path).expanduser().resolve()
  else:
    configured = Path(project_dir) / cfg.eval_ds.bagz_path
    staged = PROJECT_STORAGE / f"{task}_data" / configured.name
    path = configured if configured.is_file() else staged
  if not path.is_file():
    raise FileNotFoundError(f"Held-out Bagz data not found: {path}")
  source = sudoku_data.PicklableBagzReader(str(path))
  if example_index >= len(source):
    raise IndexError(f"Example {example_index} is outside {len(source)} records")
  raw = sudoku_data.ParseSudokuExample().map(source[example_index])
  cfg.eval_ds.bagz_path = str(path)
  cfg.eval_ds.batch_size = 1
  cfg.eval_ds.num_workers = 0
  cfg.eval_ds.slice_start = example_index
  cfg.eval_ds.slice_stop = example_index + 1
  dataset = konfig.resolve(cfg.eval_ds)
  dataset = dataclasses.replace(dataset, seed=cfg.seed)
  batch = next(iter(dataset))
  return cfg, batch, {
      "puzzle": raw["prompt"], "target": raw["response"],
      "example_index": example_index, "dataset": str(path), "split": "eval",
      "config": f"configs/sft_{task}.py",
  }


def restore_network_params(checkpoint, *, device=None, lora_only=False):
  """Restore only trained network weights, resharding onto the selected device.

  Optimizer state, data iterators and random initial weights are never loaded.
  Explicit single-device restore args allow a checkpoint saved on 2/4 GPUs to
  be inspected on one GPU (or a small fixture to be tested on CPU).
  Set lora_only=True to read just the adapter tensors, without base weights.
  """
  import jax  # pylint: disable=g-import-not-at-top
  import orbax.checkpoint as ocp  # pylint: disable=g-import-not-at-top

  device = device or jax.devices()[0]
  sharding = jax.sharding.SingleDeviceSharding(device)
  path = Path(checkpoint).resolve() / "default"
  with ocp.PyTreeCheckpointer() as checkpointer:
    metadata = checkpointer.metadata(path).item_metadata.tree
    network = metadata["params"]["gemma_network"]
    if lora_only:
      from gemma import peft  # pylint: disable=g-import-not-at-top
      network = peft.split_params(network).lora
      if not network:
        raise ValueError(f"No LoRA parameters found in {checkpoint}")
    item = {"params": {"gemma_network": jax.tree.map(
        lambda m: jax.ShapeDtypeStruct(m.shape, m.dtype, sharding=sharding),
        network)}}
    restore_args = jax.tree.map(lambda x: ocp.ArrayRestoreArgs(
        restore_type=jax.Array, dtype=x.dtype, global_shape=x.shape,
        sharding=sharding), item)
    restored = checkpointer.restore(path, args=ocp.args.PyTreeRestore(
        item=item, restore_args=restore_args, partial_restore=True))
  return restored["params"]["gemma_network"]


def save_trajectory(directory, tokens, tokenizer, metadata, *, stop_tokens):
  """Save full token states and exact sequence-decoded text, even when malformed.

  The first stop token in the final canvas fixes the displayed response span
  for every frame. No cell parser, clue repair or thinking-block removal runs.
  Full canvases remain available in tokens.npy and canvas_texts.json.
  """
  directory = Path(directory)
  directory.mkdir(parents=True, exist_ok=True)
  tokens = np.asarray(tokens, dtype=np.int32)
  if tokens.ndim != 2 or not len(tokens):
    raise ValueError("tokens must have shape (states, canvas_length)")
  stops = np.flatnonzero(np.isin(tokens[-1], stop_tokens))
  end = int(stops[0]) if stops.size else tokens.shape[1]
  # Decode complete sequences rather than concatenating individually decoded
  # tokens: SentencePiece whitespace and byte fallback depend on neighbours.
  texts = [tokenizer.decode(row[:end].tolist()) for row in tokens]
  canvas_texts = [tokenizer.decode(row.tolist()) for row in tokens]
  np.save(directory / "tokens.npy", tokens)
  (directory / "texts.json").write_text(
      json.dumps(texts, ensure_ascii=False) + "\n")
  (directory / "canvas_texts.json").write_text(
      json.dumps(canvas_texts, ensure_ascii=False) + "\n")
  manifest = dict(metadata, canvas_states=len(tokens),
                  denoising_updates=len(tokens) - 1, canvas_length=tokens.shape[1],
                  response_start=0, response_end=end, final_text=texts[-1],
                  stop_tokens=[int(t) for t in stop_tokens], tokens="tokens.npy",
                  texts="texts.json", canvas_texts="canvas_texts.json")
  (directory / "manifest.json").write_text(
      json.dumps(manifest, indent=2, ensure_ascii=False) + "\n")
  return manifest


def grid_cells(text: str) -> list[str]:
  """Keep every whitespace-separated field, including invalid values and prose."""
  return text.split()


def format_raw_grid(text: str, size: int) -> str:
  """Reflow whitespace into rows, retaining every field and any overflow rows."""
  cells = grid_cells(text)
  return "\n".join(" ".join(cells[i:i + size])
                   for i in range(0, len(cells), size))


@functools.lru_cache(maxsize=128)
def _font(size: int, *, mono=False):
  from matplotlib import font_manager  # pylint: disable=g-import-not-at-top
  name = "DejaVu Sans Mono" if mono else "DejaVu Sans"
  return ImageFont.truetype(font_manager.findfont(name), size)


def _fit_text(image, text, box, *, max_size=32, mono=False):
  """Draw the entire string in a box, scaling long noisy fields to fit."""
  left, top, right, bottom = box
  draw = ImageDraw.Draw(image)
  font = _font(max_size, mono=mono)
  bounds = draw.multiline_textbbox((0, 0), text or " ", font=font, spacing=5)
  width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
  # Rasterize separately so even very long fields cannot overwrite neighbours.
  layer = Image.new("RGBA", (max(1, width + 4), max(1, height + 4)))
  ImageDraw.Draw(layer).multiline_text(
      (2 - bounds[0], 2 - bounds[1]), text, font=font, fill="#17212b", spacing=5)
  layer.thumbnail((max(1, right - left), max(1, bottom - top)), Image.Resampling.LANCZOS)
  image.paste(layer, (left, top), layer)


def render_grid(task: str, text: str, *, side=720) -> Image.Image:
  """Draw literal model fields with neutral styling and no validity checks.

  Nurikabe '#' alone is sea. All other fields are shown verbatim. Sudoku has
  bold 3x3 boundaries. Missing cells stay empty; overflow stays in the raw view.
  No clues or targets are overlaid on the model's grid.
  """
  size = TASK_SIZES[task]
  cells = grid_cells(text)
  image = Image.new("RGB", (side, side), "white")
  draw = ImageDraw.Draw(image)
  edge = 3
  cell = (side - 2 * edge) / size
  for i, token in enumerate(cells[:size * size]):
    row, col = divmod(i, size)
    x0, y0 = round(edge + col * cell), round(edge + row * cell)
    x1, y1 = round(edge + (col + 1) * cell), round(edge + (row + 1) * cell)
    if task == "nurikabe" and token == "#":
      draw.rectangle((x0, y0, x1, y1), fill="#202124")
    else:
      font = _font(round(cell * 0.5))
      bounds = draw.textbbox((0, 0), token, font=font)
      width, height = bounds[2] - bounds[0], bounds[3] - bounds[1]
      scale = min(1, (cell - 8) / max(1, width), (cell - 8) / max(1, height))
      font = _font(max(1, int(cell * 0.5 * scale)))
      # Clip unusually long fields in their own layer; preserve full raw text.
      layer = Image.new("RGB", (max(1, x1 - x0), max(1, y1 - y0)), "white")
      ImageDraw.Draw(layer).text((layer.width / 2, layer.height / 2), token,
                                font=font, fill="#17212b", anchor="mm")
      image.paste(layer, (x0, y0))
  for i in range(size + 1):
    width = 4 if i in (0, size) or (task == "sudoku" and i % 3 == 0) else 1
    pos = round(edge + i * cell)
    draw.line((edge, pos, side - edge, pos), fill="#202124", width=width)
    draw.line((pos, edge, pos, side - edge), fill="#202124", width=width)
  return image


class PuzzleTrajectory:
  """A saved trajectory; rendering does not import model or dataset packages."""

  def __init__(self, directory):
    self.directory = Path(directory)
    self.info = json.loads((self.directory / "manifest.json").read_text())
    self.task = self.info["task"]
    self.size = TASK_SIZES[self.task]
    self.tokens = np.load(self.directory / self.info["tokens"], allow_pickle=False)
    self.texts = json.loads((self.directory / self.info["texts"]).read_text())
    if self.tokens.ndim != 2 or len(self.texts) != len(self.tokens):
      raise ValueError("Saved text and token trajectories have different shapes")
    self.updates = len(self.texts) - 1

  def render(self, state: int, *, size=(1920, 1080)) -> Image.Image:
    text = self.texts[state]
    cells = grid_cells(text)
    image = Image.new("RGB", (1920, 1080), "#f4f6f8")
    draw = ImageDraw.Draw(image)
    draw.text((64, 42), f"{self.task.title()} sampling trajectory",
              font=_font(48), fill="#17212b")
    draw.text((64, 112), f"Step {state} / {self.updates}   |   "
              f"{len(cells)} fields / {self.size ** 2} grid cells",
              font=_font(28), fill="#505b66")
    draw.text((64, 181), "Sampled string (rows inserted)", font=_font(30), fill="#17212b")
    draw.text((1056, 181), "Sampled grid", font=_font(30), fill="#17212b")
    draw.rounded_rectangle((48, 234, 992, 982), radius=12, fill="white")
    _fit_text(image, format_raw_grid(text, self.size) or "(empty response)",
              (72, 258, 968, 958), max_size=64, mono=True)
    image.paste(render_grid(self.task, text, side=736), (1080, 240))
    draw.text((64, 1012), f"Checkpoint {self.info['checkpoint_step']}   |   "
              f"Held-out example {self.info['example_index']}   |   "
              "Whitespace separates cells; extra fields remain in the string view.",
              font=_font(22), fill="#505b66")
    if size != image.size:
      image = image.resize(size, Image.Resampling.LANCZOS)
    return image

  def interactive_html(self, *, seconds_per_step=0.2) -> str:
    """Self-contained play/pause/slider view with exact escaped raw strings."""
    if not math.isfinite(seconds_per_step) or seconds_per_step <= 0:
      raise ValueError("seconds_per_step must be finite and positive")
    frames = []
    for text in self.texts:
      buffer = io.BytesIO()
      render_grid(self.task, text, side=540).save(buffer, format="PNG")
      frames.append("data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode())
    payload = json.dumps({"raw": [format_raw_grid(t, self.size) for t in self.texts],
                          "exact": self.texts, "grids": frames}, ensure_ascii=True)
    # '<' must be escaped even in application/json script elements.
    payload = payload.replace("<", "\\u003c").replace(">", "\\u003e").replace("&", "\\u0026")
    uid = "puzzle_" + uuid.uuid4().hex
    return f'''<div id="{uid}" style="font:16px system-ui;color:#17212b">
      <button type="button" data-play>Play</button>
      <input data-slider type="range" min="0" max="{self.updates}" value="0"
             aria-label="Sampling step" style="width:60%">
      <span data-step></span>
      <div style="display:flex;gap:24px;align-items:flex-start;flex-wrap:wrap">
        <div style="flex:1;min-width:280px"><h4>Sampled string (rows inserted)</h4>
          <pre data-raw style="overflow:auto;max-height:580px;white-space:pre;font:18px monospace"></pre>
          <details><summary>Exact decoded response (original whitespace)</summary>
            <pre data-exact style="max-height:300px;overflow:auto;white-space:pre-wrap"></pre>
          </details></div>
        <div style="flex:1;min-width:280px"><h4>Sampled grid</h4>
          <img data-grid style="width:100%;max-width:540px" alt="Sampled puzzle grid"></div>
      </div><script type="application/json" data-frames>{payload}</script>
      <script>(function() {{
        const root = document.getElementById('{uid}');
        const frames = JSON.parse(root.querySelector('[data-frames]').textContent);
        const slider = root.querySelector('[data-slider]');
        const button = root.querySelector('[data-play]');
        let timer = null;
        function show() {{
          const n = Number(slider.value);
          root.querySelector('[data-raw]').textContent = frames.raw[n];
          root.querySelector('[data-exact]').textContent = frames.exact[n];
          root.querySelector('[data-grid]').src = frames.grids[n];
          root.querySelector('[data-step]').textContent = 'Step ' + n + ' / {self.updates}';
        }}
        slider.addEventListener('input', show);
        button.addEventListener('click', function() {{
          if (timer !== null) {{ clearInterval(timer); timer = null; button.textContent = 'Play'; }}
          else {{
            button.textContent = 'Pause';
            timer = setInterval(function() {{
              slider.value = (Number(slider.value) + 1) % frames.raw.length; show();
            }}, {round(seconds_per_step * 1000)});
          }}
        }});
        show();
      }})();</script></div>'''

  def export(self, *, prefix=None, seconds_per_step=0.2, fps=30):
    """Write a combined H.264 video, looping GIF, poster and portable HTML."""
    import imageio_ffmpeg  # pylint: disable=g-import-not-at-top

    if (not math.isfinite(seconds_per_step) or seconds_per_step < 1 / fps):
      raise ValueError("seconds_per_step must be finite and at least one frame")
    prefix = prefix or f"{self.task}_denoising"
    if not prefix or Path(prefix).name != prefix or prefix in (".", ".."):
      raise ValueError("prefix must be a filename without directories")
    paths = {kind: self.directory / f"{prefix}.{kind}" for kind in ("mp4", "gif", "html")}
    paths["poster"] = self.directory / f"{prefix}_poster.png"
    self.render(self.updates).save(paths["poster"])
    paths["html"].write_text("<!doctype html><meta charset='utf-8'><title>"
        + html.escape(self.task.title()) + " sampling</title>"
        + self.interactive_html(seconds_per_step=seconds_per_step))
    writer = imageio_ffmpeg.write_frames(
        str(paths["mp4"]), (1920, 1080), fps=fps, codec="libx264",
        pix_fmt_out="yuv420p", macro_block_size=1, ffmpeg_log_level="error",
        output_params=["-crf", "18", "-profile:v", "high", "-movflags", "+faststart",
                       "-color_primaries", "bt709", "-color_trc", "bt709", "-colorspace", "bt709"])
    writer.send(None)
    gif_frames, gif_durations = [], []
    video_frames = 0
    try:
      for state in range(len(self.texts)):
        frame = self.render(state)
        count = max(1, round((state + 1) * seconds_per_step * fps)
                    - round(state * seconds_per_step * fps))
        count += fps if state == 0 else 3 * fps if state == self.updates else 0
        for _ in range(count):
          writer.send(np.asarray(frame, dtype=np.uint8))
        video_frames += count
        gif_frames.append(frame.resize((960, 540), Image.Resampling.LANCZOS)
                          .quantize(colors=256))
        gif_durations.append(max(10, round(count / fps * 100) * 10))
    finally:
      writer.close()
    gif_frames[0].save(paths["gif"], save_all=True, append_images=gif_frames[1:],
                       duration=gif_durations, loop=0, optimize=False)
    settings = {"fps": fps, "seconds_per_step": seconds_per_step,
                "initial_hold_seconds": 1, "final_hold_seconds": 3,
                "video_frames": video_frames, "duration_seconds": video_frames / fps,
                "files": {key: path.name for key, path in paths.items()}}
    (self.directory / "export.json").write_text(json.dumps(settings, indent=2) + "\n")
    return paths
