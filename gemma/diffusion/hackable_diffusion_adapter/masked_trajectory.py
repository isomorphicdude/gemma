"""Record real LLaDA masked-diffusion states for the trajectory video renderer.

Uses one bidirectional canvas with either LLaDA's confidence-based transfer or
MD4's linear ancestral reveal schedule (https://arxiv.org/abs/2406.04329).
Requires a separate PyTorch/Transformers environment; the Gemma JAX environment
is unchanged.
"""

import argparse
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
import transformers
from transformers import AutoModel, AutoTokenizer


MODEL_ID = "GSAI-ML/LLaDA-8B-Instruct"
MODEL_REVISION = "08b83a6feb34df1a6011b80c3c00c7563e963b07"
DEFAULT_PROMPT = "Why is the sky blue? Explain in one short paragraph."


@torch.inference_mode()
def sample_tokens(model, prompt_tokens, *, mask_id, canvas_length, steps,
                  temperature=0.0, sampler="confidence"):
  """Return the all-mask start and every actual post-update canvas on CPU.

  Confidence sampling reveals a fixed number of highest-confidence masks.
  MD4 linear sampling uses alpha(t)=1-t and a uniform reverse-time grid:
  each remaining mask reveals with probability 1/(steps-completed_steps).
  Positive temperatures sample the joint token/mask categorical distribution;
  temperature zero keeps random reveals but uses greedy token predictions.
  Revealed tokens stay fixed. Only token IDs are retained on the host.
  """
  if not 1 <= steps <= canvas_length:
    raise ValueError("steps must be between 1 and canvas_length")
  if not math.isfinite(temperature) or temperature < 0:
    raise ValueError("temperature must be finite and nonnegative")
  if sampler not in ("confidence", "md4-linear"):
    raise ValueError(f"Unknown sampler: {sampler}")
  if prompt_tokens.ndim != 2 or prompt_tokens.shape[0] != 1:
    raise ValueError("Expected one unpadded prompt, with shape (1, length)")
  if (prompt_tokens == mask_id).any():
    raise ValueError("The prompt must not contain the mask token")
  prompt_length = prompt_tokens.shape[1]
  x = torch.full((1, prompt_length + canvas_length), mask_id,
                 dtype=torch.long, device=prompt_tokens.device)
  x[:, :prompt_length] = prompt_tokens

  def snapshot():
    return x[0, prompt_length:].cpu().numpy().astype(np.int32, copy=True)

  states = [snapshot()]
  base, remainder = divmod(canvas_length, steps)
  for step in range(steps):
    # All response positions attend to each other, including unrevealed masks.
    logits = model(x).logits[:, prompt_length:, :].float()
    logits[..., mask_id] = -torch.inf
    response = x[:, prompt_length:]
    if sampler == "md4-linear":
      # MD4: p(reveal | masked)=(alpha(s)-alpha(t))/(1-alpha(t)).
      # On a uniform grid with alpha(t)=1-t this is 1/(steps-step).
      # It is independent of vocabulary confidence, including EOS padding.
      reveal_probability = 1.0 / (steps - step)
      if temperature:
        probabilities = torch.softmax(logits / temperature, dim=-1)
        probabilities.mul_(reveal_probability)
        remain_masked = torch.full(
            (*response.shape, 1), 1.0 - reveal_probability,
            device=x.device, dtype=probabilities.dtype,
        )
        # An extra category represents remaining masked, separate from the
        # model's vocabulary mask token (whose prediction is excluded).
        joint = torch.cat((probabilities, remain_masked), dim=-1)
        draws = torch.multinomial(joint[0], num_samples=1).T
        predictions = torch.where(draws == logits.shape[-1], mask_id, draws)
        response.copy_(torch.where(response == mask_id, predictions, response))
        del probabilities, remain_masked, joint, draws, predictions
      else:
        # Greedy-token control isolates the effect of changing reveal order.
        reveal = ((torch.rand(response.shape, device=x.device) < reveal_probability)
                  & (response == mask_id))
        response.copy_(torch.where(reveal, logits.argmax(dim=-1), response))
    elif temperature:
      # The authors' float64 Gumbel-max parameterization.
      scores = logits.double().exp() / (
          -torch.rand_like(logits, dtype=torch.float64).log()
      ).pow(temperature)
      predictions = scores.argmax(dim=-1)
      del scores
    else:
      predictions = logits.argmax(dim=-1)
    if sampler == "confidence":
      confidence = (
          logits.gather(-1, predictions.unsqueeze(-1)).squeeze(-1)
          - torch.logsumexp(logits, dim=-1)
      )
      confidence.masked_fill_(response != mask_id, -torch.inf)
      count = base + (step < remainder)
      indices = confidence.topk(count, dim=-1).indices
      response.scatter_(1, indices, predictions.gather(1, indices))
      del confidence, predictions
    states.append(snapshot())
    del logits
    print(f"Step {step + 1}/{steps}: {(states[-1] == mask_id).sum()} masks remain",
          flush=True)
  return np.stack(states)


def main():
  parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
  parser.add_argument("--model-path", default=MODEL_ID,
                      help="HF model ID or a directory downloaded with hf download")
  parser.add_argument("--output-dir", type=Path,
                      help="saved run directory (default depends on the selected sampler)")
  parser.add_argument("--prompt", default=DEFAULT_PROMPT)
  parser.add_argument("--seed", type=int, default=42)
  parser.add_argument("--canvas-length", type=int, default=256)
  parser.add_argument("--steps", type=int, default=128,
                      help="number of model calls; saved states include the start")
  parser.add_argument("--sampler", choices=("confidence", "md4-linear"),
                      default="confidence",
                      help="confidence-ranked transfer or MD4 linear ancestral sampling")
  parser.add_argument("--temperature", type=float,
                      help="token temperature (default: 0 for confidence, 1 for md4-linear)")
  args = parser.parse_args()
  if args.temperature is None:
    args.temperature = 1.0 if args.sampler == "md4-linear" else 0.0
  if args.output_dir is None:
    args.output_dir = Path("outputs/llada_md4_linear_trajectory"
                           if args.sampler == "md4-linear" else "outputs/llada_trajectory")
  if (not 1 <= args.steps <= args.canvas_length
      or not math.isfinite(args.temperature) or args.temperature < 0):
    parser.error("Require 1 <= steps <= canvas-length and finite temperature >= 0")
  if not torch.cuda.is_available():
    raise RuntimeError("Sampling requires a CUDA GPU; CPU fallback is disabled")
  torch.manual_seed(args.seed)
  model_kwargs = {"trust_remote_code": True}
  local_model = Path(args.model_path).is_dir()
  if not local_model:
    model_kwargs["revision"] = MODEL_REVISION
  tokenizer = AutoTokenizer.from_pretrained(args.model_path, **model_kwargs)
  model = AutoModel.from_pretrained(
      args.model_path, torch_dtype=torch.bfloat16, device_map="cuda",
      **model_kwargs,
  ).eval()
  mask_id = model.config.mask_token_id
  stop_tokens = [tokenizer.eos_token_id, tokenizer.convert_tokens_to_ids("<|eot_id|>")]
  stop_tokens = sorted({int(token) for token in stop_tokens if token is not None})
  formatted = tokenizer.apply_chat_template(
      [{"role": "user", "content": args.prompt}],
      add_generation_prompt=True, tokenize=False,
  )
  prompt_tokens = tokenizer(formatted, add_special_tokens=False,
                            return_tensors="pt").input_ids.to("cuda")
  print(f"Loaded {MODEL_ID} on {torch.cuda.get_device_name()}; sampling...", flush=True)
  torch.cuda.synchronize()
  started = time.monotonic()
  states = sample_tokens(
      model, prompt_tokens, mask_id=mask_id, canvas_length=args.canvas_length,
      steps=args.steps, temperature=args.temperature,
      sampler=args.sampler,
  )
  torch.cuda.synchronize()
  seconds = time.monotonic() - started
  # Check the defining invariants against the actual model trajectory.
  assert np.all(states[0] == mask_id)
  assert not np.any(states[-1] == mask_id)
  assert np.all((states[:-1] == mask_id) | (states[1:] == states[:-1]))
  stops = np.flatnonzero(np.isin(states[-1], stop_tokens))
  response_end = int(stops[0]) if stops.size else args.canvas_length
  if response_end == 0:
    raise RuntimeError("Model produced an empty response; try another prompt or seed")
  response_unmasked = (states[:, :response_end] != mask_id).sum(axis=1)
  final_text = tokenizer.decode(states[-1, :response_end].tolist(),
                                 clean_up_tokenization_spaces=False)
  token_text = {
      int(token): tokenizer.decode([int(token)], clean_up_tokenization_spaces=False)
      for token in np.unique(states)
  }
  output_dir = args.output_dir.resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  np.save(output_dir / "tokens.npy", states)
  (output_dir / "token_text.json").write_text(
      json.dumps(token_text, ensure_ascii=False) + "\n"
  )
  revision = MODEL_REVISION
  if local_model:
    metadata_path = (Path(args.model_path) / ".cache/huggingface/download/config.json.metadata")
    revision = (metadata_path.read_text().splitlines()[0] if metadata_path.is_file()
                else getattr(model.config, "_commit_hash", None))
  manifest = {
      "model": "LLaDA 8B Instruct", "model_id": MODEL_ID,
      "model_revision": revision,
      "checkpoint": args.model_path, "prompt": args.prompt, "seed": args.seed,
      "canvas_length": args.canvas_length, "steps": args.steps,
      "model_calls": args.steps, "canvas_states": len(states),
      "temperature": args.temperature, "process": "masked", "mask_token_id": mask_id,
      "corruption": "absorbing mask process, linear schedule",
      "sampler": ("MD4 linear ancestral reveal schedule" if args.sampler == "md4-linear"
                  else "whole-canvas confidence-based unmasking; no classifier-free guidance"),
      "sampling_policy": args.sampler, "alpha_schedule": "1-t", "time_grid": "uniform",
      "position_selection": ("independent random reveals" if args.sampler == "md4-linear"
                             else "highest token confidence"),
      "token_sampling": "categorical" if args.temperature else "greedy",
      "schedule_source": "https://arxiv.org/abs/2406.04329",
      "source": ("https://arxiv.org/html/2406.04329v4#S4" if args.sampler == "md4-linear"
                 else "https://github.com/ML-GSAI/LLaDA/blob/main/generate.py"),
      "stop_tokens": stop_tokens, "response_start": 0, "response_end": response_end,
      "final_text": final_text, "sampling_seconds": round(seconds, 2),
      "mask_counts": (states == mask_id).sum(axis=1).tolist(),
      "response_unmasked_counts": response_unmasked.tolist(),
      "torch": torch.__version__, "device": str(model.device),
      "transformers": transformers.__version__,
      "gpu": torch.cuda.get_device_name(),
      "peak_gpu_memory_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
      "tokens": "tokens.npy", "token_text": "token_text.json",
  }
  (output_dir / "manifest.json").write_text(
      json.dumps(manifest, indent=2, ensure_ascii=False) + "\n"
  )
  print(f"Saved {len(states)} states in {seconds:.1f}s to {output_dir}\n\n{final_text}",
        flush=True)


if __name__ == "__main__":
  main()
