"""
experiment_utils.py — reusable feature extraction helpers for method sweeps.

These utilities are intentionally separate from the competition starter
pipeline. They support faster experimentation by caching compact hidden-state
summaries that can be reused across multiple methods from
``METHODS_DETAILED.md``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from model import get_model_and_tokenizer

SEQUENCE_MODES: tuple[str, ...] = ("last_token", "response_last", "response_mean")


def get_best_available_device() -> torch.device:
    """Return the best local inference device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _combine_prompt_and_response(
    prompt_ids: list[int],
    response_ids: list[int],
    max_length: int,
) -> tuple[list[int], int, int]:
    """Assemble a sequence while preserving the response tail.

    We keep the full response whenever possible and crop the *start* of the
    prompt first. This is better suited for hallucination analysis than the
    default tokenizer truncation, which would otherwise remove the end of the
    model's answer.
    """
    if not response_ids:
        raise ValueError("Encountered an empty tokenized response.")

    if len(response_ids) >= max_length:
        kept_response = response_ids[-max_length:]
        return kept_response, 0, len(kept_response)

    prompt_budget = max_length - len(response_ids)
    kept_prompt = prompt_ids[-prompt_budget:] if prompt_budget > 0 else []
    input_ids = kept_prompt + response_ids
    response_start = len(kept_prompt)
    response_end = len(input_ids)
    return input_ids, response_start, response_end


def build_response_preserving_batch(
    tokenizer,
    prompts: list[str],
    responses: list[str],
    max_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[tuple[int, int]]]:
    """Tokenize a batch while preserving the response tokens."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    prompt_tokens = tokenizer(prompts, add_special_tokens=False)
    response_tokens = tokenizer(responses, add_special_tokens=False)

    input_ids_list: list[list[int]] = []
    response_spans: list[tuple[int, int]] = []

    for prompt_ids, response_ids in zip(
        prompt_tokens["input_ids"],
        response_tokens["input_ids"],
    ):
        input_ids, response_start, response_end = _combine_prompt_and_response(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            max_length=max_length,
        )
        input_ids_list.append(input_ids)
        response_spans.append((response_start, response_end))

    batch = tokenizer.pad(
        {"input_ids": input_ids_list},
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    return (
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        response_spans,
    )


def _get_response_content_bounds(response_start: int, response_end: int) -> tuple[int, int]:
    """Return the token span that excludes the terminal EOS token when present."""
    if response_end - response_start <= 1:
        return response_start, response_end
    return response_start, response_end - 1


def _compute_spectrum_features(
    layerwise_response_states: torch.Tensor,
    top_k: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute normalized singular values and log-det scores for each layer.

    Args:
        layerwise_response_states: Tensor of shape ``(n_layers, n_tokens, hidden_dim)``.
        top_k: Number of singular values to retain per layer.
    """
    n_layers = layerwise_response_states.size(0)
    spectra = []
    logdets = []

    for layer_idx in range(n_layers):
        layer_states = layerwise_response_states[layer_idx]
        if layer_states.size(0) < 2:
            spectra.append(layer_states.new_zeros(top_k))
            logdets.append(layer_states.new_tensor(0.0))
            continue

        centered = layer_states - layer_states.mean(dim=0, keepdim=True)
        singular_values = torch.linalg.svdvals(centered)

        spectrum = layer_states.new_zeros(top_k)
        retained = singular_values[:top_k]
        if retained.numel() > 0:
            spectrum[: retained.numel()] = retained / (retained.sum() + 1e-8)
        spectra.append(spectrum)

        eigenvalues = (singular_values ** 2) / max(centered.size(0), 1)
        logdet = torch.log(eigenvalues + 1e-6).sum()
        logdets.append(logdet)

    return torch.stack(spectra, dim=0), torch.stack(logdets, dim=0)


def extract_feature_cache(
    df: pd.DataFrame,
    batch_size: int = 2,
    max_length: int = 512,
    spectrum_top_k: int = 16,
    device: torch.device | None = None,
    cache_dtype: np.dtype = np.float16,
    include_icr: bool = False,
    include_spectrum: bool = False,
) -> dict[str, np.ndarray]:
    """Extract reusable hidden-state summaries for all rows in *df*."""
    if device is None:
        device = get_best_available_device()

    model, tokenizer = get_model_and_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)
    n_samples = len(df)
    n_layers = model.config.num_hidden_layers + 1
    hidden_dim = model.config.hidden_size

    prompts = df["prompt"].tolist()
    responses = df["response"].tolist()

    cache: dict[str, np.ndarray] = {
        "last_token": np.empty((n_samples, n_layers, hidden_dim), dtype=cache_dtype),
        "response_last": np.empty((n_samples, n_layers, hidden_dim), dtype=cache_dtype),
        "response_mean": np.empty((n_samples, n_layers, hidden_dim), dtype=cache_dtype),
        "prompt_token_length": np.empty(n_samples, dtype=np.int32),
        "response_token_length": np.empty(n_samples, dtype=np.int32),
        "response_truncated": np.empty(n_samples, dtype=np.int8),
    }
    if include_icr:
        cache["icr_norms"] = np.empty((n_samples, n_layers - 1), dtype=cache_dtype)
        cache["icr_cosines"] = np.empty((n_samples, n_layers - 2), dtype=cache_dtype)
    if include_spectrum:
        cache["spectrum"] = np.empty((n_samples, n_layers, spectrum_top_k), dtype=cache_dtype)
        cache["spectrum_logdet"] = np.empty((n_samples, n_layers), dtype=cache_dtype)

    for start in tqdm(range(0, len(df), batch_size), desc="Caching features", unit="batch"):
        batch_prompts = prompts[start : start + batch_size]
        batch_responses = responses[start : start + batch_size]
        input_ids, attention_mask, response_spans = build_response_preserving_batch(
            tokenizer=tokenizer,
            prompts=batch_prompts,
            responses=batch_responses,
            max_length=max_length,
            device=device,
        )

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

        # Move layer outputs to CPU one at a time to avoid a large float32 stack
        # on the GPU before the copy happens.
        hidden = torch.stack(
            [layer.detach().to(device="cpu", dtype=torch.float32) for layer in outputs.hidden_states],
            dim=1,
        )
        batch_attention_mask = attention_mask.cpu()

        for sample_idx in range(hidden.size(0)):
            row_idx = start + sample_idx
            sample_hidden = hidden[sample_idx]
            seq_len = int(batch_attention_mask[sample_idx].sum().item())
            last_token_idx = seq_len - 1

            response_start, response_end = response_spans[sample_idx]
            response_end = min(response_end, seq_len)
            response_content_start, response_content_end = _get_response_content_bounds(
                response_start=response_start,
                response_end=response_end,
            )

            response_states = sample_hidden[:, response_content_start:response_content_end, :]
            if response_states.size(1) == 0:
                response_states = sample_hidden[:, response_start:response_end, :]

            last_token = sample_hidden[:, last_token_idx, :]
            response_last = response_states[:, -1, :]
            response_mean = response_states.mean(dim=1)

            cache["last_token"][row_idx] = last_token.numpy().astype(cache_dtype, copy=False)
            cache["response_last"][row_idx] = response_last.numpy().astype(cache_dtype, copy=False)
            cache["response_mean"][row_idx] = response_mean.numpy().astype(cache_dtype, copy=False)
            cache["prompt_token_length"][row_idx] = response_start
            cache["response_token_length"][row_idx] = response_states.size(1)
            cache["response_truncated"][row_idx] = int(response_start == 0)

            if include_icr:
                deltas = response_last[1:] - response_last[:-1]
                icr_norms = torch.linalg.vector_norm(deltas, dim=-1)

                if deltas.size(0) < 2:
                    icr_cosines = deltas.new_zeros(0)
                else:
                    icr_cosines = torch.nn.functional.cosine_similarity(
                        deltas[1:],
                        deltas[:-1],
                        dim=-1,
                        eps=1e-8,
                    )
                cache["icr_norms"][row_idx] = icr_norms.numpy().astype(cache_dtype, copy=False)
                cache["icr_cosines"][row_idx] = icr_cosines.numpy().astype(cache_dtype, copy=False)

            if include_spectrum:
                spectra, spectrum_logdet = _compute_spectrum_features(
                    layerwise_response_states=response_states,
                    top_k=spectrum_top_k,
                )
                cache["spectrum"][row_idx] = spectra.numpy().astype(cache_dtype, copy=False)
                cache["spectrum_logdet"][row_idx] = spectrum_logdet.numpy().astype(cache_dtype, copy=False)

        del hidden
        del outputs
        del batch_attention_mask
        del input_ids
        del attention_mask
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if "label" in df.columns and df["label"].notna().all():
        cache["labels"] = df["label"].astype(int).to_numpy(dtype=np.int32)

    return cache


def save_feature_cache(
    output_file: str | Path,
    cache: dict[str, np.ndarray],
) -> None:
    """Persist a feature cache as a compressed ``.npz`` file."""
    output_path = Path(output_file)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **cache)


def load_feature_cache(
    input_file: str | Path,
) -> dict[str, np.ndarray]:
    """Load a feature cache from disk."""
    with np.load(Path(input_file), allow_pickle=False) as data:
        return {key: data[key] for key in data.files}
