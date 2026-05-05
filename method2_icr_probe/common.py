"""Shared utilities for Method 2 ICR Probe runs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiment_utils import get_best_available_device, load_feature_cache, save_feature_cache


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DEFAULT_DATA_FILE = ROOT / "data" / "dataset.csv"
DEFAULT_CACHE_FILE = ROOT / "method2_icr_probe" / "artifacts" / "cache" / "method2_icr_cache.npz"
DEFAULT_OUTPUT_FILE = ROOT / "method2_icr_probe" / "artifacts" / "method2_results.json"
DEFAULT_TOP_K = 10
DEFAULT_BATCH_SIZE = 1
DEFAULT_MLP_HIDDEN_DIMS = (32,)

USER_START_MARKER = "<|im_start|>user\n"
USER_END_MARKER = "\n<|im_end|>\n<|im_start|>assistant\n"


def maybe_take_subset(df: pd.DataFrame, subset_size: int | None) -> pd.DataFrame:
    """Take a roughly stratified subset for smoke tests."""
    if subset_size is None or subset_size >= len(df):
        return df.reset_index(drop=True)

    parts = []
    label_counts = df["label"].value_counts(normalize=True).sort_index()
    for label, frac in label_counts.items():
        n_label = max(1, int(round(subset_size * frac)))
        label_df = df[df["label"] == label]
        parts.append(label_df.sample(n=min(n_label, len(label_df)), random_state=42))
    return pd.concat(parts, axis=0).sample(frac=1.0, random_state=42).reset_index(drop=True)


def parse_hidden_dims(hidden_dims_arg: str) -> tuple[int, ...]:
    """Parse a comma-separated hidden-dim string."""
    hidden_dims = tuple(int(item) for item in hidden_dims_arg.split(",") if item.strip())
    if not hidden_dims:
        raise ValueError("hidden_dims must contain at least one width, e.g. 32 or 64,32")
    return hidden_dims


def format_hidden_dims(hidden_dims: tuple[int, ...]) -> str:
    """Format hidden dims for logs and metadata."""
    return ",".join(str(width) for width in hidden_dims)


def get_icr_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL_NAME,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load Qwen with eager attention so per-layer attentions are available."""
    print(f"[Method 2] Loading '{model_name}' with eager attention ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
    )
    model.eval()
    return model, tokenizer


def _split_prompt_for_icr(prompt: str) -> tuple[str, str, str]:
    """Split ChatML prompt into prefix, user body, and assistant prefix."""
    user_marker_idx = prompt.find(USER_START_MARKER)
    assistant_marker_idx = prompt.rfind(USER_END_MARKER)

    if (
        user_marker_idx == -1
        or assistant_marker_idx == -1
        or assistant_marker_idx < user_marker_idx
    ):
        return "", prompt, ""

    user_body_start = user_marker_idx + len(USER_START_MARKER)
    prefix = prompt[:user_body_start]
    user_body = prompt[user_body_start:assistant_marker_idx]
    assistant_prefix = prompt[assistant_marker_idx:]
    return prefix, user_body, assistant_prefix


def _combine_prompt_and_response(
    prompt_ids: list[int],
    response_ids: list[int],
    max_length: int,
) -> tuple[list[int], int, int]:
    """Assemble a sequence while preserving the response tail."""
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


def build_icr_batch(
    tokenizer,
    prompts: list[str],
    responses: list[str],
    max_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, tuple[int, int]]]]:
    """Tokenize a batch while preserving the response tail and ChatML user span."""
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    split_prompts = [_split_prompt_for_icr(prompt) for prompt in prompts]
    prompt_prefixes = [parts[0] for parts in split_prompts]
    user_bodies = [parts[1] for parts in split_prompts]
    prompt_suffixes = [parts[2] for parts in split_prompts]

    prefix_tokens = tokenizer(prompt_prefixes, add_special_tokens=False)
    user_tokens = tokenizer(user_bodies, add_special_tokens=False)
    suffix_tokens = tokenizer(prompt_suffixes, add_special_tokens=False)
    response_tokens = tokenizer(responses, add_special_tokens=False)

    input_ids_list: list[list[int]] = []
    spans: list[dict[str, tuple[int, int]]] = []

    for prefix_ids, user_ids, suffix_ids, response_ids in zip(
        prefix_tokens["input_ids"],
        user_tokens["input_ids"],
        suffix_tokens["input_ids"],
        response_tokens["input_ids"],
    ):
        prompt_ids = prefix_ids + user_ids + suffix_ids
        input_ids, response_start, response_end = _combine_prompt_and_response(
            prompt_ids=prompt_ids,
            response_ids=response_ids,
            max_length=max_length,
        )

        kept_prompt_len = response_start
        dropped_prompt_tokens = len(prompt_ids) - kept_prompt_len
        user_start_raw = len(prefix_ids)
        user_end_raw = len(prefix_ids) + len(user_ids)

        user_start = max(0, user_start_raw - dropped_prompt_tokens)
        user_end = max(0, min(kept_prompt_len, user_end_raw - dropped_prompt_tokens))

        if user_end <= user_start and kept_prompt_len > 0:
            user_start = 0
            user_end = kept_prompt_len

        input_ids_list.append(input_ids)
        spans.append(
            {
                "user_span": (user_start, user_end),
                "response_span": (response_start, response_end),
            }
        )

    batch = tokenizer.pad(
        {"input_ids": input_ids_list},
        padding=True,
        return_attention_mask=True,
        return_tensors="pt",
    )
    return (
        batch["input_ids"].to(device),
        batch["attention_mask"].to(device),
        spans,
    )


def _standardize(values: torch.Tensor) -> torch.Tensor:
    """Match the repo's z-score-before-softmax step, with stable fallbacks."""
    if values.numel() <= 1:
        return torch.zeros_like(values)

    std = values.std(unbiased=True)
    if not torch.isfinite(std) or float(std) < 1e-8:
        std = values.std(unbiased=False)
    std = torch.clamp(std, min=1e-8)
    return (values - values.mean()) / std


def _kl_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    p = torch.clamp(p, min=1e-12)
    q = torch.clamp(q, min=1e-12)
    return torch.sum(p * torch.log(p / q))


def _js_divergence(
    hidden_scores: torch.Tensor,
    attention_scores: torch.Tensor,
    z_normalize: bool,
) -> torch.Tensor:
    """Compute JS divergence between the two top-k distributions."""
    if z_normalize:
        hidden_scores = _standardize(hidden_scores)
        attention_scores = _standardize(attention_scores)

    p = F.softmax(hidden_scores, dim=0)
    q = F.softmax(attention_scores, dim=0)
    m = 0.5 * (p + q)
    return 0.5 * _kl_divergence(p, m) + 0.5 * _kl_divergence(q, m)


def _mask_attention_row(
    attention_row: torch.Tensor,
    user_span: tuple[int, int],
    response_start: int,
) -> torch.Tensor:
    """Zero all attention outside the user-message span and response span."""
    user_start, user_end = user_span
    mask = torch.zeros_like(attention_row, dtype=torch.bool)
    if user_end > user_start:
        mask[user_start:user_end] = True
    if response_start < attention_row.numel():
        mask[response_start:] = True

    masked_row = torch.zeros_like(attention_row)
    masked_row[mask] = attention_row[mask]
    return masked_row


def compute_sample_icr_vector(
    hidden_states: list[torch.Tensor],
    attentions: list[torch.Tensor],
    user_span: tuple[int, int],
    response_span: tuple[int, int],
    top_k: int = DEFAULT_TOP_K,
    z_normalize: bool = True,
) -> torch.Tensor:
    """Compute the adapted ICR vector for one sample.

    The returned vector has one value per transformer layer: mean token-wise ICR
    score over the response span.
    """
    response_start, response_end = response_span
    n_layers = len(attentions)
    icr_vector = hidden_states[0].new_zeros(n_layers)

    for layer_idx in range(n_layers):
        pooled_attention = attentions[layer_idx].mean(dim=0)
        previous_layer_states = hidden_states[layer_idx]
        current_layer_states = hidden_states[layer_idx + 1]

        token_scores: list[torch.Tensor] = []
        for token_idx in range(response_start, response_end):
            attention_row = pooled_attention[token_idx]
            masked_attention_row = _mask_attention_row(
                attention_row=attention_row,
                user_span=user_span,
                response_start=response_start,
            )
            current_top_k = min(top_k, masked_attention_row.numel())
            top_attention, top_indices = torch.topk(masked_attention_row, k=current_top_k)

            residual_update = current_layer_states[token_idx] - previous_layer_states[token_idx]
            attended_hidden_states = previous_layer_states.index_select(0, top_indices)
            hidden_norms = torch.linalg.vector_norm(attended_hidden_states, dim=-1).clamp_min(1e-8)
            projection_scores = (attended_hidden_states * residual_update.unsqueeze(0)).sum(dim=-1)
            projection_scores = projection_scores / hidden_norms

            token_scores.append(
                _js_divergence(
                    hidden_scores=projection_scores,
                    attention_scores=top_attention,
                    z_normalize=z_normalize,
                )
            )

        if token_scores:
            icr_vector[layer_idx] = torch.stack(token_scores).mean()

    return icr_vector


def extract_icr_feature_cache(
    df: pd.DataFrame,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_length: int = 512,
    top_k: int = DEFAULT_TOP_K,
    z_normalize: bool = True,
    device: torch.device | None = None,
    cache_dtype: np.dtype = np.float32,
) -> dict[str, np.ndarray]:
    """Extract reduced ICR vectors for all rows in *df*."""
    if device is None:
        device = get_best_available_device()

    model, tokenizer = get_icr_model_and_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device)

    n_samples = len(df)
    n_layers = model.config.num_hidden_layers
    prompts = df["prompt"].tolist()
    responses = df["response"].tolist()

    cache: dict[str, np.ndarray] = {
        "icr_vector": np.empty((n_samples, n_layers), dtype=cache_dtype),
        "prompt_token_length": np.empty(n_samples, dtype=np.int32),
        "response_token_length": np.empty(n_samples, dtype=np.int32),
        "top_k": np.asarray([top_k], dtype=np.int32),
        "z_normalize": np.asarray([int(z_normalize)], dtype=np.int8),
        "max_length": np.asarray([max_length], dtype=np.int32),
    }

    for start in tqdm(range(0, n_samples, batch_size), desc="Caching Method 2 ICR", unit="batch"):
        batch_prompts = prompts[start : start + batch_size]
        batch_responses = responses[start : start + batch_size]
        input_ids, attention_mask, spans = build_icr_batch(
            tokenizer=tokenizer,
            prompts=batch_prompts,
            responses=batch_responses,
            max_length=max_length,
            device=device,
        )

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                output_attentions=True,
                output_hidden_states=True,
            )

        for sample_idx in range(input_ids.size(0)):
            row_idx = start + sample_idx
            seq_len = int(attention_mask[sample_idx].sum().item())
            sample_hidden_states = [
                layer[sample_idx, :seq_len].detach().to(dtype=torch.float32)
                for layer in outputs.hidden_states
            ]
            sample_attentions = [
                layer[sample_idx, :, :seq_len, :seq_len].detach().to(dtype=torch.float32)
                for layer in outputs.attentions
            ]
            sample_spans = spans[sample_idx]
            response_start, response_end = sample_spans["response_span"]
            response_end = min(response_end, seq_len)
            response_span = (response_start, response_end)

            icr_vector = compute_sample_icr_vector(
                hidden_states=sample_hidden_states,
                attentions=sample_attentions,
                user_span=sample_spans["user_span"],
                response_span=response_span,
                top_k=top_k,
                z_normalize=z_normalize,
            )

            cache["icr_vector"][row_idx] = icr_vector.cpu().numpy().astype(cache_dtype, copy=False)
            cache["prompt_token_length"][row_idx] = response_start
            cache["response_token_length"][row_idx] = response_end - response_start

        del outputs
        del input_ids
        del attention_mask
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if "label" in df.columns and df["label"].notna().all():
        cache["labels"] = df["label"].astype(int).to_numpy(dtype=np.int32)

    return cache


def build_feature_matrix(cache: dict[str, np.ndarray]) -> np.ndarray:
    """Return the Method 2 feature matrix."""
    return cache["icr_vector"].astype(np.float32, copy=False)


def load_or_build_cache(
    df: pd.DataFrame,
    cache_file: Path,
    data_file: Path,
    batch_size: int,
    max_length: int,
    top_k: int,
    z_normalize: bool,
    cache_dtype: str,
    overwrite_cache: bool,
    subset_size: int | None,
) -> dict[str, np.ndarray]:
    """Load the cached ICR vectors or build them once."""
    should_rebuild_cache = overwrite_cache or subset_size is not None or not cache_file.exists()

    if not should_rebuild_cache:
        print(f"[Method 2] Loading cache from {cache_file}")
        cache = load_feature_cache(cache_file)
        labels_match = "labels" in cache and len(cache["labels"]) == len(df)
        top_k_match = int(cache.get("top_k", np.asarray([-1], dtype=np.int32))[0]) == top_k
        z_norm_match = int(cache.get("z_normalize", np.asarray([-1], dtype=np.int8))[0]) == int(z_normalize)
        max_length_match = int(cache.get("max_length", np.asarray([-1], dtype=np.int32))[0]) == max_length
        if not (labels_match and top_k_match and z_norm_match and max_length_match):
            print("[Method 2] Cache metadata mismatch detected. Rebuilding cache.")
            should_rebuild_cache = True

    if should_rebuild_cache:
        print(f"[Method 2] Building cache from {data_file}")
        cache = extract_icr_feature_cache(
            df=df,
            batch_size=batch_size,
            max_length=max_length,
            top_k=top_k,
            z_normalize=z_normalize,
            cache_dtype=np.float16 if cache_dtype == "float16" else np.float32,
        )
        if subset_size is None:
            save_feature_cache(cache_file, cache)
            print(f"[Method 2] Saved cache to {cache_file}")
        else:
            print("[Method 2] Subset run detected. Cache was not persisted.")
        return cache

    return cache
