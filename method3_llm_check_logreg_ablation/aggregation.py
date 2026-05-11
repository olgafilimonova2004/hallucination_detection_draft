"""Feature-cache loading and feature-set assembly for the Method 3 log-reg study."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from experiment_utils import (
    build_response_preserving_batch,
    get_best_available_device,
    load_feature_cache,
    save_feature_cache,
)
from method3_llm_check.common import (
    DEFAULT_LOGIT_METRIC_NAMES,
    FEATURE_COMPONENT_LABELS,
    FEATURE_SET_COMPONENTS,
    FEATURE_SET_DESCRIPTIONS,
    _compute_attention_layer_score,
    _compute_hidden_layer_score,
    _compute_logit_metrics,
)


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL_NAME = "Qwen/Qwen2.5-0.5B"
DEFAULT_DATA_FILE = ROOT / "data" / "dataset.csv"
DEFAULT_CACHE_FILE = (
    ROOT
    / "method3_llm_check_logreg_ablation"
    / "artifacts"
    / "cache"
    / "method3_llm_check_logreg_cache_float32.npz"
)
DEFAULT_OUTPUT_FILE = (
    ROOT / "method3_llm_check_logreg_ablation" / "artifacts" / "results.json"
)
DEFAULT_ABLATION_DIR = (
    ROOT / "method3_llm_check_logreg_ablation" / "artifacts" / "ablation"
)
DEFAULT_BATCH_SIZE = 1
DEFAULT_FEATURE_SET = "logit_hidden"
DEFAULT_FEATURE_SETS = tuple(FEATURE_SET_COMPONENTS.keys())

REQUIRED_CACHE_KEYS = {
    "logit_metrics_vector",
    "hidden_score_vector",
    "attention_score_vector",
    "logit_metric_names",
    "selected_hidden_state_indices_hf",
    "selected_hidden_transformer_layers_zero_based",
    "selected_hidden_transformer_layers_1based",
    "selected_attention_layer_indices_zero_based",
    "selected_attention_transformer_layers_1based",
    "labels",
    "cache_dtype_name",
    "model_forward_dtype",
    "truncation_disabled",
}


def maybe_take_subset(df: pd.DataFrame, subset_size: int | None) -> pd.DataFrame:
    """Take a roughly stratified subset for smoke tests."""
    if subset_size is None or subset_size >= len(df):
        return df.reset_index(drop=True)

    parts: list[pd.DataFrame] = []
    label_counts = df["label"].value_counts(normalize=True).sort_index()
    for label, frac in label_counts.items():
        n_label = max(1, int(round(subset_size * frac)))
        label_df = df[df["label"] == label]
        parts.append(label_df.sample(n=min(n_label, len(label_df)), random_state=42))

    return (
        pd.concat(parts, axis=0)
        .sample(frac=1.0, random_state=42)
        .reset_index(drop=True)
    )


def parse_feature_sets(feature_sets_arg: str) -> list[str]:
    """Parse a comma-separated list of supported feature sets."""
    feature_sets: list[str] = []
    for item in feature_sets_arg.split(","):
        feature_set = item.strip()
        if not feature_set:
            continue
        if feature_set not in FEATURE_SET_COMPONENTS:
            raise ValueError(
                f"Unknown feature_set '{feature_set}'. "
                f"Expected one of {sorted(FEATURE_SET_COMPONENTS)}."
            )
        if feature_set not in feature_sets:
            feature_sets.append(feature_set)
    if not feature_sets:
        raise ValueError("feature_sets must contain at least one supported name")
    return feature_sets


def get_llm_check_model_and_tokenizer(
    model_name: str = DEFAULT_MODEL_NAME,
) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load Qwen with eager attention enabled in float32."""
    print(f"[Method 3 LogReg] Loading '{model_name}' in float32 with eager attention ...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        attn_implementation="eager",
    )
    model.requires_grad_(False)
    model.eval()
    return model, tokenizer


def extract_llm_check_feature_cache(
    df: pd.DataFrame,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_length: int = 512,
    device: torch.device | None = None,
) -> dict[str, np.ndarray]:
    """Extract the existing LLM-Check feature families at float32 precision."""
    if device is None:
        device = get_best_available_device()

    model, tokenizer = get_llm_check_model_and_tokenizer()
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model.to(device=device, dtype=torch.float32)

    n_samples = len(df)
    n_layers = model.config.num_hidden_layers
    selected_hidden_state_indices_hf = np.arange(1, n_layers + 1, dtype=np.int32)
    selected_hidden_transformer_layers_zero_based = np.arange(n_layers, dtype=np.int32)
    selected_hidden_transformer_layers_1based = (
        selected_hidden_transformer_layers_zero_based + 1
    )
    selected_attention_layer_indices_zero_based = np.arange(1, n_layers, dtype=np.int32)
    selected_attention_transformer_layers_1based = (
        selected_attention_layer_indices_zero_based + 1
    )

    prompts = df["prompt"].tolist()
    responses = df["response"].tolist()

    cache: dict[str, np.ndarray] = {
        "logit_metrics_vector": np.empty(
            (n_samples, len(DEFAULT_LOGIT_METRIC_NAMES)),
            dtype=np.float32,
        ),
        "hidden_score_vector": np.empty(
            (n_samples, len(selected_hidden_state_indices_hf)),
            dtype=np.float32,
        ),
        "attention_score_vector": np.empty(
            (n_samples, len(selected_attention_layer_indices_zero_based)),
            dtype=np.float32,
        ),
        "logit_metric_names": np.asarray(DEFAULT_LOGIT_METRIC_NAMES),
        "selected_hidden_state_indices_hf": selected_hidden_state_indices_hf,
        "selected_hidden_transformer_layers_zero_based": (
            selected_hidden_transformer_layers_zero_based
        ),
        "selected_hidden_transformer_layers_1based": (
            selected_hidden_transformer_layers_1based
        ),
        "selected_attention_layer_indices_zero_based": (
            selected_attention_layer_indices_zero_based
        ),
        "selected_attention_transformer_layers_1based": (
            selected_attention_transformer_layers_1based
        ),
        "selected_layer_indices_zero_based": selected_attention_layer_indices_zero_based,
        "selected_transformer_layers_1based": (
            selected_attention_transformer_layers_1based
        ),
        "prompt_token_length": np.empty(n_samples, dtype=np.int32),
        "response_token_length": np.empty(n_samples, dtype=np.int32),
        "response_truncated": np.empty(n_samples, dtype=np.int8),
        "max_length_requested": np.asarray([max_length], dtype=np.int32),
        "truncation_disabled": np.asarray([1], dtype=np.int8),
        "cache_dtype_name": np.asarray(["float32"]),
        "model_forward_dtype": np.asarray(["float32"]),
    }

    for start in tqdm(
        range(0, n_samples, batch_size),
        desc="Caching Method 3 log-reg features",
        unit="batch",
    ):
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
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                use_cache=False,
                output_attentions=True,
                output_hidden_states=True,
                return_dict=True,
            )

        batch_attention_mask = attention_mask.cpu()
        for sample_idx in range(input_ids.size(0)):
            row_idx = start + sample_idx
            seq_len = int(batch_attention_mask[sample_idx].sum().item())
            response_start, response_end = response_spans[sample_idx]
            response_end = min(response_end, seq_len)
            response_span = (response_start, response_end)

            sample_input_ids = input_ids[sample_idx, :seq_len].detach().to(device="cpu")
            sample_logits = outputs.logits[sample_idx, :seq_len].detach().to(
                device="cpu",
                dtype=torch.float32,
            )
            sample_hidden_states = [
                layer[sample_idx, :seq_len].detach().to(
                    device="cpu",
                    dtype=torch.float32,
                )
                for layer in outputs.hidden_states
            ]
            sample_attentions = [
                layer[sample_idx, :, :seq_len, :seq_len].detach().to(
                    device="cpu",
                    dtype=torch.float32,
                )
                for layer in outputs.attentions
            ]

            logit_metrics = _compute_logit_metrics(
                sample_logits=sample_logits,
                sample_input_ids=sample_input_ids,
                response_span=response_span,
            )
            hidden_scores = torch.stack(
                [
                    _compute_hidden_layer_score(
                        sample_hidden_states[layer_idx],
                        response_span,
                    )
                    for layer_idx in selected_hidden_state_indices_hf.tolist()
                ]
            )
            attention_scores = torch.stack(
                [
                    _compute_attention_layer_score(
                        sample_attentions[layer_idx],
                        response_span,
                    )
                    for layer_idx in selected_attention_layer_indices_zero_based.tolist()
                ]
            )

            cache["logit_metrics_vector"][row_idx] = logit_metrics.numpy().astype(
                np.float32,
                copy=False,
            )
            cache["hidden_score_vector"][row_idx] = hidden_scores.numpy().astype(
                np.float32,
                copy=False,
            )
            cache["attention_score_vector"][row_idx] = attention_scores.numpy().astype(
                np.float32,
                copy=False,
            )
            cache["prompt_token_length"][row_idx] = response_start
            cache["response_token_length"][row_idx] = response_end - response_start
            cache["response_truncated"][row_idx] = int(response_start == 0)

        del outputs
        del input_ids
        del attention_mask
        del batch_attention_mask
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if "label" in df.columns and df["label"].notna().all():
        cache["labels"] = df["label"].astype(int).to_numpy(dtype=np.int32)

    return cache


def build_feature_matrix(
    cache: dict[str, np.ndarray],
    feature_set: str = DEFAULT_FEATURE_SET,
) -> np.ndarray:
    """Return the selected LLM-Check feature matrix."""
    if feature_set not in FEATURE_SET_COMPONENTS:
        raise ValueError(f"Unknown feature_set '{feature_set}'")

    arrays = [
        cache[key].astype(np.float32, copy=False)
        for key in FEATURE_SET_COMPONENTS[feature_set]
    ]
    if len(arrays) == 1:
        return arrays[0]
    return np.concatenate(arrays, axis=1)


def build_feature_set_metadata(
    cache: dict[str, np.ndarray],
    feature_set: str,
) -> dict[str, object]:
    """Describe the selected feature set for metadata and experiment logs."""
    if feature_set not in FEATURE_SET_COMPONENTS:
        raise ValueError(f"Unknown feature_set '{feature_set}'")

    components = [
        FEATURE_COMPONENT_LABELS[key] for key in FEATURE_SET_COMPONENTS[feature_set]
    ]
    component_dims = {
        FEATURE_COMPONENT_LABELS[key]: int(cache[key].shape[1])
        for key in FEATURE_SET_COMPONENTS[feature_set]
    }
    metadata: dict[str, object] = {
        "feature_set": feature_set,
        "description": FEATURE_SET_DESCRIPTIONS[feature_set],
        "components": components,
        "component_dims": component_dims,
        "feature_dim": int(sum(component_dims.values())),
        "response_span": "response_only",
        "truncation": "disabled",
        "cache_dtype": "float32",
        "model_forward_dtype": "float32",
    }

    if "logit_metrics_vector" in FEATURE_SET_COMPONENTS[feature_set]:
        metadata["logit_metric_names"] = cache["logit_metric_names"].tolist()

    if "hidden_score_vector" in FEATURE_SET_COMPONENTS[feature_set]:
        metadata["selected_hidden_state_indices_hf"] = (
            cache["selected_hidden_state_indices_hf"].astype(int).tolist()
        )
        metadata["selected_hidden_transformer_layers_zero_based"] = (
            cache["selected_hidden_transformer_layers_zero_based"]
            .astype(int)
            .tolist()
        )
        metadata["selected_hidden_transformer_layers_1based"] = (
            cache["selected_hidden_transformer_layers_1based"]
            .astype(int)
            .tolist()
        )

    if "attention_score_vector" in FEATURE_SET_COMPONENTS[feature_set]:
        metadata["selected_attention_layer_indices_zero_based"] = (
            cache["selected_attention_layer_indices_zero_based"]
            .astype(int)
            .tolist()
        )
        metadata["selected_attention_transformer_layers_1based"] = (
            cache["selected_attention_transformer_layers_1based"]
            .astype(int)
            .tolist()
        )

    return metadata


def load_or_build_cache(
    df: pd.DataFrame,
    cache_file: Path = DEFAULT_CACHE_FILE,
    data_file: Path = DEFAULT_DATA_FILE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_length: int = 512,
    overwrite_cache: bool = False,
    subset_size: int | None = None,
) -> dict[str, np.ndarray]:
    """Load the float32 Method 3 cache or build it once."""
    cache_file = Path(cache_file)
    should_rebuild_cache = overwrite_cache or subset_size is not None or not cache_file.exists()

    if not should_rebuild_cache:
        print(f"[Method 3 LogReg] Loading cache from {cache_file}")
        cache = load_feature_cache(cache_file)
        labels_match = "labels" in cache and len(cache["labels"]) == len(df)
        keys_match = REQUIRED_CACHE_KEYS.issubset(cache.keys())
        truncation_disabled_match = (
            int(cache.get("truncation_disabled", np.asarray([0], dtype=np.int8))[0]) == 1
        )
        dtype_match = all(
            str(cache[key].dtype) == "float32"
            for key in (
                "logit_metrics_vector",
                "hidden_score_vector",
                "attention_score_vector",
            )
        )
        model_dtype_match = (
            str(cache.get("model_forward_dtype", np.asarray([""], dtype="<U7"))[0])
            == "float32"
        )
        if not (
            labels_match
            and keys_match
            and truncation_disabled_match
            and dtype_match
            and model_dtype_match
        ):
            print("[Method 3 LogReg] Cache metadata mismatch detected. Rebuilding cache.")
            should_rebuild_cache = True

    if should_rebuild_cache:
        print(f"[Method 3 LogReg] Building cache from {data_file}")
        cache = extract_llm_check_feature_cache(
            df=df,
            batch_size=batch_size,
            max_length=max_length,
        )
        if subset_size is None:
            save_feature_cache(cache_file, cache)
            print(f"[Method 3 LogReg] Saved cache to {cache_file}")
        else:
            print("[Method 3 LogReg] Subset run detected. Cache was not persisted.")
        return cache

    return cache


def prepare_feature_matrix(
    df: pd.DataFrame,
    feature_set: str = DEFAULT_FEATURE_SET,
    cache_file: Path = DEFAULT_CACHE_FILE,
    data_file: Path = DEFAULT_DATA_FILE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_length: int = 512,
    overwrite_cache: bool = False,
    subset_size: int | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, object]]:
    """Load or build the shared cache and return one selected feature matrix."""
    cache = load_or_build_cache(
        df=df,
        cache_file=cache_file,
        data_file=data_file,
        batch_size=batch_size,
        max_length=max_length,
        overwrite_cache=overwrite_cache,
        subset_size=subset_size,
    )
    metadata = build_feature_set_metadata(cache, feature_set)
    X = build_feature_matrix(cache, feature_set=feature_set)
    return X, cache, metadata
