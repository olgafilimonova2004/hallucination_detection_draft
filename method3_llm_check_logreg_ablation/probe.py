"""Logistic probe families for the Method 3 regularization ablation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler


DEFAULT_L2_C_VALUES = (0.01, 0.1, 1.0, 10.0)
DEFAULT_L1_C_VALUES = (0.01, 0.1, 1.0, 10.0)
DEFAULT_ELASTICNET_C_VALUES = (0.01, 0.1, 1.0)
DEFAULT_ELASTICNET_L1_RATIOS = (0.25, 0.5, 0.75)
DEFAULT_CLASS_WEIGHT_TOKENS = ("balanced", "none")
DEFAULT_MAX_ITER = 2000
RANDOM_STATE = 42

PENALTY_SIMPLICITY_RANK = {
    "l2": 0,
    "l1": 1,
    "elasticnet": 2,
}
SOLVER_SIMPLICITY_RANK = {
    "liblinear": 0,
    "lbfgs": 1,
    "saga": 2,
}


def format_class_weight(class_weight: str | None) -> str:
    """Format ``class_weight`` for logs and JSON output."""
    return "none" if class_weight is None else str(class_weight)


def parse_class_weights(class_weights_arg: str) -> list[str | None]:
    """Parse a comma-separated list of class-weight modes."""
    parsed: list[str | None] = []
    for item in class_weights_arg.split(","):
        token = item.strip().lower()
        if not token:
            continue
        if token == "none":
            value = None
        elif token == "balanced":
            value = "balanced"
        else:
            raise ValueError(
                f"Unsupported class_weight '{item}'. Expected 'balanced' or 'none'."
            )
        if value not in parsed:
            parsed.append(value)
    if not parsed:
        raise ValueError("class_weights must contain at least one of: balanced, none")
    return parsed


def _format_numeric_token(value: float) -> str:
    text = f"{value:g}"
    return text.replace(".", "p").replace("-", "m")


@dataclass(frozen=True)
class LogisticProbeConfig:
    """Serializable logistic-regression configuration."""

    penalty: str
    solver: str
    c: float
    l1_ratio: float | None = None
    class_weight: str | None = "balanced"
    max_iter: int = DEFAULT_MAX_ITER
    random_state: int = RANDOM_STATE

    def __post_init__(self) -> None:
        if self.penalty not in PENALTY_SIMPLICITY_RANK:
            raise ValueError(
                f"Unsupported penalty '{self.penalty}'. "
                f"Expected one of {sorted(PENALTY_SIMPLICITY_RANK)}."
            )
        if self.penalty == "l2" and self.solver not in {"liblinear", "lbfgs"}:
            raise ValueError("L2 logistic must use liblinear or lbfgs in this study.")
        if self.penalty == "l1" and self.solver not in {"liblinear", "saga"}:
            raise ValueError("L1 logistic must use liblinear or saga in this study.")
        if self.penalty == "elasticnet" and self.solver != "saga":
            raise ValueError("Elastic-net logistic must use saga.")
        if self.penalty == "elasticnet":
            if self.l1_ratio is None:
                raise ValueError("Elastic-net logistic requires l1_ratio.")
            if not 0.0 <= self.l1_ratio <= 1.0:
                raise ValueError("l1_ratio must be within [0, 1].")
        elif self.l1_ratio is not None:
            raise ValueError("l1_ratio is only valid for elastic-net logistic.")
        if self.c <= 0.0:
            raise ValueError("C must be strictly positive.")
        if self.class_weight not in {None, "balanced"}:
            raise ValueError("class_weight must be None or 'balanced'.")


def probe_config_to_metadata(config: LogisticProbeConfig) -> dict[str, object]:
    """Convert a config to row-friendly metadata."""
    return {
        "classifier": "logistic",
        "penalty": config.penalty,
        "solver": config.solver,
        "C": float(config.c),
        "l1_ratio": None if config.l1_ratio is None else float(config.l1_ratio),
        "class_weight": format_class_weight(config.class_weight),
        "max_iter": int(config.max_iter),
        "random_state": int(config.random_state),
    }


def probe_config_from_record(record: Mapping[str, object]) -> LogisticProbeConfig:
    """Reconstruct a config from a CSV/JSON record."""
    l1_ratio = record.get("l1_ratio")
    if l1_ratio in ("", None):
        parsed_l1_ratio = None
    else:
        parsed_l1_ratio = float(l1_ratio)
        if np.isnan(parsed_l1_ratio):
            parsed_l1_ratio = None

    class_weight_token = str(record.get("class_weight", "balanced")).strip().lower()
    class_weight = None if class_weight_token in {"", "none", "nan"} else class_weight_token

    return LogisticProbeConfig(
        penalty=str(record["penalty"]),
        solver=str(record["solver"]),
        c=float(record["C"]),
        l1_ratio=parsed_l1_ratio,
        class_weight=class_weight,
        max_iter=int(float(record.get("max_iter", DEFAULT_MAX_ITER))),
        random_state=int(float(record.get("random_state", RANDOM_STATE))),
    )


def format_probe_name(feature_set: str, config: LogisticProbeConfig) -> str:
    """Build a stable experiment name."""
    parts = [
        f"{feature_set}__logreg",
        config.penalty,
        config.solver,
        f"c{_format_numeric_token(config.c)}",
        f"cw{format_class_weight(config.class_weight)}",
    ]
    if config.l1_ratio is not None:
        parts.append(f"l1r{_format_numeric_token(config.l1_ratio)}")
    return "_".join(parts)


def probe_signature(config: LogisticProbeConfig) -> tuple[object, ...]:
    """Return a feature-set-independent key for config deduplication."""
    return (
        config.penalty,
        config.solver,
        float(config.c),
        None if config.l1_ratio is None else float(config.l1_ratio),
        format_class_weight(config.class_weight),
        int(config.max_iter),
        int(config.random_state),
    )


def regularization_sort_fields(config: LogisticProbeConfig) -> dict[str, float]:
    """Return scalar fields used to rank stronger regularization first."""
    return {
        "regularization_c_rank": float(config.c),
        "regularization_l1_ratio_rank": (
            1.0 - float(config.l1_ratio)
            if config.penalty == "elasticnet" and config.l1_ratio is not None
            else 1.0
        ),
    }


def simplicity_sort_fields(config: LogisticProbeConfig) -> dict[str, int]:
    """Return scalar fields used to rank simpler penalties and solvers first."""
    return {
        "simplicity_penalty_rank": PENALTY_SIMPLICITY_RANK[config.penalty],
        "simplicity_solver_rank": SOLVER_SIMPLICITY_RANK[config.solver],
        "weighting_rank": 0 if config.class_weight == "balanced" else 1,
    }


def build_primary_logistic_configs(
    class_weights: list[str | None],
) -> list[LogisticProbeConfig]:
    """Build the required logistic-regression comparison grid."""
    configs: list[LogisticProbeConfig] = []
    for class_weight in class_weights:
        for solver in ("liblinear", "lbfgs"):
            for c in DEFAULT_L2_C_VALUES:
                configs.append(
                    LogisticProbeConfig(
                        penalty="l2",
                        solver=solver,
                        c=c,
                        class_weight=class_weight,
                    )
                )
        for solver in ("liblinear", "saga"):
            for c in DEFAULT_L1_C_VALUES:
                configs.append(
                    LogisticProbeConfig(
                        penalty="l1",
                        solver=solver,
                        c=c,
                        class_weight=class_weight,
                    )
                )
        for c in DEFAULT_ELASTICNET_C_VALUES:
            for l1_ratio in DEFAULT_ELASTICNET_L1_RATIOS:
                configs.append(
                    LogisticProbeConfig(
                        penalty="elasticnet",
                        solver="saga",
                        c=c,
                        l1_ratio=l1_ratio,
                        class_weight=class_weight,
                    )
                )
    return configs


class _ThresholdedProbe(nn.Module):
    """Shared validation-threshold tuning helpers for binary probes."""

    def __init__(self) -> None:
        super().__init__()
        self._threshold: float = 0.5

    def fit_hyperparameters(
        self,
        X_val: np.ndarray,
        y_val: np.ndarray,
    ) -> "_ThresholdedProbe":
        probs = self.predict_proba(X_val)[:, 1]
        candidates = np.unique(np.concatenate([probs, np.linspace(0.0, 1.0, 101)]))

        best_threshold = 0.5
        best_f1 = -1.0
        for threshold in candidates:
            y_pred = (probs >= threshold).astype(int)
            score = f1_score(y_val, y_pred, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_threshold = float(threshold)

        self._threshold = best_threshold
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= self._threshold).astype(int)


class HallucinationLogisticProbe(_ThresholdedProbe):
    """Scaled sklearn logistic regression over the selected Method 3 features."""

    def __init__(self, config: LogisticProbeConfig) -> None:
        super().__init__()
        self.config = config
        self._scaler = StandardScaler()
        self._classifier: LogisticRegression | None = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        probs = self.predict_proba(x.detach().cpu().numpy())[:, 1]
        probs = np.clip(probs, 1e-6, 1.0 - 1e-6)
        logits = np.log(probs / (1.0 - probs)).astype(np.float32)
        return torch.from_numpy(logits)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "HallucinationLogisticProbe":
        X_arr = np.asarray(X, dtype=np.float32)
        y_arr = np.asarray(y, dtype=int)
        X_scaled = self._scaler.fit_transform(X_arr)

        self._classifier = LogisticRegression(
            penalty=self.config.penalty,
            solver=self.config.solver,
            C=self.config.c,
            l1_ratio=self.config.l1_ratio,
            class_weight=self.config.class_weight,
            max_iter=self.config.max_iter,
            random_state=self.config.random_state,
        )
        self._classifier.fit(X_scaled, y_arr)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self._classifier is None:
            raise RuntimeError("Probe has not been fitted yet. Call fit() first.")

        X_arr = np.asarray(X, dtype=np.float32)
        X_scaled = self._scaler.transform(X_arr)
        prob_pos = self._classifier.predict_proba(X_scaled)[:, 1]
        return np.stack([1.0 - prob_pos, prob_pos], axis=1)
