"""Single registry for every model participating in benchmark comparisons."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

TaskName = Literal["q_to_t", "closed_loop_hp"]
BackendName = Literal["legacy", "lstm"]


@dataclass(frozen=True)
class ModelSpec:
    name: str
    task: TaskName
    backend: BackendName
    probabilistic: bool
    legacy_model_kind: str | None
    description: str


MODEL_REGISTRY: dict[str, ModelSpec] = {
    "q_to_t_deterministic_ss": ModelSpec(
        name="q_to_t_deterministic_ss",
        task="q_to_t",
        backend="legacy",
        probabilistic=False,
        legacy_model_kind="deterministic",
        description="Metadata-conditioned deterministic stable state-space Qroom-to-T model.",
    ),
    "q_to_t_probabilistic_ss": ModelSpec(
        name="q_to_t_probabilistic_ss",
        task="q_to_t",
        backend="legacy",
        probabilistic=True,
        legacy_model_kind="probabilistic",
        description="Metadata-conditioned probabilistic stable state-space Qroom-to-T model.",
    ),
    "closed_loop_hp_deterministic_ss": ModelSpec(
        name="closed_loop_hp_deterministic_ss",
        task="closed_loop_hp",
        backend="legacy",
        probabilistic=False,
        legacy_model_kind="closed_loop_hp",
        description="Deterministic HP/buffer model with generated stable thermal state space.",
    ),
    "closed_loop_hp_probabilistic_ss": ModelSpec(
        name="closed_loop_hp_probabilistic_ss",
        task="closed_loop_hp",
        backend="legacy",
        probabilistic=True,
        legacy_model_kind="closed_loop_hp_probabilistic",
        description="Probabilistic HP/buffer model with generated stable thermal state space.",
    ),
    "closed_loop_hp_contracting_deterministic": ModelSpec(
        name="closed_loop_hp_contracting_deterministic",
        task="closed_loop_hp",
        backend="legacy",
        probabilistic=False,
        legacy_model_kind="closed_loop_hp_contracting",
        description="Deterministic HP/buffer model with bounded thermal dynamics.",
    ),
    "closed_loop_hp_contracting_probabilistic": ModelSpec(
        name="closed_loop_hp_contracting_probabilistic",
        task="closed_loop_hp",
        backend="legacy",
        probabilistic=True,
        legacy_model_kind="closed_loop_hp_contracting_probabilistic",
        description="Probabilistic HP/buffer model with bounded time-varying thermal dynamics.",
    ),
    "q_to_t_lstm": ModelSpec(
        name="q_to_t_lstm",
        task="q_to_t",
        backend="lstm",
        probabilistic=False,
        legacy_model_kind=None,
        description="Unstructured autoregressive LSTM predicting Tin from known Qroom and weather.",
    ),
    "closed_loop_hp_lstm": ModelSpec(
        name="closed_loop_hp_lstm",
        task="closed_loop_hp",
        backend="lstm",
        probabilistic=False,
        legacy_model_kind=None,
        description="Unstructured autoregressive LSTM predicting Tin, Qroom and Pel from Tset.",
    ),
}


def get_model_spec(name: str) -> ModelSpec:
    try:
        return MODEL_REGISTRY[name]
    except KeyError as exc:
        choices = ", ".join(sorted(MODEL_REGISTRY))
        raise ValueError(f"Unknown model {name!r}. Expected one of: {choices}") from exc
