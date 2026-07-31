"""Model blocks for the neural building emulator."""

from .emulator import InputEncoderFeedback, MetadataStateSpaceEmulator, SwitchingDynamics
from .probabilistic_emulator import ProbabilisticStableStateSpaceEmulator
from .schur import simba_schur_matrix, spectral_radius
from .state_space import StateSpaceMatrices, rollout_state_space

__all__ = [
    "MetadataStateSpaceEmulator",
    "InputEncoderFeedback",
    "SwitchingDynamics",
    "ProbabilisticStableStateSpaceEmulator",
    "StateSpaceMatrices",
    "rollout_state_space",
    "simba_schur_matrix",
    "spectral_radius",
]
