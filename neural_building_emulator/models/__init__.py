"""Model blocks for the neural building emulator."""

from .closed_loop_hp import ClosedLoopHPEmulator
from .causal_hybrid_hp import CausalHybridHPEmulator, CausalHybridRollout
from .contracting_closed_loop_hp import (
    ContractingClosedLoopHPEmulator,
    ThermalQResponseMode,
    ThermostatDemandMode,
    TransitionConditioningMode,
)
from .emulator import InputEncoderFeedback, MetadataStateSpaceEmulator, SwitchingDynamics
from .probabilistic_closed_loop_hp import (
    HPElectricRolloutMode,
    HPElectricScenarioMode,
    HPActivationModel,
    HPTrainingMode,
    ProbabilisticClosedLoopHPEmulator,
    ProbHpEmissionMode,
)
from .probabilistic_contracting_closed_loop_hp import ProbabilisticContractingClosedLoopHPEmulator
from .probabilistic_emulator import ProbabilisticStableStateSpaceEmulator
from .schur import simba_schur_matrix, spectral_radius
from .state_space import StateSpaceMatrices, rollout_state_space

__all__ = [
    "ClosedLoopHPEmulator",
    "CausalHybridHPEmulator",
    "CausalHybridRollout",
    "ContractingClosedLoopHPEmulator",
    "ThermalQResponseMode",
    "ThermostatDemandMode",
    "TransitionConditioningMode",
    "HPElectricRolloutMode",
    "HPElectricScenarioMode",
    "HPActivationModel",
    "HPTrainingMode",
    "MetadataStateSpaceEmulator",
    "InputEncoderFeedback",
    "SwitchingDynamics",
    "ProbabilisticClosedLoopHPEmulator",
    "ProbabilisticContractingClosedLoopHPEmulator",
    "ProbHpEmissionMode",
    "ProbabilisticStableStateSpaceEmulator",
    "StateSpaceMatrices",
    "rollout_state_space",
    "simba_schur_matrix",
    "spectral_radius",
]
