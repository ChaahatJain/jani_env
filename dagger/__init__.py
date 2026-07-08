from .interfaces import (
    PolicyInterface,
    OracleInterface,
    TraceSamplerInterface,
    FaultCollectorInterface,
    PolicyUpdaterInterface,
)
from .sampler import StandardTraceSampler
from .fault_collector import OracleFaultCollector

__all__ = [
    "PolicyInterface",
    "OracleInterface",
    "TraceSamplerInterface",
    "FaultCollectorInterface",
    "PolicyUpdaterInterface",
    "StandardTraceSampler",
    "OracleFaultCollector",
    "NNPolicyWrapper",
]


def __getattr__(name):
    if name == "NNPolicyWrapper":
        from .policy_wrapper import NNPolicyWrapper

        return NNPolicyWrapper
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
