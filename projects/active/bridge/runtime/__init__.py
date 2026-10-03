"""Canonical BRIDGE runtime interfaces."""

__all__ = [
    "FibreEvidenceRuntime",
    "RankedEvidenceCandidate",
    "R2EExpertTypeRuntime",
    "ExpertTypeOutput",
]


def __getattr__(name: str):
    if name in {"FibreEvidenceRuntime", "RankedEvidenceCandidate"}:
        from .evidence import FibreEvidenceRuntime, RankedEvidenceCandidate
        return {
            "FibreEvidenceRuntime": FibreEvidenceRuntime,
            "RankedEvidenceCandidate": RankedEvidenceCandidate,
        }[name]
    if name in {"R2EExpertTypeRuntime", "ExpertTypeOutput"}:
        from .expert_types import R2EExpertTypeRuntime, ExpertTypeOutput
        return {
            "R2EExpertTypeRuntime": R2EExpertTypeRuntime,
            "ExpertTypeOutput": ExpertTypeOutput,
        }[name]
    raise AttributeError(name)
