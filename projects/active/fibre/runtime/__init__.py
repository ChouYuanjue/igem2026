"""Canonical FIBRE runtime interfaces."""

__all__ = ["FibreEvidenceRuntime", "RankedEvidenceCandidate"]


def __getattr__(name: str):
    if name in __all__:
        from .evidence import FibreEvidenceRuntime, RankedEvidenceCandidate

        return {
            "FibreEvidenceRuntime": FibreEvidenceRuntime,
            "RankedEvidenceCandidate": RankedEvidenceCandidate,
        }[name]
    raise AttributeError(name)
