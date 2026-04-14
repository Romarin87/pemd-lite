class PEMDLiteError(RuntimeError):
    """Base error for PEMD-Lite."""


class ProjectValidationError(PEMDLiteError):
    """Configuration or artifact validation failed."""


class ResumeStateError(PEMDLiteError):
    """Existing artifacts are inconsistent with the requested run."""


class ChargeBackendError(PEMDLiteError):
    """Charge backend failed."""


class LigParGenBackendError(ChargeBackendError):
    """LigParGen backend failed or returned incomplete output."""


class TopologyMismatchError(PEMDLiteError):
    """Topology-derived artifacts are inconsistent."""


class AtomTypingError(PEMDLiteError):
    """Foyer could not assign atom types."""


class MissingBondParameterError(PEMDLiteError):
    """The generated XML lacks bonded parameters required by the long chain."""


class GromacsRuntimeError(PEMDLiteError):
    """GROMACS command execution failed."""
