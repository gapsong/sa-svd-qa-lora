"""Structure-Aware SVD (SA-SVD) initialization for QA-LoRA."""

from .core import SASVDResult, reconstruction_error, sa_svd_init
from .integration import (
    apply_sa_svd_to_base,
    find_target_linears,
    write_adapter_weights,
)

__all__ = [
    "SASVDResult",
    "sa_svd_init",
    "reconstruction_error",
    "find_target_linears",
    "apply_sa_svd_to_base",
    "write_adapter_weights",
]

__version__ = "0.1.0"
