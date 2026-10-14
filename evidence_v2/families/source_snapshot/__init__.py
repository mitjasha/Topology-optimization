"""Small, auditable topology-optimization benchmark. Not a production solver."""
from .solver import Case, Model, optimize

__all__ = ["Case", "Model", "optimize"]
