"""Small, auditable topology-optimization benchmark. Not a production solver."""
from .solver import Case, Model, optimize
from .solver3d import Case3D, Model3D, optimize3d

__all__ = ["Case", "Model", "optimize", "Case3D", "Model3D", "optimize3d"]
