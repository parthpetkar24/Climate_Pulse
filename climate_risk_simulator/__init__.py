"""Climate Risk & Adaptation Simulator (simplified scenario model)."""
from .engine import SimulationError, run_simulation
from .router import router

__all__ = ["router", "run_simulation", "SimulationError"]
