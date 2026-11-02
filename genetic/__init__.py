"""Cosmological initial conditions with double precision JAX arrays."""
import os
from pathlib import Path

from jax import config

if config.jax_compilation_cache_dir is None:
    cache_root = Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
    config.update("jax_compilation_cache_dir", str(cache_root / "genetic-jax" / "jax"))
if "JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS" not in os.environ:
    config.update("jax_persistent_cache_min_compile_time_secs", 0)

config.update("jax_enable_x64", True)
from .core import Cosmology, Grid, Simulation, PowerSpectrum, Geometry

__all__ = ["Cosmology", "Grid", "Simulation", "PowerSpectrum", "Geometry"]
