"""Cosmological initial conditions with double precision JAX arrays."""
from jax import config
config.update("jax_enable_x64", True)
from .core import Cosmology, Grid, Simulation, PowerSpectrum, Geometry

__all__ = ["Cosmology", "Grid", "Simulation", "PowerSpectrum", "Geometry"]
