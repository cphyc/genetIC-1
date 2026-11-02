"""genetIC's GSL ranlxs2/ziggurat stream, transferred to JAX in bulk."""
import jax.numpy as jnp
from . import _random


class LegacyRandom:
    def __init__(self, seed):
        self.seed = seed
        self.state = _random.seed(seed)

    def normal(self, size):
        return jnp.frombuffer(_random.normal(self.state, size), dtype=jnp.float64)

    def field(self, n, mode="parallel", fixed=False, half_spectrum=False):
        if mode == "real":
            transform = jnp.fft.rfftn if half_spectrum else jnp.fft.fftn
            result = transform(self.normal(n**3).reshape((n,)*3), norm="ortho").at[0, 0, 0].set(0)
        else:
            half = jnp.frombuffer(_random.fourier(self.state, self.seed, n, mode == "parallel", mode == "reverse"),
                                  dtype=jnp.complex128).reshape(n, n, n//2+1)
            if fixed:
                half /= jnp.where(jnp.abs(half) > 0, jnp.abs(half), 1)
            result = half if half_spectrum else jnp.fft.fftn(jnp.fft.irfftn(half, s=(n,)*3, norm="ortho"), norm="ortho")
        if fixed and mode == "real":
            result = jnp.where(jnp.abs(result) > 0, result/jnp.where(jnp.abs(result) > 0, jnp.abs(result), 1), 0)
        return result
