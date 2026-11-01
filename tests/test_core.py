import jax
import jax.numpy as jnp
import numpy as np
from genetic import Cosmology, Grid, PowerSpectrum, Simulation
from genetic.core import fft, ifft, interpolate, displacement


def test_interpolation_adjoint_and_periodicity():
    values = jnp.arange(64.).reshape(4, 4, 4)
    points = jnp.array([[-.3, 1.2, 3.7], [3.7, 1.2, -.3]])
    evaluate = lambda v: interpolate(v, points)
    w = jnp.array([.4, -.2])
    adjoint = jax.linear_transpose(evaluate, values)(w)[0]
    np.testing.assert_allclose(jnp.vdot(evaluate(values), w), jnp.vdot(values, adjoint), atol=1e-12)
    np.testing.assert_allclose(evaluate(values), interpolate(values, points+4), atol=1e-12)


def test_poisson_solution_and_nyquist_derivative():
    grid = Grid(8, 8)
    phase = jnp.asarray(grid.points[..., 0])*2*jnp.pi/8
    density = jnp.cos(phase)
    offsets = displacement(fft(density), grid)
    np.testing.assert_allclose(offsets[..., 0], -jnp.sin(phase)/(2*jnp.pi/8), atol=1e-14)
    np.testing.assert_allclose(offsets[..., 1:], 0, atol=1e-14)
    nyquist = (-1.)**jnp.arange(8)[:, None, None]*jnp.ones((8, 8, 8))
    np.testing.assert_allclose(displacement(fft(nyquist), grid), 0, atol=1e-14)


def test_linear_constraints_preserve_others():
    s = Simulation(spectrum=PowerSpectrum(Cosmology(ns=-2)))
    s.base_grid(8, 8)
    s.seed, s.seed_mode = 42, "real"
    s.center = jnp.full(3, 4.)
    s.select("sphere", 2)
    s.modify("overdensity", "absolute", .2)
    s.center = jnp.full(3, 1.)
    s.select("nearest")
    s.modify("overdensity", "absolute", -.1)
    s.apply_modifications()
    np.testing.assert_allclose(s.calculate("overdensity"), -.1, atol=1e-12)
    s.center = jnp.full(3, 4.)
    s.select("sphere", 2)
    np.testing.assert_allclose(s.calculate("overdensity"), .2, atol=1e-12)


def test_splice_inserts_selected_physical_field():
    s = Simulation(spectrum=PowerSpectrum(Cosmology(ns=-2)))
    s.base_grid(8, 8)
    s.seed, s.seed_mode = 42, "real"
    s.center = jnp.full(3, 4.)
    s.select("sphere", 2)
    s.draw()
    root = jnp.sqrt(s.spectrum.covariance(s.grids[0])).at[0, 0, 0].set(1)
    before = np.asarray(ifft(s.noise[0]*root))
    from genetic.random import LegacyRandom
    inserted = np.asarray(ifft(LegacyRandom(43).field(8, "real")*root))
    s.splice(43, accuracy=1e-12)
    after = np.asarray(ifft(s.noise[0]*root))
    np.testing.assert_allclose(after[s.grids[0].flags], inserted[s.grids[0].flags], atol=1e-10)
    assert np.linalg.norm(after-before) > .1


def test_fixed_fourier_power_preserves_empty_modes():
    from genetic.random import LegacyRandom
    modes = np.asarray(LegacyRandom(42).field(8, "parallel", fixed=True))
    np.testing.assert_allclose(modes[4, :, :], 0, atol=1e-14)
    np.testing.assert_allclose(modes[:, 4, :], 0, atol=1e-14)
    np.testing.assert_allclose(modes[:, :, 4], 0, atol=1e-14)
    nonzero = np.abs(modes) > 1e-12
    np.testing.assert_allclose(np.abs(modes[nonzero]), 1, atol=1e-14)


def test_multilevel_synthesis_transpose_and_splice():
    from genetic.random import LegacyRandom
    s = Simulation(spectrum=PowerSpectrum(Cosmology(ns=-2)))
    s.base_grid(8, 4)
    s.seed, s.seed_mode = 42, "real"
    s.center = jnp.full(3, 4.)
    s.select("sphere", 2)
    s.zoom_grid(2, 4)
    s.select("sphere", 1)
    s.draw()
    unpack = lambda v: (v[:64].reshape(4, 4, 4), v[64:].reshape(4, 4, 4))
    pack = lambda fields: jnp.concatenate([v.reshape(-1) for v in fields])
    observe = lambda v: s.synthesize(unpack(v))[1][s.grids[1].flags]
    matrix = jax.jacfwd(observe)(jnp.zeros(128))
    original = pack(tuple(ifft(v) for v in s.noise))
    rng = LegacyRandom(43)
    donor = pack(tuple(ifft(rng.field(4, "real")) for _ in s.grids))
    expected = original+matrix.T@jnp.linalg.solve(matrix@matrix.T, matrix@(donor-original))
    for solver in ("cg", "minres"):
        s.noise = [fft(v) for v in unpack(original)]
        s.splice(43, accuracy=1e-12, solver=solver, border=1)
        result = pack(tuple(ifft(v) for v in s.noise))
        np.testing.assert_allclose(result, expected, atol=1e-10, rtol=1e-10)
        assert all(isinstance(v, jax.Array) for v in s.noise+list(s.grids[0].wavevectors))


def test_core_has_no_host_numeric_dependencies():
    import ast
    from pathlib import Path
    import genetic
    root = Path(genetic.__file__).parent
    for name in ("core.py", "random.py"):
        tree = ast.parse((root/name).read_text())
        imports = [node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)]
        imports += [alias.name for node in ast.walk(tree) if isinstance(node, ast.Import) for alias in node.names]
        assert not any(module and module.split('.')[0] in {"numpy", "scipy", "ctypes"} for module in imports)
