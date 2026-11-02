"""Numerical core. Lengths are Mpc/h, masses 1e10 Msun/h, velocities km/s/sqrt(a).

Internal routines assume aligned, nested cubic grids with integral resolution ratios.
Fourier arrays use unitary normalization and real fields use cell centred samples.
"""

import math
from dataclasses import dataclass, field
from functools import cached_property, partial

import jax
import jax.numpy as jnp

fft = partial(jnp.fft.fftn, norm="ortho")
ifft = lambda x: jnp.fft.ifftn(x, norm="ortho").real


@jax.jit
def natural_spline(x, y):
    """Natural cubic coefficients, ordered as scipy.interpolate.CubicSpline.c."""
    h = jnp.diff(x)
    lower, upper = jnp.r_[0.0, h[:-1], 0.0], jnp.r_[0.0, h[1:], 0.0]
    diagonal = jnp.r_[1.0, 2 * (h[:-1] + h[1:]), 1.0]
    rhs = jnp.r_[0.0, 6 * jnp.diff(jnp.diff(y) / h), 0.0]
    second = jax.lax.linalg.tridiagonal_solve(lower, diagonal, upper, rhs[:, None])[:, 0]
    return jnp.stack([jnp.diff(second) / (6 * h), second[:-1] / 2, jnp.diff(y) / h - h * (2 * second[:-1] + second[1:]) / 6, y[:-1]])


def dilate(mask, cells):
    padded = jnp.pad(mask, cells, mode="wrap")
    return jax.lax.reduce_window(padded, False, jax.lax.bitwise_or, (2 * cells + 1,) * 3, (1,) * 3, "VALID")


@dataclass
class Cosmology:
    Om: float = 0.3
    Ol: float = 0.7
    Ob: float = 0.0
    hubble: float = 0.701
    s8: float = 0.817
    ns: float = 0.96
    z: float = 99.0
    TCMB: float = 2.725

    @property
    def a(self):
        return 1 / (1 + self.z)

    def growth(self, a=None):
        a = self.a if a is None else a
        # The reference intentionally uses powf for this one occurrence of a**3.
        a3 = jnp.asarray(a, jnp.float32) ** jnp.float32(3)
        a3 = a3.astype(jnp.float64)
        h2 = self.Om / a3 + (1 - self.Om - self.Ol) / a**2 + self.Ol
        m, l = self.Om / h2 / a**3, self.Ol / h2
        return 2.5 * a * self.Om / a**3 / h2 / (m ** jnp.float32(4 / 7).astype(jnp.float64) - l + (1 + 0.5 * m) * (1 + l / 70))

    @property
    def velocity_factor(self):
        return 100 * jnp.sqrt(self.Om / self.a**3 + self.Ol) * jnp.sqrt(self.a)

    @property
    def temperature(self):
        return self.TCMB * min(self.a, 1 / 150) / self.a**2

    @property
    def internal_energy(self):
        return 1.5 * self.temperature * 1.380649e-29 / (1.6726219e-27 * 4 / 3.28) / self.a


@jax.jit
def spline_transfer(k, x, coefficients):
    logk = jnp.log(jnp.where(k > 0, k, 1))
    index = jnp.clip(jnp.searchsorted(jnp.asarray(x), logk) - 1, 0, len(x) - 2)
    dx = logk - jnp.asarray(x)[index]
    c = jnp.asarray(coefficients)[:, index]
    value = jnp.exp(((c[0] * dx + c[1]) * dx + c[2]) * dx + c[3])
    return jnp.where((k > 0) & (logk >= x[0]) & (logk <= x[-1]), value, 0)


@jax.jit
def spectrum_power(k, amplitude, ns, spline=None):
    transfer = spline_transfer(k, *spline) if spline is not None else 1
    return jnp.where(k > 0, amplitude * jnp.where(k > 0, k, 1) ** ns * transfer * transfer, 0)


class PowerSpectrum:
    """CAMB transfer tables (7 or 13 columns), or a power law for experiments."""

    def __init__(self, cosmo, table=None, amplitude=1.0):
        self.cosmo, self.amplitude, self.ns = cosmo, amplitude, cosmo.ns
        self.splines = {}
        if table is not None:
            table = jnp.asarray(table)
            k = table[:, 0]
            for species, col in {"dm": 1, "baryon": 2, "all": 6}.items():
                t = table[:, col] / table[0, 1]
                valid = (k > 0) & (t > 0) & jnp.isfinite(t)
                x, y = jnp.log(k[valid]), jnp.log(t[valid])
                self.splines[species] = (x, natural_spline(x, y))
            # Match the reference's left Riemann sum, including its integration range.
            lo, hi = k[0] * 1.000001, min(k[-1], 25.0) * 0.999999
            dk = (hi - lo) / 50000
            kval = lo + dk * jnp.arange(50000)
            t = jnp.asarray(self.transfer(kval, "all"))
            kr = kval * 8
            variance = jnp.sum(kval ** (cosmo.ns + 2) * ((jnp.sin(kr) - kr * jnp.cos(kr)) / kr**3) ** 2 * t * t) * 9 / (2 * jnp.pi**2) * dk
            self.amplitude = (cosmo.s8 * cosmo.growth() / cosmo.growth(1)) ** 2 / variance

    def transfer(self, k, species="all"):
        return spline_transfer(k, *self.splines[species])

    def __call__(self, k, species="all"):
        return spectrum_power(jnp.asarray(k), self.amplitude, self.ns, self.splines[species] if self.splines else None)

    def covariance(self, grid, species="all", half_spectrum=False):
        k = grid.k[..., : grid.n // 2 + 1] if half_spectrum else grid.k
        return self(k, species) / grid.dx**3


@partial(jax.jit, static_argnums=(0,))
def grid_points(n, dx, origin):
    return (jnp.moveaxis(jnp.indices((n,) * 3), 0, -1) + 0.5) * dx + origin


@dataclass(eq=False)
class Grid:
    length: float
    n: int
    origin: tuple = (0.0, 0.0, 0.0)
    box: float = None
    flags: jax.Array = field(default=None, repr=False)

    def __post_init__(self):
        if self.length <= 0 or self.n < 2:
            raise ValueError("Grid length must be positive and resolution at least two")
        self.box = self.length if self.box is None else self.box
        self.origin = jnp.asarray(self.origin, dtype=float)
        self.flags = jnp.zeros((self.n,) * 3, bool) if self.flags is None else self.flags
        k = jnp.fft.fftfreq(self.n) * self.n * 2 * jnp.pi / self.length
        self.wavevectors = tuple(k.reshape(tuple(self.n if a == b else 1 for b in range(3))) for a in range(3))

    @cached_property
    def k(self):
        with jax.ensure_compile_time_eval():
            return jnp.sqrt(sum(v * v for v in self.wavevectors))

    @property
    def points(self):
        return grid_points(self.n, self.dx, self.origin)

    @property
    def dx(self):
        return self.length / self.n

    def delta(self, points, center):
        return (jnp.asarray(points) - center + self.box / 2) % self.box - self.box / 2

    def indices(self, points):
        return jnp.floor((jnp.asarray(points) - self.origin) % self.box / self.dx + 1e-10).astype(int) % self.n

    def contains(self, points):
        p = (jnp.asarray(points) - self.origin) % self.box
        return jnp.all(p < self.length - 1e-10, axis=-1)

    def select(self, center, shape, size=0):
        d = self.delta(self.points, center)
        if shape == "sphere":
            self.flags = jnp.sum(d * d, axis=-1) < jnp.float32(size).astype(jnp.float64) ** 2
        elif shape == "ellipse":
            self.flags = jnp.sum((d / jnp.asarray(size, dtype=jnp.float32)) ** 2, axis=-1) < 1
        elif shape == "cube":
            self.flags = jnp.all(jnp.abs(d) < jnp.float32(size).astype(jnp.float64) / 2, axis=-1)
        elif shape == "nearest":
            self.flags = jnp.zeros_like(self.flags).at[tuple(self.indices(center))].set(True)
        else:
            raise ValueError(f"Unknown selection {shape}")

    def expand(self, cells):
        self.flags = dilate(self.flags, cells)


@jax.jit
def cubic_weights(t):
    return jnp.stack((-t * (t - 1) ** 2, 2 - 5 * t * t + 3 * t**3, t + 4 * t * t - 3 * t**3, (t - 1) * t * t), axis=-1) / 2


@partial(jax.jit, static_argnames=("periodic",))
def interpolate(values, coordinates, periodic=True):
    """Tensor product Catmull–Rom interpolation; coordinates are in sample units."""
    base = jnp.floor(coordinates).astype(int)
    weights = cubic_weights(coordinates - base)
    n = values.shape[0]

    def accumulate(term, result):
        i, j, k = term // 16, term // 4 % 4, term % 4
        indices = base + jnp.array([i - 1, j - 1, k - 1])
        indices = indices % n if periodic else jnp.clip(indices, 0, n - 1)
        weight = weights[..., 0, i] * weights[..., 1, j] * weights[..., 2, k]
        return result + values[indices[..., 0], indices[..., 1], indices[..., 2]] * weight.reshape(weight.shape + (1,) * (values.ndim - 3))

    return jax.lax.fori_loop(0, 64, accumulate, jnp.zeros(coordinates.shape[:-1] + values.shape[3:], dtype=values.dtype))


def sample(values, grid, points):
    coordinates = ((jnp.asarray(points) - jnp.asarray(grid.origin)) % grid.box) / grid.dx - 0.5
    return interpolate(values, coordinates, grid.length == grid.box)


@partial(jax.jit, static_argnames=("periodic",))
def interpolate_grid(values, coordinates, periodic=True):
    """Separable cubic interpolation on a Cartesian product of sample axes."""
    for axis, coordinate in enumerate(coordinates):
        base = jnp.floor(coordinate).astype(int)
        index = base[:, None] + jnp.arange(-1, 3)
        index = index % values.shape[axis] if periodic else jnp.clip(index, 0, values.shape[axis] - 1)
        shape = [1] * (values.ndim + 1)
        shape[axis : axis + 2] = index.shape
        values = jnp.sum(jnp.take(values, index, axis=axis) * cubic_weights(coordinate - base).reshape(shape), axis=axis + 1)
    return values


def grid_coordinates(source, target, ranges=None):
    ranges = (slice(0, target.n),) * 3 if ranges is None else ranges
    return tuple(((target.origin[a] + (jnp.arange(r.start, r.stop) + 0.5) * target.dx - source.origin[a]) % source.box) / source.dx - 0.5 for a, r in enumerate(ranges))


def grid_indices(source, target):
    return tuple(jnp.floor(c + 0.5 + 1e-10).astype(int) % source.n for c in grid_coordinates(source, target))


def selected_points(grid, mask):
    from .splice import selected_cells

    cpu = jax.devices("cpu")[0]
    with jax.default_device(cpu):
        mask = jax.device_put(mask, cpu)
        index = selected_cells(mask, size=int(jnp.sum(mask)))
        cells = jnp.stack((index // grid.n**2, index // grid.n % grid.n, index % grid.n), -1)
        points = (cells + 0.5) * grid.dx + jax.device_put(grid.origin, cpu)
    return jax.device_put(points, grid.origin.device)


def sample_grid(values, source, target):
    return interpolate_grid(values, grid_coordinates(source, target), source.length == source.box)


def resample(values, source, target):
    """Sample an aligned grid, averaging child cells when reducing resolution."""
    if target.dx < source.dx * (1 - 1e-8):
        return sample_grid(values, source, target)
    ratio = round(target.dx / source.dx)
    for axis in range(3):
        lower = target.origin[axis] + jnp.arange(target.n) * target.dx
        base = jnp.rint(((lower - source.origin[axis]) % source.box) / source.dx).astype(int)
        index = base[:, None] + jnp.arange(ratio)
        valid = index < source.n if source.length != source.box else jnp.ones_like(index, bool)
        taken = jnp.take(values, jnp.asarray(index % source.n), axis=axis)
        shape = [1] * taken.ndim
        shape[axis : axis + 2] = index.shape
        weights = jnp.asarray(valid).reshape(shape)
        values = jnp.sum(taken * weights, axis=axis + 1) / jnp.maximum(jnp.sum(weights, axis=axis + 1), 1)
    return values


def lowpass(k, cutoff):
    return jax.nn.sigmoid((cutoff - k) / (0.1 * cutoff))


def displacement(density, grid):
    """Zel'dovich displacements, with undefined zero/Nyquist derivatives removed."""
    k2 = grid.k**2
    return jnp.stack([ifft(jnp.where((jnp.abs(v) < jnp.pi / grid.dx * (1 - 1e-12)) & (k2 > 0), 1j * v * density / jnp.where(k2 > 0, k2, 1), 0)) for v in grid.wavevectors], axis=-1)


@jax.jit
def displacement_mode(density, wave, k, dx):
    return jnp.where((jnp.abs(wave) < jnp.pi / dx * (1 - 1e-12)) & (k > 0), 1j * wave * density / jnp.where(k > 0, k * k, 1), 0)


@partial(jax.jit, donate_argnums=(0,))
def store_component(destination, values, axis):
    return destination.at[..., axis].set(values)


def derivative(values, axis, dx):
    return (jnp.roll(values, 2, axis) - 8 * jnp.roll(values, 1, axis) + 8 * jnp.roll(values, -1, axis) - jnp.roll(values, -2, axis)) / (12 * dx)


def log10_split(x):
    """Split log10(2) to preserve legacy shell boundaries at powers of two."""
    mantissa, exponent = jnp.frexp(x)
    exponent = exponent - 1
    return (exponent * 3.69423907715893078616e-13 + 0.434294481903251816668 * jnp.log(2 * mantissa)) + exponent * 0.301029995663611771306


def estimate_power_spectrum(density, grid, spectrum):
    """Logarithmic shell estimates and theoretical power, in physical units."""
    integer = jnp.fft.fftfreq(grid.n) * grid.n
    k = jnp.sqrt(sum(v * v for v in jnp.meshgrid(integer, integer, integer, indexing="ij"))) * (2 * jnp.pi / grid.length)
    lo, hi = 2 * jnp.pi / grid.length, jnp.pi / grid.length * grid.n
    step = log10_split(hi / lo) / 100
    valid = (k >= lo) & (k < hi)
    # Correct reciprocal-multiply division before assigning boundary modes.
    ratio = k / lo
    ratio += (k - ratio * lo) / lo
    bins = jnp.where(valid, ((1 / step) * log10_split(jnp.maximum(ratio, 1))).astype(int), 100).reshape(-1)
    count = jnp.bincount(bins, length=101)[:100]
    average = lambda v: jnp.bincount(bins, weights=jnp.where(valid, v, 0).reshape(-1), length=101)[:100] / jnp.maximum(count, 1)
    columns = [10 ** (log10_split(lo) + step * (jnp.arange(100) + 0.5)), average(k), average(spectrum(k)), average(jnp.abs(fft(density)) ** 2) * grid.dx**3, count]
    return jnp.stack(columns, -1)[count > 0]


class Simulation:
    """Mutable command state; numerical operations use immutable JAX arrays."""

    def __init__(self, cosmology=None, spectrum=None):
        self.cosmo = cosmology or Cosmology()
        self.spectrum = spectrum
        self.grids, self.replaced, self.noise = [], [], []
        self.center = jnp.zeros(3)
        self.seed, self.seed_mode, self.fixed = None, "parallel", False
        self.combined = False
        self.modifications = []
        self.filter_scale = -1
        self.strays = False
        self.autopad = 0
        self.exterior_flags = {}
        self.supersample = self.supersample_gas = self.subsample = 1
        self.eps_norm, self.neighbours = 0.01075, 64
        self.velocity_offset = jnp.zeros(3)
        self.types, self.current_type, self.flagged_type = [], 1, None
        self.baryons_all_levels = False
        self.splice_solver = "cg"

    def base_grid(self, length, n):
        if self.grids or self.noise:
            raise ValueError("Cannot re-initialize the base grid")
        self.grids.append(Grid(float(length), int(n)))
        self.types.append(self.current_type)

    def zoom_grid(self, factor, n, origin=None):
        if self.noise:
            raise ValueError("Trying to initialize a grid after the random field was already drawn")
        parent = self.grids[-1]
        flagged = jnp.argwhere(parent.flags)
        if not len(flagged):
            raise ValueError("Cannot initialise zoom without marking particles to be replaced")
        width = parent.n // int(factor)
        if origin is None:
            # Find a minimal wrapped interval on each axis; expand to requested size.
            starts, spans = [], []
            for axis in range(3):
                v = jnp.unique(flagged[:, axis])
                gaps = jnp.diff(jnp.r_[v, v[0] + parent.n])
                start = v[(jnp.argmax(gaps) + 1) % len(v)]
                span = (v[jnp.argmax(gaps)] - start) % parent.n + 1
                starts.append(start)
                spans.append(span)
            if max(spans) > width and not self.strays:
                raise ValueError("Requested zoom grid is too small for flagged cells")
            origin = jnp.array(starts)
            origin = jnp.where(jnp.max(jnp.array(spans)) < width, origin + (jnp.array(spans) - 1) // 2 - width // 2, origin)
            if width == round(parent.box / parent.dx):
                origin = jnp.zeros_like(origin)
            elif parent.length == parent.box:
                origin %= parent.n
            else:
                origin = jnp.clip(origin, 0, parent.n - width)
        origin = jnp.asarray(origin, int)
        child = Grid(parent.length / int(factor), int(n), parent.origin + origin * parent.dx, parent.box)
        mask = parent.flags.copy()
        if not self.strays:
            mask &= child.contains(parent.points)
        self.replaced.append(mask)
        child.flags = parent.flags[tuple(jnp.moveaxis(parent.indices(child.points), -1, 0))]
        self.grids.append(child)
        self.types.append(self.current_type)
        self.current_type = self.types[-1]

    def select(self, shape, size=0):
        for grid in self.grids:
            grid.select(self.center, shape, size)
        if shape == "nearest":
            for grid in self.grids[:-1]:
                grid.flags = jnp.zeros_like(grid.flags)
        self.exterior_flags = {}
        if self.strays and len(self.grids) > 1 and shape != "nearest":
            for level, _, points, _ in self.particle_layout():
                if not level:
                    continue
                grid = self.grids[level]
                d = grid.delta(points, self.center)
                if shape == "sphere":
                    selected = jnp.sum(d * d, axis=-1) < jnp.float32(size).astype(jnp.float64) ** 2
                elif shape == "ellipse":
                    selected = jnp.sum((d / jnp.asarray(size)) ** 2, axis=-1) < 1
                else:
                    selected = jnp.all(jnp.abs(d) < jnp.float32(size).astype(jnp.float64) / 2, axis=-1)
                outside = points[selected & ~grid.contains(points)]
                self.exterior_flags[level] = jnp.floor(outside / grid.dx + 1e-8).astype(int)

    def draw(self):
        if not self.noise:
            from .random import LegacyRandom

            if self.seed is None:
                raise ValueError("The random number generator has not been seeded")
            rng = LegacyRandom(self.seed)
            self.noise = [rng.field(g.n, self.seed_mode, self.fixed) for g in self.grids]
        if self.spectrum is None:
            self.spectrum = PowerSpectrum(self.cosmo)

    def filters(self, k):
        filters, remaining = [], jnp.ones_like(k)
        for grid in self.grids[:-1]:
            f = lowpass(k, 0.3 * grid.n * 2 * jnp.pi / grid.length)
            filters.append(remaining * f)
            remaining *= jnp.sqrt(1 - f * f)
        return filters + [remaining]

    @partial(jax.jit, static_argnums=(0, 2))
    def metric(self, fields, covector=False):
        result = []
        for i, (grid, value) in enumerate(zip(self.grids, fields)):
            f = self.filters(grid.k)[i]
            if covector and i < len(fields) - 1:
                mask = jnp.asarray(self.grids[i + 1].contains(grid.points))
                value = fft(ifft(value) * (1 - mask)) + fft(ifft(value) * mask) * f
                value = fft(ifft(value) * (1 - mask) + ifft(value * f) * mask)
            else:
                value = value * f * f
            for j, (other, source) in enumerate(zip(self.grids, fields)):
                if i != j:
                    fs = self.filters(other.k)
                    weighted = ifft(source * fs[i] * fs[j] * math.sqrt((grid.dx / other.dx) ** 3))
                    value += fft(resample(weighted, other, grid))
            result.append(value)
        return result

    def covector(self, name, masks=None):
        masks = [g.flags for g in self.grids] if masks is None else masks
        grid = self.grids[-1]
        mask = jnp.asarray(masks[-1], dtype=float)
        local = mask / jnp.sum(mask)
        name = name.lower()
        match name:
            case "overdensity" | "potential":
                pass
            case "vx" | "vy" | "vz":
                local = derivative(local, "xyz".index(name[1]), grid.dx)
            case "lx" | "ly" | "lz":
                axis = "xyz".index(name[1])
                a, b = (axis + 1) % 3, (axis + 2) % 3
                positions = grid.points[masks[-1]]
                center = positions[0] + jnp.mean(grid.delta(positions, positions[0]), axis=0)
                q = grid.delta(grid.points, center)
                local = derivative(local * q[..., a], b, grid.dx) - derivative(local * q[..., b], a, grid.dx)
            case _:
                raise ValueError(f"Unknown modification {name}")
        fields = [None] * len(self.grids)
        fields[-1] = local
        for i in range(len(fields) - 2, -1, -1):
            parent, child = self.grids[i : i + 2]
            coords = ((jnp.asarray(child.points) - jnp.asarray(parent.origin)) % parent.box) / parent.dx - 0.5
            transpose = jax.linear_transpose(lambda v: interpolate(v, coords, parent.length == parent.box), jnp.zeros((parent.n,) * 3))
            fields[i] = transpose(fields[i + 1])[0]
        result = []
        for i, (g, v) in enumerate(zip(self.grids, fields)):
            v = fft(v) * self.filters(g.k)[i] * jnp.sqrt(self.spectrum.covariance(g, "dm"))
            match name:
                case "potential":
                    v *= jnp.where(g.k > 0, -1.5 * self.cosmo.Om / self.cosmo.a * 100**2 / 299792**2 / jnp.where(g.k > 0, g.k**2, 1), 0)
                case "vx" | "vy" | "vz" | "lx" | "ly" | "lz":
                    v *= jnp.where(g.k > 0, -self.cosmo.velocity_factor / jnp.where(g.k > 0, g.k**2, 1), 0)
            result.append(v)
        return result

    @staticmethod
    def dot(a, b):
        return sum(jnp.vdot(x, y).real for x, y in zip(a, b))

    @partial(jax.jit, static_argnums=(0, 3))
    def quadratic(self, fields, masks, scale):
        result = []
        for grid, noise, mask in zip(self.grids, fields, masks):
            transfer = jnp.sqrt(self.spectrum.covariance(grid, "dm"))
            m = jnp.asarray(mask)
            f = 1 - lowpass(grid.k, 2 * jnp.pi / scale)
            v = ifft(fft(ifft(noise * transfer) * m) * f) * m
            v = (v - jnp.sum(v) / jnp.sum(m)) * m / jnp.sum(m)
            result.append(fft(ifft(fft(v) * f) * m))
        result = self.metric(result, covector=True)
        return [x * jnp.sqrt(self.spectrum.covariance(g, "dm")) for x, g in zip(result, self.grids)]

    def calculate(self, name, masks=None, scale=None):
        self.draw()
        masks = [g.flags for g in self.grids] if masks is None else masks
        if name.lower() == "variance" and self.combined:
            total = 0.0
            for i, (grid, density, mask) in enumerate(zip(self.grids, self.density, masks)):
                mask = jnp.asarray(mask)
                f = 1 - lowpass(grid.k, 2 * jnp.pi / self.filter_scale)
                value = ifft(fft(density * mask) * f) * mask
                value = (value - jnp.sum(value) / jnp.sum(mask)) * mask / jnp.sum(mask)
                value = ifft(fft(value) * f) * mask
                if i < len(self.grids) - 1:
                    value *= 1 - jnp.asarray(self.grids[i + 1].contains(grid.points))
                total += jnp.sum(value * density)
            return total
        if name.lower() == "variance":
            return self.dot(self.quadratic(self.noise, masks, self.filter_scale if scale is None else scale), self.noise)
        if self.combined and name.lower() == "overdensity":
            return jnp.sum(self.density[-1] * jnp.asarray(masks[-1])) / jnp.sum(masks[-1])
        return self.dot(self.covector(name, masks), self.noise)

    def modify(self, name, mode, target):
        self.draw()
        if mode.lower() not in ("absolute", "relative"):
            raise ValueError("Modification type must be either 'relative' or 'absolute'")
        masks = [g.flags.copy() for g in self.grids]
        target = jnp.asarray(target, jnp.float32).astype(jnp.float64)
        if mode.lower() == "relative":
            target *= self.calculate(name)
        self.modifications.append((name.lower(), target, masks, self.filter_scale))

    def apply_modifications(self):
        self.draw()
        alphas, targets = [], []
        for name, target, masks, scale in self.modifications:
            if name == "variance":
                continue
            alpha = self.covector(name, masks)
            for old, old_target in zip(alphas, targets):
                overlap = self.dot(alpha, self.metric(old))
                alpha = [x - overlap * y for x, y in zip(alpha, old)]
                target -= overlap * old_target
            norm = jnp.sqrt(self.dot(alpha, self.metric(alpha)))
            alphas.append([x / norm for x in alpha])
            targets.append(target / norm)
        for alpha, target in zip(alphas, targets):
            change = target - self.dot(alpha, self.noise)
            self.noise = [x + change * y for x, y in zip(self.noise, self.metric(alpha))]
        # The reference passes its linear update vectors to the quadratic projection.
        alphas = [self.metric(alpha) for alpha in alphas]
        for name, target, masks, scale in self.modifications:
            if name != "variance":
                continue

            @jax.jit
            def iterate(fields, steps):
                start = self.dot(self.quadratic(fields, masks, scale), fields)

                def step(i, fields):
                    goal = start + (i + 1) * ((target - start) / steps)
                    pushed = self.quadratic(fields, masks, scale)
                    value = self.dot(pushed, fields)
                    norm = jnp.sqrt(self.dot(pushed, self.metric(pushed)))
                    for alpha in alphas:
                        overlap = self.dot(pushed, alpha)
                        pushed = [x - overlap * y for x, y in zip(pushed, alpha)]
                    change = 0.5 * (goal - value) / norm / jnp.sqrt(self.dot(pushed, self.metric(pushed)))
                    return [x + change * y for x, y in zip(fields, self.metric(pushed))]

                return jax.lax.fori_loop(0, steps, step, fields)

            trial = iterate(self.noise, jnp.int32(10))
            error = jnp.abs(self.dot(self.quadratic(trial, masks, scale), trial) - target)
            steps = jnp.maximum(10, 10 * jnp.ceil(jnp.sqrt(error / (target * 0.001)))).astype(jnp.int32)
            self.noise = jax.lax.cond(steps > 10, lambda _: iterate(self.noise, steps), lambda _: trial, None)

    def chi2(self):
        self.draw()
        return self.dot(self.metric(self.noise, True), self.noise)

    def combine(self):
        self.draw()
        if self.combined:
            return
        density = [x[..., : g.n // 2 + 1] * jnp.sqrt(self.spectrum.covariance(g, half_spectrum=True)) for x, g in zip(self.noise, self.grids)]
        self.density = self.combine_levels(density, half_spectrum=True)
        # Spill retained fields temporarily while FFT workspaces coexist.
        self.noise, self.density = jax.device_put((self.noise, self.density), jax.devices("cpu")[0])
        jax.block_until_ready((self.noise, self.density))
        self.offsets = [jnp.empty((g.n,) * 3 + (3,)) for g in self.grids]
        for axis in range(3):
            components = self.combine_levels([displacement_mode(v, g.wavevectors[axis][..., : g.n // 2 + 1], g.k[..., : g.n // 2 + 1], g.dx) for v, g in zip(density, self.grids)], half_spectrum=True)
            for i, values in enumerate(components):
                self.offsets[i] = store_component(self.offsets[i], values, axis)
            del components, values
        del density
        jax.block_until_ready(self.offsets)
        self.noise, self.density = jax.device_put((self.noise, self.density), self.grids[0].origin.device)
        self.combined = True

    def combine_levels(self, fields, half_spectrum=False):
        """Production synthesis, from independent Fourier fields to real density."""
        inverse = lambda v, g: jnp.fft.irfftn(v, s=(g.n,) * 3, norm="ortho") if half_spectrum else ifft(v)
        forward = partial(jnp.fft.rfftn, norm="ortho") if half_spectrum else fft
        wave = lambda g: g.k[..., : g.n // 2 + 1] if half_spectrum else g.k
        result = [inverse(fields[0], self.grids[0])]
        for i, child in enumerate(self.grids[1:], 1):
            parent = self.grids[i - 1]
            cutoff = 0.3 * parent.n * 2 * jnp.pi / parent.length
            high = jnp.sqrt(1 - lowpass(wave(child), cutoff) ** 2)
            result.append(inverse(fields[i] * high, child) + sample_grid(inverse(forward(result[-1]) * lowpass(wave(parent), cutoff), parent), parent, child))
        return tuple(result)

    def synthesize(self, latents):
        """Real unit-variance latents -> production multilevel density (linear)."""
        return self.combine_levels([fft(v) * jnp.sqrt(self.spectrum.covariance(g)) for v, g in zip(latents, self.grids)])

    def splice(self, seed, accuracy=1e-6, solver=None, border=4):
        """Insert donor density on owned selected cells; minimize latent change.

        Implements splicing-multilevel, commit 715d758: solve MT T^T M^T lambda
        using the exact transpose of production synthesis, without lifting DC.
        """
        from .random import LegacyRandom
        from .splice import SpliceOperator, axpy, solve

        self.draw()
        solver = self.splice_solver if solver is None else solver
        if self.combined or not 0 < accuracy < 1 or solver not in ("cg", "minres") or border < 1:
            raise ValueError("Splicing requires uncombined fields, 0 < accuracy < 1, cg/minres, and positive border safety")
        masks = []
        for i, g in enumerate(self.grids):
            mask = g.flags
            if i:
                mask = mask & self.replaced[i - 1][jnp.ix_(*grid_indices(self.grids[i - 1], g))]
            if i < len(self.grids) - 1:
                mask = mask & ~self.replaced[i]
            for ancestor in self.grids[1 : i + 1]:
                axes = tuple((c + 0.5 >= border) & (c + 0.5 < ancestor.n - border) for c in grid_coordinates(ancestor, g))
                safe = axes[0][:, None, None] & axes[1][None, :, None] & axes[2][None, None, :]
                if jnp.any(mask & ~safe):
                    raise ValueError("Splice region is too close to a zoom edge; enlarge the zoom grid")
            masks.append(mask)
        if not jnp.any(jnp.stack([jnp.any(mask) for mask in masks])):
            return
        original = tuple(v[..., : g.n // 2 + 1] for v, g in zip(self.noise, self.grids))
        operator = SpliceOperator(self.grids, [self.spectrum.covariance(g, half_spectrum=True) for g in self.grids], masks)
        forward, transpose = operator.forward, operator.transpose
        current = forward(original)
        maximum = jnp.maximum(100, 4 * sum(jnp.sum(m) for m in masks))
        dtype = jnp.float32 if accuracy >= 1e-5 else jnp.float64
        rng = LegacyRandom(seed)
        donor = tuple(rng.field(g.n, self.seed_mode, half_spectrum=True) for g in self.grids)
        target = forward(donor)
        rhs = axpy(target, current, -1)
        atol = 8 * jnp.finfo(jnp.float64).eps * jnp.maximum(jnp.sqrt(self.dot(target, target)), jnp.sqrt(self.dot(current, current)))
        dual, error, iterations = solve(operator, original, target, current, accuracy, atol, maximum, solver=solver, dtype=dtype)
        tolerance = jnp.maximum(accuracy * jnp.sqrt(self.dot(rhs, rhs)), atol)
        if not error <= tolerance:
            raise RuntimeError("Splice solver did not satisfy the requested density tolerance")
        self.noise = []  # The accepted candidate only needs the retained half spectra.
        self.noise = [fft(jnp.fft.irfftn(v, s=(g.n,) * 3, norm="ortho")) for v, g in zip(axpy(original, transpose(dual)), self.grids)]
        self.splice_residual = error
        self.splice_iterations = int(iterations)

    def particle_hierarchy(self, gas=False):
        """Sampling grids and refinement masks, including virtual padding layers."""
        hierarchy, masks = [], []
        for level, grid in enumerate(self.grids):
            factor = (self.supersample_gas if gas else self.supersample) if level == len(self.grids) - 1 else 1
            factor /= self.subsample if level == 0 and not gas else 1
            output = Grid(grid.length, round(grid.n * factor), grid.origin, grid.box)
            output.soft_dx = max(output.dx, grid.dx)
            if hierarchy:
                previous = hierarchy[-1][1]
                selected = jnp.zeros((previous.n,) * 3, bool)
                parent = self.grids[level - 1]
                points = selected_points(parent, self.replaced[level - 1])
                selected = selected.at[tuple(previous.indices(points).T)].set(True)
                while self.autopad and not gas and previous.dx / output.dx > 2 + 1e-8:
                    expanded = dilate(selected, self.autopad)
                    masks.append(expanded)
                    virtual = Grid(previous.length, previous.n * 2, previous.origin, previous.box)
                    virtual.soft_dx = max(virtual.dx, min(previous.dx, self.grids[level - 1].dx))
                    hierarchy.append((level - 1, virtual, self.types[level - 1]))
                    selected = selected[tuple(jnp.moveaxis(previous.indices(virtual.points), -1, 0))]
                    previous = virtual
                masks.append(selected)
            hierarchy.append((level, output, self.types[level]))
        return hierarchy, masks

    def particle_layout(self, gas=False):
        """Return (source level, sampling grid, points, type) in mapper order."""
        hierarchy, masks = self.particle_hierarchy(gas)
        blocks = []
        for i, (level, output, kind) in enumerate(hierarchy):
            if gas and not (self.baryons_all_levels or level == len(self.grids) - 1):
                continue
            if i == 0:
                points = output.points.reshape(-1, 3)
            else:
                parent = hierarchy[i - 1][1]
                origins = parent.points[masks[i - 1]] - parent.dx / 2
                ratio = round(parent.dx / output.dx)
                offsets = (jnp.indices((ratio,) * 3).reshape(3, -1).T + 0.5) * output.dx
                points = (origins[:, None, :] + offsets).reshape(-1, 3) % output.box
            if i < len(masks):
                index = output.indices(points)
                points = points[~masks[i][tuple(index.T)]]
            blocks.append((level, output, points, 0 if gas else kind))
        return blocks

    def particle_count(self):
        """Count the layout without allocating particle coordinate arrays."""
        if self.strays:
            return sum(len(b[2]) for b in self.particle_layout())
        hierarchy, masks = self.particle_hierarchy()
        total = 0
        for i, (_, grid, _) in enumerate(hierarchy):
            owned = jnp.ones((grid.n,) * 3, bool)
            if i:
                parent = hierarchy[i - 1][1]
                owned = masks[i - 1][jnp.ix_(*grid_indices(parent, grid))]
            total += jnp.sum(owned & ~masks[i] if i < len(masks) else owned)
        return int(total)

    def adapt_mask(self, padding):
        if len(self.grids) < 3:
            raise ValueError("Adapting the mask requires there to be more than one zoom level")
        for level in range(len(self.grids) - 2, 0, -1):
            fine, coarse = self.grids[level], self.grids[level - 1]
            mask = jnp.zeros((coarse.n,) * 3, bool)
            mask = mask.at[tuple(coarse.indices(fine.points[self.replaced[level]]).T)].set(True)
            self.replaced[level - 1] = dilate(mask, padding)

    def export_grids(self, center=None):
        """Grafic hierarchy, including virtual powers-of-two resolutions and masks."""
        specs = []
        for i, grid in enumerate(self.grids):
            if i == 0:
                factors = [2**p for p in range(round(math.log2(self.subsample)), 0, -1)]
            else:
                contrast = self.grids[i - 1].dx / grid.dx
                factors = [2**p for p in range(round(math.log2(contrast)) - 1, 0, -1)]
            specs.extend((i, grid.n // factor) for factor in factors)
            specs.append((i, grid.n))
        specs.extend((len(self.grids) - 1, self.grids[-1].n * 2**p) for p in range(1, round(math.log2(self.supersample)) + 1))
        shift = jnp.zeros(3)
        result = []
        center = jnp.full(3, self.grids[0].box / 2) if center is None else center
        for i, n in specs:
            source = self.grids[i]
            if source.length == source.box:
                shift = jnp.trunc((source.box / 2 - center) / (source.length / n)) * (source.length / n)
            target = Grid(source.length, n, jnp.zeros(3) if source.length == source.box else (source.origin + shift) % source.box, source.box)

            points = (target.points - shift) % source.box
            mask = jnp.ones((n,) * 3, bool)
            if self.replaced:
                parent_level = min(i, len(self.replaced) - 1)
                parent = self.grids[parent_level]
                if target.dx <= parent.dx:
                    mask = self.replaced[parent_level][tuple(jnp.moveaxis(parent.indices(points), -1, 0))]
                else:
                    mask = jnp.zeros_like(mask)
                    selected = (parent.points[self.replaced[parent_level]] + shift) % source.box
                    mask = mask.at[tuple(target.indices(selected).T)].set(True)
            if (i, n) == specs[-1]:
                mask = jnp.ones_like(mask)
            result.append((i, target, points, mask))
        # Legacy Grafic masks identify their first level by raw cell ID range.
        # Preserve this convention even when an added coarse grid claims those IDs.
        if self.replaced and specs[0][1] != self.grids[0].n:
            ids = jnp.flatnonzero(self.replaced[0])
            first = result[0][1]
            ids = ids[ids < first.n**3]
            if len(ids):
                mask = jnp.zeros((first.n,) * 3, bool)
                mask = mask.reshape(-1).at[ids].set(True).reshape(mask.shape)
                shift_cells = jnp.trunc((first.box / 2 - center) / first.dx).astype(int)
                mask = jnp.roll(mask, tuple(shift_cells), (0, 1, 2))
                updated = []
                for i, (level, target, points, _) in enumerate(result):
                    if i:
                        mask = mask[tuple(jnp.moveaxis(first.indices(target.points), -1, 0))]
                    if i == len(result) - 1:
                        mask = jnp.ones_like(mask)
                    updated.append((level, target, points, mask))
                    first = target
                result = updated
        return result

    def evaluate(self, fields, level, output, points, aligned=False):
        grid = self.grids[level]
        if aligned and math.isclose(output.dx, grid.dx, rel_tol=1e-8):
            value = fields[level][tuple(grid.indices(points).T)]
        elif output.dx > grid.dx * (1 + 1e-8):
            averaged = resample(fields[level], grid, output)
            value = jnp.asarray(averaged)[tuple(output.indices(points).T)]
        else:
            value = jnp.asarray(sample(fields[level], grid, points))
        if level:
            valid = grid.contains(points)
            if aligned and bool(jnp.all(valid)):
                return value
            valid = valid.reshape(valid.shape + (1,) * (value.ndim - valid.ndim))
            value = jnp.where(valid, value, self.evaluate(fields, level - 1, output, points, aligned))
        return value

    def particle_block(self, level, output, points, ptype, gas):
        points = jnp.asarray(points)
        grid = self.grids[level]
        offsets = self.evaluate(self.offsets, level, output, points, aligned=True)
        split = self.baryons_all_levels or level == len(self.grids) - 1
        fraction = self.cosmo.Ob / self.cosmo.Om if gas else (1 - self.cosmo.Ob / self.cosmo.Om if split else 1)
        mass = 27.744948 * self.cosmo.Om * output.dx**3 * fraction
        n = len(points)
        kind = jnp.full(n, ptype)
        if self.flagged_type is not None and not gas and level == len(self.grids) - 1:
            kind = jnp.where(grid.flags[tuple(grid.indices(points).T)], self.flagged_type, kind)
        return dict(pos=(points + offsets) % grid.box, vel=offsets * self.cosmo.velocity_factor + self.velocity_offset, mass=jnp.full(n, mass), eps=jnp.full(n, output.soft_dx * self.eps_norm), gas=jnp.full(n, gas), ptype=kind,
                    density=self.evaluate(self.density, level, output, points, aligned=True))

    def particles(self, tipsy=False):
        self.combine()
        blocks = []
        species = [True, False] if tipsy else [False, True]
        for gas in species:
            if gas and self.cosmo.Ob == 0:
                continue
            for level, output, points, ptype in self.particle_layout(gas):
                blocks.append(self.particle_block(level, output, points, ptype, gas))
        result = {key: jnp.concatenate([block[key] for block in blocks]) for key in blocks[0]}
        if self.flagged_type is not None:
            flagged = result["ptype"] == self.flagged_type
            order = jnp.r_[jnp.flatnonzero(~flagged), jnp.flatnonzero(flagged)]
            result = {key: value[order] for key, value in result.items()}
        result["id"] = jnp.arange(len(result["mass"]), dtype=jnp.uint64)
        if not tipsy:
            order = jnp.argsort(result["ptype"], stable=True)
            result = {key: value[order] for key, value in result.items()}
        return result


class Geometry:
    """Particle IDs and cell-volume mapping, without field generation or file access.

    Input IDs refer to mapper order. Gas IDs are ignored when flagging, matching genetIC.
    """

    def __init__(self, simulation, format="tipsy", center_output=False):
        self.sim, self.format, self.center_output = simulation, format, center_output

    def layout_cells(self):
        s = self.sim
        if self.format == "grafic":
            blocks = [(level, grid, points.reshape(-1, 3), 0) for level, grid, points, _ in s.export_grids()]
        else:
            order = [True, False] if self.format == "tipsy" else [False, True]
            blocks = [block for gas in order if not gas or s.cosmo.Ob > 0 for block in s.particle_layout(gas)]
        points = jnp.concatenate([b[2] for b in blocks])
        sizes = jnp.concatenate([jnp.full(len(b[2]), b[1].dx) for b in blocks])
        levels = jnp.concatenate([jnp.full(len(b[2]), b[0], int) for b in blocks])
        if s.flagged_type is not None:
            grid = s.grids[-1]
            flags = grid.flags[tuple(grid.indices(points).T)] & (levels == len(s.grids) - 1)
            order = jnp.r_[jnp.flatnonzero(~flags), jnp.flatnonzero(flags)]
            points, sizes, levels = points[order], sizes[order], levels[order]
        return points, sizes, levels

    def flagged_ids(self):
        points, sizes, levels = self.layout_cells()
        selected = jnp.zeros(len(points), bool)
        for i, grid in enumerate(self.sim.grids):
            mask = levels == i
            if not jnp.any(mask):
                continue
            p, widths = points[mask], sizes[mask]
            valid = grid.contains(p)
            flags = grid.flags[tuple(grid.indices(p).T)] & valid
            for width in jnp.unique(widths[widths > grid.dx * (1 + 1e-8)]):
                marked = grid.points[grid.flags]
                encode = lambda pos: jnp.floor(((pos - grid.origin) % grid.box) / width + 1e-9).astype(int)
                bins = encode(marked)
                q = encode(p)
                radix = round(grid.box / width) + 1
                codes = lambda x: (x[:, 0] * radix + x[:, 1]) * radix + x[:, 2]
                flags |= jnp.isin(codes(q), codes(bins)) & (widths == width)
            if i in self.sim.exterior_flags:
                outside = ~valid
                codes = jnp.floor((p[outside] % grid.box) / grid.dx + 1e-8).astype(int)
                radix = round(grid.box / grid.dx) + 1
                encode = lambda a: (a[:, 0] * radix + a[:, 1]) * radix + a[:, 2]
                flags = flags.at[outside].set(jnp.isin(encode(codes), encode(self.sim.exterior_flags[i])))
            else:
                for parent in self.sim.grids[i - 1 :: -1]:
                    outside = ~valid
                    flags = flags.at[outside].set(parent.flags[tuple(parent.indices(p[outside]).T)])
                    valid |= parent.contains(p)
            selected = selected.at[mask].set(flags)
        if self.sim.cosmo.Ob > 0 and self.format != "grafic":
            gas_count = sum(len(b[2]) for b in self.sim.particle_layout(True))
            if self.format == "tipsy":
                selected = selected.at[:gas_count].set(False)
            else:
                selected = selected.at[-gas_count:].set(False)
        return jnp.flatnonzero(selected)

    def flag_regions(self, points, sizes, merge=False, levels=None):
        if not merge:
            self.sim.exterior_flags = {}
        for level, grid in enumerate(self.sim.grids):
            if not merge:
                grid.flags = jnp.zeros_like(grid.flags)
            chosen = jnp.ones(len(points), bool) if levels is None else jnp.asarray(levels) == level
            p, widths = points[chosen], sizes[chosen]
            if not len(p):
                continue
            ratios = jnp.maximum(1, jnp.rint(widths / grid.dx)).astype(int)
            maximum = int(jnp.max(ratios))
            offsets = jnp.moveaxis(jnp.indices((maximum,) * 3), 0, -1).reshape(-1, 3)
            cells = (p[:, None, :] + (offsets + 0.5) * widths[:, None, None] / ratios[:, None, None] - widths[:, None, None] / 2) % grid.box
            active = jnp.all(offsets[None, :, :] < ratios[:, None, None], axis=-1)
            inside = grid.contains(cells)
            if level and self.sim.strays:
                outside = cells[active & ~inside]
                if len(outside):
                    codes = jnp.floor(outside / grid.dx + 1e-8).astype(int)
                    self.sim.exterior_flags[level] = jnp.concatenate([self.sim.exterior_flags.get(level, jnp.empty((0, 3), int)), codes])
            cells = cells[active & inside]
            grid.flags = grid.flags.at[tuple(grid.indices(cells).T)].set(True)

    def ancestor_regions(self, ids):
        """Decode padded IDs through each mapper layer, preserving its index ordering.

        Child cells follow grid order; replaced parent particles follow mapper order.
        Those orders can differ in nested padding, so physical nearest-parent lookup
        does not reproduce the legacy flagging operation.
        """
        hierarchy, masks = self.sim.particle_hierarchy()
        base = hierarchy[0][1]
        points = base.points.reshape(-1, 3)
        stages = jnp.zeros(len(points), int)
        nodes = []
        for stage, (level, grid, _) in enumerate(hierarchy[1:], 1):
            parent = hierarchy[stage - 1][1]
            selected = (stages == stage - 1) & masks[stage - 1][tuple(parent.indices(points).T)]
            keep, replaced = jnp.flatnonzero(~selected), jnp.flatnonzero(selected)
            ratio = round(parent.dx / grid.dx)
            origins = parent.points[masks[stage - 1]] - parent.dx / 2
            offsets = (jnp.indices((ratio,) * 3).reshape(3, -1).T + 0.5) * grid.dx
            children = (origins[:, None, :] + offsets).reshape(-1, 3) % grid.box
            nodes.append((keep, replaced, children, grid, level, ratio**3))
            points = jnp.concatenate([points[keep], children])
            stages = jnp.r_[stages[keep], jnp.full(len(children), stage)]
        ids = jnp.asarray(ids)
        if self.sim.cosmo.Ob > 0 and self.format == "tipsy":
            ids = ids - sum(len(b[2]) for b in self.sim.particle_layout(True))
        ids = ids[(ids >= 0) & (ids < len(points))]
        regions, widths, levels = [], [], []
        for keep, replaced, children, grid, level, volume_ratio in reversed(nodes):
            high = ids >= len(keep)
            child = ids[high] - len(keep)
            regions.append(children[child])
            widths.append(jnp.full(len(child), grid.dx))
            levels.append(jnp.full(len(child), level))
            ids = jnp.unique(jnp.r_[keep[ids[~high]], replaced[child // volume_ratio]])
        regions.append(base.points.reshape(-1, 3)[ids])
        widths.append(jnp.full(len(ids), base.dx))
        levels.append(jnp.zeros(len(ids), int))
        return (jnp.concatenate(regions), jnp.concatenate(widths), jnp.concatenate(levels))

    def transfer_ids(self, source, ids, merge=False):
        points, sizes, levels = source.layout_cells()
        points, sizes, levels = (points[ids].copy(), sizes[ids].copy(), levels[ids].copy())
        levels = jnp.minimum(levels, len(self.sim.grids) - 1)
        extra_points, extra_sizes, extra_levels = [points], [sizes], [levels]
        # Flagging a child also flags its ancestors in the reference mapper.
        for level in range(min(len(source.sim.grids), len(self.sim.grids)) - 1):
            use = levels > level
            parent = source.sim.grids[level]
            width = parent.dx * (source.sim.subsample if level == 0 and source.format != "grafic" else 1)
            centers = jnp.floor((points[use] - parent.origin) / width) * width + parent.origin + width / 2
            extra_points.append(centers)
            extra_sizes.append(jnp.full(len(centers), width))
            extra_levels.append(jnp.full(len(extra_points[-1]), level))
        for level in range(len(source.sim.grids), len(self.sim.grids)):
            use = levels == len(source.sim.grids) - 1
            extra_points.append(points[use])
            extra_sizes.append(sizes[use])
            extra_levels.append(jnp.full(len(extra_points[-1]), level))
        if source.sim.autopad and source.format != "grafic":
            p, w, level = source.ancestor_regions(ids)
            extra_points = [p]
            extra_sizes = [w]
            extra_levels = [jnp.minimum(level, len(self.sim.grids) - 1)]
        self.flag_regions(jnp.concatenate(extra_points), jnp.concatenate(extra_sizes), merge, jnp.concatenate(extra_levels))
        if self.format == "grafic":
            for i in range(len(self.sim.grids) - 1, 0, -1):
                fine, coarse = self.sim.grids[i], self.sim.grids[i - 1]
                marked = fine.points[fine.flags]
                coarse.flags = coarse.flags.at[tuple(coarse.indices(marked).T)].set(True)
        return points
