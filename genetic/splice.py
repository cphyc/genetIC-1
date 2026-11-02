"""Compact selected-cell splice operators and preconditioned Krylov solvers."""
from functools import partial
import jax
import jax.numpy as jnp
from .core import grid_coordinates, interpolate_grid, lowpass

fft = partial(jnp.fft.rfftn, norm="ortho")
selected_cells = jax.jit(jnp.flatnonzero, static_argnames=("size",))

@partial(jax.jit, static_argnames=("n",))
def selection_bounds(index, n):
    if not len(index):
        return jnp.array([[0, 1]]*3)
    coordinates = (index//n**2, index//n % n, index % n)
    return jnp.stack([jnp.array([jnp.min(v), jnp.max(v)+1]) for v in coordinates])

@partial(jax.jit, static_argnames=("stride",))
def prepare_kernels(covariance, k, high, low, stride):
    covariance, k = covariance[..., :k.shape[0]//2+1], k[..., :k.shape[0]//2+1]
    root = jnp.sqrt(covariance)
    if high is not None:
        root *= jnp.sqrt(1-lowpass(k, high)**2)
    local = covariance[::stride, ::stride, ::stride]
    return root, 1/jnp.where(local > 0, local, 1), lowpass(k, low) if low is not None else None


@jax.tree_util.register_pytree_node_class
class SpliceOperator:
    """MT and its adjoint, using real FFTs and separable grid interpolation.

    Dual vectors contain only selected cells. The finest interpolation covers
    their bounding box; ancestors remain complete for propagation to children.
    Gram applications retain latent corrections in Fourier space.
    """
    def __init__(self, grids, covariances, masks):
        self.shapes = [(g.n,)*3 for g in grids]
        # Compact selection metadata on CPU; avoid large GPU prefix-scan
        # workspaces and their one-off compilation for each selected count.
        with jax.default_device(jax.devices("cpu")[0]):
            selection = [jax.device_put(m, jax.devices("cpu")[0]) for m in masks]
            indices = [selected_cells(m, size=int(jnp.sum(m))) for m in selection]
            bounds = [tuple(slice(lo, hi) for lo, hi in selection_bounds(index, grid.n).tolist()) for index, grid in zip(indices, grids)]
            local_indices = selected_cells(selection[-1][bounds[-1]], size=len(indices[-1]))
        self.indices = [jax.device_put(v, masks[0].device) for v in indices]
        self.crop = bounds[-1]
        self.local_indices = jax.device_put(local_indices, masks[0].device)
        self.coordinates, self.periodic, self.low = [], [], []
        self.root, self.inverse, self.pre_shapes, self.pre_indices = [], [], [], []
        for i, (grid, covariance) in enumerate(zip(grids, covariances)):
            # An inverse-covariance stencil is local: apply it on a padded
            # selected-cell box instead of FFTing an almost empty full grid.
            width = max(r.stop-r.start for r in bounds[i])
            size = min(grid.n, 1 << (width+3).bit_length())
            size = size if grid.n % size == 0 else grid.n
            stride = grid.n//size
            high = .3*grids[i-1].n*2*jnp.pi/grids[i-1].length if i else None
            low = .3*grid.n*2*jnp.pi/grid.length if i < len(grids)-1 else None
            root, inverse, low = prepare_kernels(covariance, grid.k, high, low, stride)
            self.inverse.append(inverse)
            if low is not None:
                self.low.append(low)
            self.pre_shapes.append((size,)*3)
            index = self.indices[i]
            coordinates = (index//grid.n**2, index//grid.n % grid.n, index % grid.n)
            offsets = tuple((v-r.start+(size-(r.stop-r.start))//2) % size for v, r in zip(coordinates, bounds[i]))
            self.pre_indices.append((offsets[0]*size+offsets[1])*size+offsets[2])
            if i:
                parent = grids[i-1]
                ranges = self.crop if i == len(grids)-1 else (slice(0, grid.n),)*3
                coordinates = grid_coordinates(parent, grid, ranges)
                self.coordinates.append(coordinates)
                self.periodic.append(parent.length == parent.box)
            self.root.append(root)

    def tree_flatten(self):
        arrays = (self.indices, self.local_indices, self.root, self.inverse, self.low, self.coordinates, self.pre_indices)
        metadata = (tuple(self.shapes), tuple((r.start, r.stop) for r in self.crop), tuple(self.periodic), tuple(self.pre_shapes))
        return arrays, metadata

    @classmethod
    def tree_unflatten(cls, metadata, arrays):
        self = object.__new__(cls)
        self.shapes, bounds, self.periodic, self.pre_shapes = metadata
        self.crop = tuple(slice(a, b) for a, b in bounds)
        self.indices, self.local_indices, self.root, self.inverse, self.low, self.coordinates, self.pre_indices = arrays
        return self

    def interpolate(self, i, values):
        return interpolate_grid(values, self.coordinates[i], self.periodic[i])

    def adjoint(self, i, values):
        return jax.linear_transpose(partial(self.interpolate, i), jax.ShapeDtypeStruct(self.shapes[i], values.dtype))(values)[0]

    def embed(self, values, i, cropped=False):
        shape = tuple(r.stop-r.start for r in self.crop) if cropped else self.shapes[i]
        indices = self.local_indices if cropped else self.indices[i]
        return jnp.zeros(shape, values.dtype).reshape(-1).at[indices].set(values).reshape(shape)

    def inverse_fft(self, values, i):
        return jnp.fft.irfftn(values, s=self.shapes[i], norm="ortho")

    def combine(self, fields):
        result = []
        for i, value in enumerate(fields):
            density = self.inverse_fft(value, i)
            if i == len(fields)-1:
                density = density[self.crop]
            if i:
                density += self.interpolate(i-1, self.inverse_fft(previous*self.low[i-1], i-1))
            indices = self.local_indices if i == len(fields)-1 else self.indices[i]
            result.append(density.reshape(-1)[indices])
            if i < len(fields)-1:
                previous = value if i == 0 else fft(density)
        return tuple(result)

    def pullback(self, fields):
        result = [None]*len(fields)
        for i in range(len(fields)-1, -1, -1):
            value = fft(self.embed(fields[i], i))
            if i < len(fields)-1:
                value += self.low[i]*fft(self.adjoint(i, density))
            result[i] = value
            if i:
                density = self.embed(fields[i], i, cropped=True) if i == len(fields)-1 else self.inverse_fft(value, i)
        return tuple(result)

    @jax.jit
    def forward(self, fields):
        return self.combine(tuple(v*r for v, r in zip(fields, self.root)))

    @jax.jit
    def transpose(self, fields):
        return tuple(v*r for v, r in zip(self.pullback(fields), self.root))

    @jax.jit
    def gram(self, fields):
        return self.combine(tuple(v*r*r for v, r in zip(self.pullback(fields), self.root)))

    @jax.jit
    def precondition(self, fields):
        result = []
        for v, c, index, shape in zip(fields, self.inverse, self.pre_indices, self.pre_shapes):
            values = jnp.zeros(shape, v.dtype).reshape(-1).at[index].set(v).reshape(shape)
            result.append(jnp.fft.irfftn(fft(values)*c, s=shape, norm="ortho").reshape(-1)[index])
        return tuple(result)


@partial(jax.jit, static_argnames=("solver", "dtype"))
def solve(operator, original, target, current, accuracy, atol, maximum, solver, dtype=jnp.float64):
    rhs = axpy(target, current, -1)
    cast = lambda values, kind: jax.tree.map(lambda v: v.astype(kind), values)
    physical = lambda dual: axpy(target, operator.forward(axpy(original, operator.transpose(cast(dual, jnp.float64)))), -1)
    if dtype == jnp.float64:
        return krylov(operator.gram, rhs, operator.precondition, physical, accuracy, atol, maximum, solver)
    # Loose tolerances can use fast FFTs, with residual replacement in double
    # precision and final acceptance on the actual double-precision candidate.
    work = jax.tree.map(lambda v: v.astype(dtype) if jnp.issubdtype(v.dtype, jnp.floating) else v, operator)
    apply = lambda dual: cast(work.gram(cast(dual, dtype)), jnp.float64)
    precondition = lambda dual: cast(work.precondition(cast(dual, dtype)), jnp.float64)
    residual = lambda dual: axpy(rhs, operator.gram(dual), -1)
    dual, _, iterations = krylov(apply, rhs, precondition, residual, accuracy, atol, maximum, solver, check_interval=20)
    error = jnp.sqrt(sum(jnp.vdot(v, v).real for v in physical(dual)))
    return cast(dual, jnp.float64), error, iterations


def axpy(a, b, scale=1):
    return jax.tree.map(lambda x, y: x+scale*y, a, b)

def krylov(operator, b, precondition, residual, rtol, atol, maximum, solver, check_interval=10):
    """Preconditioned CG/MINRES with verification in output-density units."""
    dot = lambda a, b: sum(jnp.vdot(x, y).real for x, y in zip(a, b))
    scale = lambda a, t: jax.tree.map(lambda x: x*t, a)
    zero = scale(b, 0)
    norm = lambda v: jnp.sqrt(dot(v, v))
    initial_norm = norm(b)
    tolerance = jnp.maximum(rtol*initial_norm, atol)
    def progress(iteration, relative):
        label = "preconditioned residual" if solver == "minres" else "residual"
        print(f"{solver.upper()} iteration {int(iteration):<4d} {label}={float(relative):.2e}", flush=True)
    def finish(state):
        error = state[-1]  # Acceptance and the iteration limit both verify it.
        def report(iteration, relative):
            print(f"{solver.upper()} ended after {int(iteration)} iterations; true relative residual={float(relative):.3e}", flush=True)
        jax.debug.callback(report, state[0], error/jnp.where(initial_norm > 0, initial_norm, 1), ordered=True)
        return state[1], error, state[0]
    z = precondition(b)
    if solver == "cg":
        def step(state):
            i, x, r, p, rho, error = state
            image = operator(p)
            alpha = rho/dot(p, image)
            x, recursive = axpy(x, p, alpha), axpy(r, image, -alpha)
            check = (norm(recursive) <= tolerance) | ((i+1) % check_interval == 0) | (i+1 == maximum)
            r = jax.lax.cond(check, lambda _: residual(x), lambda _: recursive, None)
            weighted = precondition(r)
            next_rho = dot(r, weighted)
            restart = norm(axpy(recursive, r, -1)) > .1*norm(r)
            p = axpy(weighted, p, jnp.where(restart, 0, next_rho/rho))
            jax.debug.callback(progress, i+1, norm(r)/initial_norm, ordered=True)
            return i+1, x, r, p, next_rho, norm(r)
        state = (jnp.int64(0), zero, b, z, dot(b, z), norm(b))
        state = jax.lax.while_loop(lambda s: (s[0] < maximum) & (s[-1] > tolerance), step, state)
        return finish(state)
    beta0 = jnp.sqrt(dot(b, z))
    divide = lambda a, t: scale(a, 1/jnp.where(t > 0, t, 1))
    def step(state):
        i, x, z, v, previous, direction, previous_direction, beta, dbar, epsilon, cosine, sine, estimate, threshold, error = state
        next_z = axpy(operator(v), previous, -beta)
        alpha = dot(v, next_z)
        next_z = axpy(next_z, z, -alpha)
        next_v = precondition(next_z)
        next_beta = jnp.sqrt(jnp.maximum(dot(next_z, next_v), 0))
        delta, gbar = cosine*dbar+sine*alpha, sine*dbar-cosine*alpha
        gamma = jnp.hypot(gbar, next_beta)
        next_direction = divide(axpy(axpy(v, previous_direction, -epsilon), direction, -delta), gamma)
        x = axpy(x, next_direction, gbar/gamma*estimate)
        estimate *= next_beta/gamma
        usual = (divide(next_z, next_beta), divide(next_v, next_beta), z, next_direction, direction, next_beta, -cosine*next_beta, sine*next_beta, gbar/gamma, next_beta/gamma, estimate)
        def verify(_):
            r = residual(x)
            weighted = precondition(r)
            actual = jnp.sqrt(jnp.maximum(dot(r, weighted), 0))
            restart = (next_beta == 0) | (actual > 1.1*estimate)
            fresh = (divide(r, actual), divide(weighted, actual), zero, zero, zero, 0., 0., 0., -1., 0., actual)
            values = jax.tree.map(lambda a, b: jnp.where(restart, a, b), fresh, usual)
            error = norm(r)
            return values, values[-1]*tolerance/jnp.maximum(error, tolerance), error
        # Calibrate the metric trigger against the measured physical norm;
        # otherwise a loose preconditioned tolerance triggers a costly check
        # on every remaining iteration before the density tolerance is met.
        check = (estimate <= threshold) | ((i+1) % check_interval == 0) | (next_beta == 0) | (i+1 == maximum)
        values, threshold, error = jax.lax.cond(check, verify, lambda _: (usual, threshold, error), None)
        jax.debug.callback(progress, i+1, values[-1]/beta0, ordered=True)
        return (i+1, x, *values, threshold, error)
    scalar = lambda value: jnp.asarray(value, b[0].dtype)
    state = (jnp.int64(0), zero, divide(b, beta0), divide(z, beta0), zero, zero, zero, scalar(0), scalar(0), scalar(0), scalar(-1), scalar(0), beta0, tolerance*beta0/jnp.where(initial_norm > 0, initial_norm, 1), norm(b))
    state = jax.lax.while_loop(lambda s: (s[0] < maximum) & (s[-1] > tolerance), step, state)
    return finish(state)

