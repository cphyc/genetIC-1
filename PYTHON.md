# genetIC in Python and JAX

The Python implementation lives in `genetic/`. The original C++ implementation and
its fixtures remain available as the compatibility reference.
Install a C++ compiler and the GSL development package first (for example,
`libgsl-dev` on Debian/Ubuntu). The `scikit-build-core` build compiles a small RNG extension with CMake.

```sh
python -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
genetic-jax example/paramfile.txt
python -m pytest
```

Simulation arrays and kernels use JAX with 64-bit arithmetic. JAX selects the available
CPU or accelerator; install the appropriate accelerator build of JAX to use a GPU.
NumPy is used at file input/output boundaries. The C++ genetIC executable is not
needed. Python manages commands,
static grid sizes, and input validation; numerical buffers remain on JAX devices.
JAX's persistent compilation cache is enabled at `~/.cache/genetic-jax/jax` by
default (or `$XDG_CACHE_HOME/genetic-jax/jax`). Set
`JAX_COMPILATION_CACHE_DIR` before starting the process to choose another path;
set `JAX_ENABLE_COMPILATION_CACHE=false` to disable it. The default minimum
compile-time threshold is zero so short JIT kernels are cached too; set
`JAX_PERSISTENT_CACHE_MIN_COMPILE_TIME_SECS` to change it.
The RNG uses genetIC's GSL ranlxs2 and Gaussian ziggurat generator directly through
`_random.cpp`; `random.py` only transfers its buffers to JAX and applies FFTs.
Independent shell streams use OpenMP on Linux. Seeds, draw order and continuation
are preserved, without duplicating GSL's lookup tables in Python.

`genetic-jax` accepts the existing ordered parameter files, including deprecated
command aliases. Paths are relative to the parameter file. Output writers cover
Tipsy, Gadget 2/3, Gadget HDF5, Swift, Grafic, NumPy grids, and power spectra.
The separate geometry tool avoids generating random fields:

```sh
genetic-jax-mapper source.txt target.txt input-ids.txt output-ids.txt
genetic-jax-mapper parameters.txt flagged-ids.txt
```

For direct use:

```python
import jax.numpy as jnp
from genetic import Cosmology, PowerSpectrum, Simulation

cosmo = Cosmology(Om=0.3, Ol=0.7, z=99)
sim = Simulation(cosmo, PowerSpectrum(cosmo, amplitude=1e-4))
sim.base_grid(50, 32)
sim.seed, sim.seed_mode = 42, "parallel"
sim.center = jnp.full(3, 25.)
sim.select("sphere", 5)
sim.modify("overdensity", "absolute", 0.1)
sim.apply_modifications()
particles = sim.particles()
```

`core.py` contains the numerical and geometric operations. `random.py` isolates
legacy seed compatibility; `splice.py` contains splice operators and solvers;
`cli.py`, `mapper.py`, and `io.py` handle parsing and files. The core stays below
1,000 physical lines, including blank lines and docstrings. Internal requirements are documented
in the core module: cubic aligned grids and integral resolution ratios.

`tests/test_regression.py` copies the tracked upstream fixtures into a temporary
directory, runs the Python implementation, and uses the unchanged upstream
comparators and tolerances. It includes the mapper fixtures and expected errors;
disabled upstream tests remain disabled. `tests/test_core.py` additionally checks
interpolation adjoints, the Poisson solution, simultaneous constraints, and splice
semantics. Existing baryon-transfer-function rejection is preserved, as in C++.

Splicing follows `splicing-multilevel` (commit `715d758`): a single constrained
solve across the hierarchy uses production density synthesis and its exact JAX
transpose. It preserves zero-power modes and checks the residual in physical
density units. `splice SEED [ACCURACY]` defaults to `1e-6`; `splice_solver cg`
(the default) and `splice_solver minres` are supported. Selected owned cells must
stay at least four enclosing-grid cells from zoom edges, as in that branch.
Splice dual vectors contain only selected cells. Operators use real FFTs,
separable cubic interpolation and its exact transpose, retaining intermediate
corrections in Fourier space. MINRES verifies the physical density residual
periodically and before accepting a solution. At tolerances of `1e-5` or looser,
the iterative FFTs use single precision; Krylov vectors, residual replacement,
and final candidate verification remain double precision. Stricter tolerances
use double precision throughout. The approximate inverse-covariance
preconditioner uses padded selection boxes. Both solvers print iteration progress
and the final verified relative residual.

The regression suite overlays the branch's updated single-level splice reference
and adds its three-level splice fixture; all other upstream references and
comparison tolerances are unchanged. Core tests also compare both splice solvers
against a small explicit dense constrained solution. `tests/test_splice.py`
checks the optimized operators against production synthesis, interpolation
adjoints, particle counts, and finite-step Krylov convergence.

Tipsy output streams particle batches in mapper order. Regular cell-centre samples
use direct lookup; other samples retain cubic interpolation. This bounds GPU
memory without changing particle ordering, units or output records.

Measure the complete CLI on two 256³ grids, including initial drawing, splicing,
particle generation, the Tipsy snapshot and both grid dumps. One complete warmup
run is discarded for JIT compilation; subsequent runs include all other work.
Outputs are written into an isolated temporary directory:

```sh
uv run python tools/benchmark_splice.py --n 256 --repeat 2
```

On a Quadro T1000 Max-Q (4 GB), the complete CLI took 15.26 s and 15.28 s
after a 28.54 s warmup, writing 892 MB per run with the default JAX allocator.
Validation: 128 CPU tests and 44 focused GPU tests pass, including the unchanged
upstream regression tolerances, native RNG streams and streamed Tipsy records.
