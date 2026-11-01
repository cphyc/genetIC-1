# genetIC in Python and JAX

The Python implementation lives in `genetic/`. The original C++ implementation and
its fixtures remain available as the compatibility reference.

```sh
python -m venv .venv
. .venv/bin/activate
pip install -e '.[test]'
genetic-jax example/paramfile.txt
python -m pytest
```

All numerical arrays and kernels, including the legacy-compatible ranlxs2 and
Gaussian ziggurat RNG, use JAX with 64-bit arithmetic. JAX selects the available
CPU or accelerator; install the appropriate accelerator build of JAX to use a GPU.
NumPy is used only at file input/output boundaries. Neither GSL nor the C++
executable is needed to run the Python implementation. Python manages commands,
static grid sizes, and input validation; numerical buffers remain on the device.

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
legacy seed compatibility; `cli.py`, `mapper.py`, and `io.py` handle parsing and
files. The core stays close to 1,000 physical lines, counting blank lines,
docstrings, geometry, and the RNG adapter. Internal requirements are documented
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

The regression suite overlays the branch's updated single-level splice reference
and adds its three-level splice fixture; all other upstream references and
comparison tolerances are unchanged. Core tests also compare both splice solvers
against a small explicit dense constrained solution. `core.py` and `random.py`
together contain 1,076 physical lines, including lookup tables, blank lines,
docstrings, and geometry.

Validation on the prepared CPU environment: all 91 tests pass (84 regression/mapper
cases and seven core tests), with unchanged comparison tolerances. The full run
took 397.75 seconds, including JAX compilation. GPU execution was not available
for validation in this environment.
