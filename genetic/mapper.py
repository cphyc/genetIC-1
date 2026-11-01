"""Map particle IDs between parameter-file geometries without generating fields."""
import sys
import numpy as np
import jax.numpy as jnp
from .cli import Interpreter


def main():
    args = sys.argv[1:]
    if len(args) not in (2, 4):
        print('Usage: genetic-jax-mapper PARAMFILE OUTPUT | SOURCE TARGET INPUT OUTPUT', file=sys.stderr)
        return 1
    source = Interpreter(geometry_only=True)
    source.run(args[0])
    if len(args) == 2:
        ids = source.flagged_ids()
    else:
        target = Interpreter(geometry_only=True)
        target.run(args[1])
        selected = jnp.atleast_1d(jnp.asarray(np.loadtxt(args[2], dtype="int64")))
        target.transfer_ids(source, selected)
        ids = target.flagged_ids()
    np.savetxt(args[-1], ids, fmt='%d')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
