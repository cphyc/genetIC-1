"""Case insensitive, ordered parameter-file interpreter compatible with genetIC."""

import argparse
import os
import shlex
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from . import io
from .core import Geometry, PowerSpectrum, Simulation, fft, ifft

ALIASES = dict(
    basegrid="base_grid",
    zoomgrid="zoom_grid",
    zoomgrid_with_origin_at="zoom_grid_with_origin_at",
    seed="random_seed_real_space",
    seedfourier="random_seed_serial",
    seedfourier_parallel="random_seed",
    zerolevel="zero_level",
    idfile="id_file",
    append_idfile="merge_id_file",
    append_idfile_="merge_id_file",
    dump_idfile="dump_id_file",
    centre="center",
    centre_on="center_on",
    zin="z",
    center_grafic_output="center_output",
    centre_output="center_output",
    exact_power_spectrum_enforcement="fix_power",
    num_neighbors="num_neighbours",
)


class Interpreter(Geometry):
    def __init__(self, simulation=None, geometry_only=False):
        self.sim = simulation or Simulation()
        self.geometry_only = geometry_only
        self.log_prefix = ""
        self.outdir, self.outname, self.format = Path("."), "IC", None
        self.files, self.center_output, self.pvar = 1, False, 0
        self.input_mapper = None

    def run(self, filename):
        filename = Path(filename).resolve()
        previous = Path.cwd()
        try:
            os.chdir(filename.parent)
            self.log()
            for number, line in enumerate(filename.read_text().splitlines(), 1):
                tokens = shlex.split(line, comments=True)
                if tokens:
                    try:
                        self.command(tokens[0].lower(), tokens[1:])
                    except (ValueError, IndexError, OSError) as error:
                        raise ValueError(
                            f'Error "{error}" on line {number} ("{line.split(chr(35))[0].strip()}")'
                        ) from error
        finally:
            os.chdir(previous)
        return self.sim

    def log(self, message=""):
        print(self.log_prefix + message)

    def log_region(self, name):
        s = self.sim
        grid = s.grids[-1]
        if len(s.grids) > 1 and jnp.all(grid.flags):
            self.log(
                "WARNING: Region selected for modification is the entire zoom grid. This is likely because the cell selection extends beyond the zoom boundaries."
            )
            self.log(
                "By design, modifications are meant to be defined inside a zoom region. Increase the size of your zoom grid or decrease your selection to avoid nasty surprises."
            )
        if name.lower() == "variance":
            positions = jnp.argwhere(grid.flags)
            spans = []
            for axis in range(3):
                v = jnp.unique(positions[:, axis])
                spans.append(grid.n - jnp.max(jnp.diff(jnp.r_[v, v[0] + grid.n])) + 1)
            size = max(spans) * grid.dx
            if s.filter_scale > size:
                self.log(
                    f"WARNING: High-pass filtering scale: {s.filter_scale:.6g} h**-1 Mpc is greater than the rough window size: {size:.6g} h**-1 Mpc used for modifications."
                )
                self.log(
                    "This is prone to numerical errors when modifying the field. Decrease filtering scale to avoid it."
                )

    def apply_modifications(self):
        before = self.sim.chi2()
        self.sim.apply_modifications()
        self.log(f"  Modification Delta chi^2 = {self.sim.chi2() - before:.6g}")
        self.log(f"           d.o.f. in field = {self.dof()}")

    @property
    def path(self):
        self.outdir.mkdir(parents=True, exist_ok=True)
        return self.outdir / self.outname

    def command(self, name, args):
        name = ALIASES.get(name, name)
        s = self.sim
        if self.geometry_only and (
            name.startswith("dump_")
            or name
            in {
                "calculate",
                "modify",
                "done",
                "chi2",
                "apply_modifications",
                "splice",
                "import_level",
                "reverse",
                "reverse_small_k",
                "zero_level",
                "zero_level_field",
                "camb",
            }
        ):
            return
        f = lambda i=0: float(args[i])
        n = lambda i=0: int(args[i])
        if name in ("om", "ol", "ob", "tcmb", "hubble", "s8", "ns", "z"):
            setattr(
                s.cosmo,
                {"om": "Om", "ol": "Ol", "ob": "Ob", "tcmb": "TCMB"}.get(name, name),
                f(),
            )
        elif name == "camb":
            s.spectrum = PowerSpectrum(s.cosmo, np.loadtxt(args[0]))
        elif name == "powerlaw_amplitude":
            s.spectrum = PowerSpectrum(s.cosmo, amplitude=f())
        elif name in (
            "random_seed",
            "random_seed_real_space",
            "random_seed_serial",
            "seedfourier_reverse",
        ):
            if s.seed is not None:
                raise ValueError("The random number generator has already been seeded")
            s.seed = n()
            s.seed_mode = {
                "random_seed": "parallel",
                "random_seed_real_space": "real",
                "random_seed_serial": "serial",
                "seedfourier_reverse": "reverse",
            }[name]
        elif name == "base_grid":
            s.base_grid(f(), n(1))
            self.log("Initialized the base grid:")
            self.log(f"  Box length            = {f():.6g} Mpc/h")
            self.log(f"  n                     = {n(1)}")
            self.log(f"  dx                    = {s.grids[-1].dx:.6g}")
        elif name in ("zoom_grid", "zoom_grid_with_origin_at"):
            if name == "zoom_grid":
                s.zoom_grid(n(), n(1))
            else:
                s.zoom_grid(n(3), n(4), [n(i) for i in range(3)])
            child, parent = s.grids[-1], s.grids[-2]
            origin = jnp.rint((child.origin - parent.origin) / parent.dx).astype(int)
            selected = jnp.argwhere(s.replaced[-1])
            relative = (selected - origin) % round(parent.box / parent.dx)
            width = parent.n // (n() if name == "zoom_grid" else n(3))
            if jnp.any(relative < 3) or jnp.any(relative >= width - 3):
                self.log(
                    "WARNING: Opening a zoom where flagged particles are within 3 pixels of the edge. This is prone to numerical errors."
                )
            self.log("Initialized a zoom region:")
            self.log(f"  Subbox length         = {child.length:.6g} Mpc/h")
            self.log(f"  n                     = {child.n}")
            self.log(f"  dx                    = {child.dx:.6g}")
            self.log(
                f"  Zoom factor           = {n() if name == 'zoom_grid' else n(3)}"
            )
            self.log(f"  Num particles         = {jnp.sum(s.replaced[-1])}")
            self.log(
                f"  Low-left corner in parent grid = ({origin[0]}, {origin[1]}, {origin[2]})"
            )
            self.log(
                "  Low-left corner (h**-1 Mpc)    = "
                + ", ".join(f"{v:.6g}" for v in child.origin)
            )
            self.log(f"  Total particles       = {s.particle_count()}")
        elif name == "center":
            s.center = jnp.array([f(i) for i in range(3)])
        elif name == "center_on":
            s.center = s.grids[0].points.reshape(-1, 3)[n()]
        elif name.startswith("select_"):
            shape = name[7:]
            size = (
                [f(i) for i in range(3)] if shape == "ellipse" else (f() if args else 0)
            )
            s.select(shape, size)
        elif name == "expand_flagged_region":
            for g in s.grids:
                g.expand(n())
        elif name in ("id_file", "merge_id_file"):
            self.load_ids(args[0], name == "merge_id_file")
        elif name == "dump_id_file":
            np.savetxt(args[0], self.flagged_ids(), fmt="%d")
        elif name == "mapper_relative_to":
            self.input_mapper = Interpreter(geometry_only=True)
            self.log(f"+ Input mapper: computing geometry from {args[0]}")
            self.input_mapper.log_prefix = self.log_prefix + "| "
            self.input_mapper.log()
            self.input_mapper.run(args[0])
        elif name == "fix_power":
            s.fixed = True
        elif name == "strays_on":
            s.strays = True
        elif name in ("supersample", "supersample_gas", "subsample"):
            if n() < 1:
                raise ValueError("Sampling factor must be positive")
            setattr(s, name, n())
            if name == "subsample" and len(s.grids) > 1 and not s.strays:
                parent, child = s.grids[:2]
                selected = parent.points[s.replaced[0]]
                width = parent.dx * n()
                corners = (
                    jnp.floor((selected - parent.origin) / width) * width
                    + parent.origin
                )
                extreme = jnp.vstack([corners + 1e-8, corners + width - 1e-8])
                if not jnp.all(child.contains(extreme)):
                    raise ValueError(
                        "Following sub/super-sampling, the zoom particles no longer fit fully on their grids. Try rerunning with strays_on, or (better) increase the size of the zoom region."
                    )
        elif name == "adapt_mask":
            s.adapt_mask(n())
        elif name == "autopad":
            s.autopad = n()
        elif name == "eps_norm":
            s.eps_norm = f()
        elif name == "num_neighbours":
            s.neighbours = n()
        elif name == "filtering_scale":
            s.filter_scale = f()
        elif name == "velocity_offset":
            s.velocity_offset = jnp.array([f(i) for i in range(3)]) / jnp.sqrt(
                s.cosmo.a
            )
        elif name == "outdir":
            self.outdir = Path(args[0])
        elif name == "outname":
            self.outname = args[0]
        elif name == "outformat":
            self.format = {
                "2": "gadget2",
                "3": "gadget3",
                "4": "tipsy",
                "5": "grafic",
                "6": "gadgethdf",
                "7": "swift",
            }.get(args[0], args[0].lower())
        elif name == "gadget_num_files":
            self.files = n()
        elif name == "gadget_particle_type":
            if not s.grids or not 0 <= n() < 6:
                raise ValueError(
                    "Gadget particle type must be between 0 and 5 and a grid must exist"
                )
            if n() in s.types[:-1] and n() != s.types[-1]:
                raise ValueError(
                    "Cannot reuse a gadget particle type; this type was already used in a different section of the ICs"
                )
            s.types[-1] = s.current_type = n()
        elif name == "gadget_flagged_particle_type":
            s.flagged_type = n()
        elif name == "baryons_all_levels":
            s.baryons_all_levels = True
        elif name == "baryon_tf_on":
            raise ValueError(
                "Use of the baryon transfer function currently has a bug which makes it highly inaccurate"
            )
        elif name == "center_output":
            self.center_output = True
        elif name == "pvar":
            self.pvar = f()
        elif name == "zero_level" or name == "zero_level_field":
            if len(args) > 1 and n(1) != 0:
                raise ValueError("Requested field does not exist")
            s.draw()
            s.noise[n()] = jnp.zeros_like(s.noise[n()])
        elif name == "import_level":
            s.draw()
            g = s.grids[n()]
            cov = jnp.sqrt(s.spectrum.covariance(g, "dm"))
            s.noise[n()] = jnp.where(
                cov > 0,
                fft(jnp.asarray(np.load(args[1]))) / jnp.where(cov > 0, cov, 1),
                0,
            )
        elif name in ("reverse", "reverse_small_k"):
            s.draw()
            s.noise = [
                -v if name == "reverse" else jnp.where((g.k < f()) & (g.k > 0), -v, v)
                for v, g in zip(s.noise, s.grids)
            ]
        elif name == "calculate":
            self.log_region(args[0])
            self.log(f"{args[0]}: calculated value = {float(s.calculate(args[0])):.6g}")
        elif name == "modify":
            self.log_region(args[0])
            s.modify(args[0], args[1], f(2))
        elif name == "apply_modifications":
            self.apply_modifications()
        elif name == "clear_modifications":
            s.modifications.clear()
            self.log("Clearing modification list")
        elif name == "chi2":
            if s.combined:
                raise ValueError(
                    "Cannot calculate the chi^2 after the output field has been generated"
                )
            print(f"Calculated chi^2 = {float(s.chi2()):.10g} (dof = {self.dof()})")
        elif name == "splice_solver":
            if args[0].lower() not in ("cg", "minres"):
                raise ValueError("Splice solver must be cg or minres")
            s.splice_solver = args[0].lower()
        elif name == "splice":
            s.splice(n(), f(1) if len(args) > 1 else 1e-6)
        elif name == "done":
            if s.modifications:
                self.apply_modifications()
            self.write()
        elif name.startswith("dump_grid") or name == "dump_vx":
            s.draw()
            level = n()
            if s.combined:
                value = s.density[level]
            else:
                value = ifft(
                    s.noise[level] * jnp.sqrt(s.spectrum.covariance(s.grids[level]))
                )
            if "fourier" in name:
                value = fft(value)
            if name == "dump_vx":
                s.combine()
                value = s.offsets[level][..., 0]
            io.write_grid(
                self.outdir,
                "vx" if name == "dump_vx" else "grid",
                level,
                value,
                s.grids[level],
            )
        elif name in ("dump_ps", "dump_ps_field"):
            s.draw()
            level = n()
            g = s.grids[level]
            value = (
                s.density[level]
                if s.combined
                else ifft(s.noise[level] * jnp.sqrt(s.spectrum.covariance(g)))
            )
            suffix = (
                "_baryons"
                if len(args) > 1 and args[1].lower() in ("1", "baryon", "gas")
                else ""
            )
            io.power_spectrum(f"{self.path}_{level}{suffix}.ps", value, g, s.spectrum)
        elif name == "dump_mask":
            for level, (_, grid, _, mask) in enumerate(s.export_grids(s.center)):
                io.write_grid(self.outdir, "mask", level, mask.astype(float), grid)
        elif name in ("dump_tipsy", "dump_tipsy_field"):
            if len(args) > 1 and n(1) != 0:
                raise ValueError("Requested field does not exist")
            p = s.particles(self.format == "tipsy")
            with open(args[0], "wb") as stream:
                np.array([len(p["mass"])], "<i4").tofile(stream)
                np.asarray(p["density"], dtype="<f4").tofile(stream)
        else:
            raise ValueError(f"Unknown command {name}")

    def dof(self):
        s = self.sim
        return sum(
            g.n**3 - (g.n**3 // round(s.grids[i - 1].dx / g.dx) ** 3 if i else 0)
            for i, g in enumerate(s.grids)
        )

    def load_ids(self, path, merge=False):
        source = self.input_mapper or self
        ids = jnp.unique(jnp.atleast_1d(jnp.asarray(np.loadtxt(path, dtype="int64"))))
        selected = self.transfer_ids(source, ids, merge)
        grid = self.sim.grids[0]
        self.sim.center = (
            selected[0] + jnp.mean(grid.delta(selected, selected[0]), axis=0)
        ) % grid.box

    def write(self):
        s = self.sim
        if self.format == "grafic":
            io.write_grafic(
                self.path, s, s.center if self.center_output else None, self.pvar
            )
            return
        if self.format is None:
            raise ValueError("No output format specified!")
        if self.format not in ("tipsy", "gadget2", "gadget3", "gadgethdf", "swift"):
            raise ValueError(f"Unsupported output format {self.format}")
        if self.format == "tipsy" and s.flagged_type is None:
            io.write_tipsy_stream(str(self.path) + ".tipsy", s, self.center_output)
            return
        p = s.particles(self.format == "tipsy")
        box = s.grids[0].box
        if self.center_output:
            p["pos"] = (p["pos"] + box / 2 - s.center) % box
        if self.format == "tipsy":
            io.write_tipsy(str(self.path) + ".tipsy", p, s.cosmo, box)
        elif self.format.startswith("gadget") and self.format != "gadgethdf":
            io.write_gadget(
                self.path, p, s.cosmo, box, self.files, int(self.format[-1])
            )
        else:
            io.write_hdf(self.path, p, s.cosmo, box, self.files, self.format == "swift")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paramfile")
    args = parser.parse_args()
    try:
        Interpreter().run(args.paramfile)
    except (ValueError, OSError) as error:
        print(error, file=sys.stderr)
        return 1
    return 0
