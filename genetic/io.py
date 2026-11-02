"""File formats and diagnostics; no simulation logic lives here."""

import struct
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from .core import estimate_power_spectrum


def record(stream, data):
    data = data if isinstance(data, bytes) else np.asarray(data).tobytes()
    marker = struct.pack("<I", len(data))
    stream.write(marker + data + marker)


def tipsy_header(stream, path, cosmo, box, total_mass, ngas, ndm):
    factor = cosmo.Om / total_mass
    kpc = box * 1000 / cosmo.hubble
    msol = 1e10 / cosmo.hubble / factor
    kms = np.sqrt(4.3022682e-6 * msol / kpc)
    stream.write(struct.pack("<d6i", cosmo.a, ngas + ndm, 3, ngas, ndm, 0, 0))
    Path(path).with_name("tipsy.param").write_text(
        f"dKpcUnit = {kpc:.6g}\ndMsolUnit = {msol:.6g}\ndHubble0 = {0.1 * cosmo.hubble * kpc / kms:.6g}\nbComove = 1\n"
    )
    return factor, kms


def tipsy_records(p, gas, cosmo, box, factor, kms):
    out = np.zeros((len(p["mass"]), 12 if gas else 9), dtype="<f4")
    out[:, 0] = p["mass"] * factor
    out[:, 1:4] = p["pos"] / box - 0.5
    out[:, 4:7] = p["vel"] / np.sqrt(cosmo.a) / kms
    out[:, 9 if gas else 7] = p["eps"] / box
    if gas:
        out[:, 8] = cosmo.temperature
    return out


def write_tipsy(path, particles, cosmo, box):
    p = jax.device_get(particles)
    gas = p["gas"]
    with open(path, "wb") as stream:
        factor, kms = tipsy_header(
            stream,
            path,
            cosmo,
            box,
            np.sum(p["mass"]),
            int(gas.sum()),
            int((~gas).sum()),
        )
        for is_gas in (True, False):
            selected = {k: p[k][gas == is_gas] for k in ("mass", "pos", "vel", "eps")}
            tipsy_records(selected, is_gas, cosmo, box, factor, kms).tofile(stream)


def particle_points(block, start, stop):
    """Host I/O batches in the mapper's parent-then-child ordering."""
    gas, level, grid, kind, parent, parents, mask = block
    index = np.arange(start, stop)
    if parent is None:
        points = np.stack(np.unravel_index(index, (grid.n,) * 3), -1)
        points = (points + 0.5) * grid.dx + np.asarray(grid.origin)
    else:
        ratio = round(parent.dx / grid.dx)
        coarse = np.stack(
            np.unravel_index(parents[index // ratio**3], (parent.n,) * 3), -1
        )
        fine = np.stack(np.unravel_index(index % ratio**3, (ratio,) * 3), -1)
        points = (
            coarse * parent.dx + np.asarray(parent.origin) + (fine + 0.5) * grid.dx
        ) % grid.box
    if mask is not None:
        cells = (
            np.floor(
                (points - np.asarray(grid.origin)) % grid.box / grid.dx + 1e-10
            ).astype(int)
            % grid.n
        )
        points = points[~mask[tuple(cells.T)]]
    return points


def write_tipsy_stream(path, sim, center_output=False, chunk_size=262144):
    """Bound GPU and host working memory while writing the complete snapshot."""
    sim.combine()
    layouts, total_mass, counts = [], 0.0, [0, 0]
    for gas in (True, False):
        if gas and sim.cosmo.Ob == 0:
            continue
        hierarchy, masks = sim.particle_hierarchy(gas)
        masks = jax.device_get(masks)
        for i, (level, grid, kind) in enumerate(hierarchy):
            if gas and not (sim.baryons_all_levels or level == len(sim.grids) - 1):
                continue
            parent = hierarchy[i - 1][1] if i else None
            parents = np.flatnonzero(masks[i - 1]) if i else None
            mask = masks[i] if i < len(masks) else None
            block = (gas, level, grid, kind, parent, parents, mask)
            raw = len(parents) * round(parent.dx / grid.dx) ** 3 if i else grid.n**3
            count = (
                raw
                if mask is None
                else (
                    raw - int(mask.sum())
                    if i == 0
                    else sum(
                        len(particle_points(block, a, min(a + chunk_size, raw)))
                        for a in range(0, raw, chunk_size)
                    )
                )
            )
            split = sim.baryons_all_levels or level == len(sim.grids) - 1
            fraction = (
                sim.cosmo.Ob / sim.cosmo.Om
                if gas
                else (1 - sim.cosmo.Ob / sim.cosmo.Om if split else 1)
            )
            total_mass += count * 27.744948 * sim.cosmo.Om * grid.dx**3 * fraction
            counts[0 if gas else 1] += count
            layouts.append((block, raw))
    box = sim.grids[0].box
    with open(path, "wb") as stream:
        factor, kms = tipsy_header(stream, path, sim.cosmo, box, total_mass, *counts)
        for block, raw in layouts:
            gas, level, grid, kind, *_ = block
            for start in range(0, raw, chunk_size):
                points = particle_points(block, start, min(start + chunk_size, raw))
                if not len(points):
                    continue
                data = sim.particle_block(level, grid, points, kind, gas)
                p = jax.device_get({k: data[k] for k in ("mass", "pos", "vel", "eps")})
                if center_output:
                    p["pos"] = (p["pos"] + box / 2 - np.asarray(sim.center)) % box
                tipsy_records(p, gas, sim.cosmo, box, factor, kms).tofile(stream)


def write_gadget(path, p, cosmo, box, files=1, version=3):
    p = jax.device_get(p)
    counts = np.bincount(p["ptype"], minlength=6).astype(np.uint64)
    # Use explicit masses throughout; both Gadget variants support this.
    groups = [np.flatnonzero(p["ptype"] == t) for t in range(6)]
    pieces = [[np.array_split(g, files)[i] for g in groups] for i in range(files)]
    for i, piece in enumerate(pieces):
        index = np.concatenate(piece)
        local = np.array([len(g) for g in piece], dtype=np.uint32)
        header = bytearray(256)
        struct.pack_into("<6I", header, 0, *local)
        struct.pack_into("<2d", header, 72, cosmo.a, cosmo.z)
        struct.pack_into("<6I", header, 96, *(counts & 0xFFFFFFFF))
        struct.pack_into(
            "<i4d", header, 124, files, box, cosmo.Om, cosmo.Ol, cosmo.hubble
        )
        struct.pack_into("<6I", header, 168, *(counts >> 32))
        struct.pack_into("<i", header, 196, 0)
        filename = f"{path}.gadget{version}" + (f".{i}" if files > 1 else "")
        with open(filename, "wb") as stream:
            record(stream, bytes(header))
            for key, dtype in [
                ("pos", "<f4"),
                ("vel", "<f4"),
                ("id", "<u8"),
                ("mass", "<f4"),
            ]:
                record(stream, np.asarray(p[key][index], dtype=dtype))
            if counts[0]:
                record(stream, np.zeros(local[0], dtype="<f4"))


def write_hdf(path, p, cosmo, box, files=1, swift=False):
    import h5py

    p = jax.device_get(p)
    counts = np.bincount(p["ptype"], minlength=6).astype(np.uint64)
    groups = [np.flatnonzero(p["ptype"] == t) for t in range(6)]
    for i in range(files):
        filename = f"{path}" + (f".{i}" if files > 1 else "") + ".hdf5"
        pieces = [np.array_split(g, files)[i] for g in groups]
        with h5py.File(filename, "w") as f:
            h = f.create_group("Header")
            for key, value in dict(
                NumPart_ThisFile=np.array([len(g) for g in pieces], np.uint32),
                NumPart_Total=counts.astype(np.uint32),
                NumPart_Total_HighWord=(counts >> 32).astype(np.uint32),
                MassTable=np.zeros(6),
                Time=cosmo.a,
                Redshift=cosmo.z,
                BoxSize=box,
                Omega0=cosmo.Om,
                OmegaLambda=cosmo.Ol,
                HubbleParam=cosmo.hubble,
                NumFilesPerSnapshot=files,
                Flag_DoublePrecision=0,
            ).items():
                h.attrs[key] = value
            for t, index in enumerate(pieces):
                if not len(index):
                    continue
                g = f.create_group(f"PartType{t}")
                for key, name in [
                    ("pos", "Coordinates"),
                    ("vel", "Velocities"),
                    ("mass", "Masses"),
                    ("id", "ParticleIDs"),
                ]:
                    g[name] = p[key][index].astype("<u8" if key == "id" else "<f4")
                if t == 0:
                    g["InternalEnergy"] = np.full(len(index), cosmo.internal_energy)


def power_spectrum(path, density, grid, spectrum):
    np.savetxt(
        path, np.asarray(estimate_power_spectrum(density, grid, spectrum)), fmt="%.6g"
    )


def write_grafic(path, simulation, center=None, pvar=0):
    from .core import Grid

    s = simulation
    s.combine()
    offset = 0
    for level, grid, points, mask in s.export_grids(center):
        sample_grid = Grid(grid.length, grid.n, points[0, 0, 0] - grid.dx / 2, grid.box)
        p = points.reshape(-1, 3)
        displacements = [
            s.evaluate(
                [v[..., axis] for v in s.offsets], level, sample_grid, p
            ).reshape((grid.n,) * 3)
            for axis in range(3)
        ]
        velocity = [
            v * s.cosmo.velocity_factor * jnp.sqrt(s.cosmo.a)
            + s.velocity_offset[a] * jnp.sqrt(s.cosmo.a)
            for a, v in enumerate(displacements)
        ]
        density = s.evaluate(s.density, level, sample_grid, p).reshape((grid.n,) * 3)
        ids = jnp.arange(grid.n**3, dtype=jnp.uint64).reshape((grid.n,) * 3) + offset
        if s.grids[level].length == grid.box:
            coords = grid.indices(points.reshape(-1, 3))
            ids = (
                jnp.ravel_multi_index(tuple(coords.T), (grid.n,) * 3)
                .reshape((grid.n,) * 3)
                .astype(jnp.uint64)
                + offset
            )
        arrays = velocity + displacements + [density, mask, mask * pvar, ids]
        names = [
            "ic_velcx",
            "ic_velcy",
            "ic_velcz",
            "ic_poscx",
            "ic_poscy",
            "ic_poscz",
            "ic_deltab",
            "ic_refmap",
            "ic_pvar_00001",
            "ic_particle_ids",
        ]
        directory = Path(f"{path}.grafic_{round(grid.box / grid.dx)}")
        directory.mkdir(parents=True, exist_ok=True)
        c = s.cosmo
        header = struct.pack(
            "<3i8f",
            *([grid.n] * 3),
            grid.dx / c.hubble,
            *(grid.origin / c.hubble),
            c.a,
            c.Om,
            c.Ol,
            c.hubble * 100,
        )
        for name, values in zip(names, arrays):
            with open(directory / name, "wb") as stream:
                record(stream, header)
                for z in range(grid.n):
                    record(
                        stream,
                        np.asarray(
                            values[:, :, z].T,
                            dtype="<u8" if name == "ic_particle_ids" else "<f4",
                        ),
                    )
        offset += grid.n**3


def write_grid(directory, prefix, level, values, grid):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory / f"{prefix}-{level}.npy", np.asarray(values))
    suffix = "-fourier" if np.iscomplexobj(values) else ""
    description = " ".join(f"{v:.6g}" for v in [*grid.origin, grid.length])
    (directory / f"{prefix}-info-{level}{suffix}.txt").write_text(
        description
        + f"\nThe line above contains information about grid level {level}\n"
        "It gives the x-offset, y-offset and z-offset of the low-left corner and also the box length\n"
    )
