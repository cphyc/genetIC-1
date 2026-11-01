"""File formats and diagnostics; no simulation logic lives here."""
from pathlib import Path
import struct
import numpy as np
import jax
import jax.numpy as jnp
from .core import estimate_power_spectrum


def record(stream, data):
    data = data if isinstance(data, bytes) else np.asarray(data).tobytes()
    marker = struct.pack("<I", len(data))
    stream.write(marker+data+marker)


def write_tipsy(path, particles, cosmo, box):
    p = jax.device_get(particles)
    gas = p["gas"]
    mass_factor = cosmo.Om/np.sum(p["mass"])
    kpc = box*1000/cosmo.hubble
    msol = 1e10/cosmo.hubble/mass_factor
    kms = np.sqrt(4.3022682e-6*msol/kpc)
    with open(path, "wb") as stream:
        stream.write(struct.pack("<d6i", cosmo.a, len(gas), 3, int(gas.sum()), int((~gas).sum()), 0, 0))
        for is_gas, width in [(True, 12), (False, 9)]:
            mask = gas == is_gas
            out = np.zeros((mask.sum(), width), dtype="<f4")
            out[:, 0] = p["mass"][mask]*mass_factor
            out[:, 1:4] = p["pos"][mask]/box-.5
            out[:, 4:7] = p["vel"][mask]/np.sqrt(cosmo.a)/kms
            out[:, 9 if is_gas else 7] = p["eps"][mask]/box
            if is_gas:
                out[:, 8] = cosmo.temperature
            out.tofile(stream)
    Path(path).with_name("tipsy.param").write_text(f"dKpcUnit = {kpc:.6g}\ndMsolUnit = {msol:.6g}\ndHubble0 = {0.1*cosmo.hubble*kpc/kms:.6g}\nbComove = 1\n")


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
        struct.pack_into("<6I", header, 96, *(counts & 0xffffffff))
        struct.pack_into("<i4d", header, 124, files, box, cosmo.Om, cosmo.Ol, cosmo.hubble)
        struct.pack_into("<6I", header, 168, *(counts >> 32))
        struct.pack_into("<i", header, 196, 0)
        filename = f"{path}.gadget{version}"+(f".{i}" if files > 1 else "")
        with open(filename, "wb") as stream:
            record(stream, bytes(header))
            for key, dtype in [("pos", "<f4"), ("vel", "<f4"), ("id", "<u8"), ("mass", "<f4")]:
                record(stream, np.asarray(p[key][index], dtype=dtype))
            if counts[0]:
                record(stream, np.zeros(local[0], dtype="<f4"))


def write_hdf(path, p, cosmo, box, files=1, swift=False):
    import h5py
    p = jax.device_get(p)
    counts = np.bincount(p["ptype"], minlength=6).astype(np.uint64)
    groups = [np.flatnonzero(p["ptype"] == t) for t in range(6)]
    for i in range(files):
        filename = f"{path}"+(f".{i}" if files > 1 else "")+".hdf5"
        pieces = [np.array_split(g, files)[i] for g in groups]
        with h5py.File(filename, "w") as f:
            h = f.create_group("Header")
            for key, value in dict(NumPart_ThisFile=np.array([len(g) for g in pieces], np.uint32), NumPart_Total=counts.astype(np.uint32), NumPart_Total_HighWord=(counts >> 32).astype(np.uint32), MassTable=np.zeros(6), Time=cosmo.a, Redshift=cosmo.z, BoxSize=box, Omega0=cosmo.Om, OmegaLambda=cosmo.Ol, HubbleParam=cosmo.hubble, NumFilesPerSnapshot=files, Flag_DoublePrecision=0).items():
                h.attrs[key] = value
            for t, index in enumerate(pieces):
                if not len(index):
                    continue
                g = f.create_group(f"PartType{t}")
                for key, name in [("pos", "Coordinates"), ("vel", "Velocities"), ("mass", "Masses"), ("id", "ParticleIDs")]:
                    g[name] = p[key][index].astype("<u8" if key == "id" else "<f4")
                if t == 0:
                    g["InternalEnergy"] = np.full(len(index), cosmo.internal_energy)


def power_spectrum(path, density, grid, spectrum):
    np.savetxt(path, np.asarray(estimate_power_spectrum(density, grid, spectrum)), fmt="%.6g")


def write_grafic(path, simulation, center=None, pvar=0):
    from .core import Grid
    s = simulation
    s.combine()
    offset = 0
    for level, grid, points, mask in s.export_grids(center):
        sample_grid = Grid(grid.length, grid.n, points[0, 0, 0]-grid.dx/2, grid.box)
        p = points.reshape(-1, 3)
        displacements = [s.evaluate([v[..., axis] for v in s.offsets], level, sample_grid, p).reshape((grid.n,)*3) for axis in range(3)]
        velocity = [v*s.cosmo.velocity_factor*jnp.sqrt(s.cosmo.a)+s.velocity_offset[a]*jnp.sqrt(s.cosmo.a) for a, v in enumerate(displacements)]
        density = s.evaluate(s.density, level, sample_grid, p).reshape((grid.n,)*3)
        ids = jnp.arange(grid.n**3, dtype=jnp.uint64).reshape((grid.n,)*3)+offset
        if s.grids[level].length == grid.box:
            coords = grid.indices(points.reshape(-1, 3))
            ids = jnp.ravel_multi_index(tuple(coords.T), (grid.n,)*3).reshape((grid.n,)*3).astype(jnp.uint64)+offset
        arrays = velocity+displacements+[density, mask, mask*pvar, ids]
        names = ['ic_velcx', 'ic_velcy', 'ic_velcz', 'ic_poscx', 'ic_poscy', 'ic_poscz', 'ic_deltab', 'ic_refmap', 'ic_pvar_00001', 'ic_particle_ids']
        directory = Path(f"{path}.grafic_{round(grid.box/grid.dx)}")
        directory.mkdir(parents=True, exist_ok=True)
        c = s.cosmo
        header = struct.pack('<3i8f', *([grid.n]*3), grid.dx/c.hubble, *(grid.origin/c.hubble), c.a, c.Om, c.Ol, c.hubble*100)
        for name, values in zip(names, arrays):
            with open(directory/name, 'wb') as stream:
                record(stream, header)
                for z in range(grid.n):
                    record(stream, np.asarray(values[:, :, z].T, dtype='<u8' if name == 'ic_particle_ids' else '<f4'))
        offset += grid.n**3


def write_grid(directory, prefix, level, values, grid):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    np.save(directory/f"{prefix}-{level}.npy", np.asarray(values))
    suffix = '-fourier' if np.iscomplexobj(values) else ''
    description = ' '.join(f'{v:.6g}' for v in [*grid.origin, grid.length])
    (directory/f'{prefix}-info-{level}{suffix}.txt').write_text(
        description+f'\nThe line above contains information about grid level {level}\n'
        'It gives the x-offset, y-offset and z-offset of the low-left corner and also the box length\n')
