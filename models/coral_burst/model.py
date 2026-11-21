"""Coral Burst: radial branching wall relief, sized for a Creality K1C (220x220 bed).

Procedural tree grown outward in angular wedges (no crossings), each branch a
round half-tube sitting flat on the bed, smooth-blended at every fork.
Run: python model.py  ->  coral_burst.stl / coral_burst.3mf
"""
import math, random
import numpy as np
from skimage import measure
import trimesh

SEED = 7
DIAMETER = 200.0       # mm overall
R_MAX = DIAMETER / 2 - 3
R_TRUNK = 2.6          # mm half-width at hub
R_TIP = 1.2           # mm half-width at tips (2.3 mm wide lines)
SPACING = 5.2          # mm arc spacing before a branch forks
STEP = 1.0             # mm growth step
RES = 0.25             # voxel size mm
BLEND = 0.9            # smooth-union radius mm
N_TRUNKS = 18

rng = random.Random(SEED)
segs = []  # (x0,y0,x1,y1,r0,r1)

def radius_at(r):
    t = min(1.0, r / R_MAX)
    return R_TRUNK + (R_TIP - R_TRUNK) * t ** 0.6

def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi

def twig(x, y, h, r):
    """Short tapered Y-fork at a branch end, like the reference tips."""
    for s in (-1, 1):
        hh = h + s * rng.uniform(0.32, 0.5)
        px, py = x, y
        n = rng.randint(4, 7)
        for k in range(n):
            hh += rng.gauss(0, 0.08)
            qx, qy = px + STEP * math.cos(hh), py + STEP * math.sin(hh)
            rr = R_TIP * (1 - 0.25 * (k + 1) / n)
            segs.append((px, py, qx, qy, R_TIP if k == 0 else rr * 1.08, rr))
            px, py = qx, qy

def grow(x, y, h, lo, hi, r_end, since=0.0):
    """Grow a wandering branch outward, kept inside the polar wedge [lo, hi]."""
    c = (lo + hi) / 2
    phase = rng.uniform(0, 2 * math.pi)
    while True:
        r = math.hypot(x, y)
        if r >= r_end:
            if rng.random() < 0.85:
                twig(x, y, h, r)
            return
        phi = c + wrap(math.atan2(y, x) - c)
        half = (hi - lo) / 2
        # steer: mostly radial, pushed back toward wedge centre near its edges
        lateral = r * (c - phi)                      # mm off the wedge centreline
        desired = phi + 0.9 * math.atan(lateral / 7.0) + 0.22 * math.sin(r / 6.5 + phase)
        h += 0.30 * wrap(desired - h) + rng.gauss(0, 0.05)
        dev = wrap(h - phi)                          # never turn back on itself
        if abs(dev) > 0.55:
            h = phi + math.copysign(0.55, dev)
        if (hi - lo) * r > 2 * SPACING and since > 5.0 and r > 9:
            split = c + rng.uniform(-0.12, 0.12) * (hi - lo)
            da = rng.uniform(0.22, 0.38)
            grow(x, y, h - da, lo, split, r_end + rng.uniform(-5, 3))
            grow(x, y, h + da, split, hi, r_end + rng.uniform(-5, 3))
            return
        nx, ny = x + STEP * math.cos(h), y + STEP * math.sin(h)
        segs.append((x, y, nx, ny, radius_at(r), radius_at(math.hypot(nx, ny))))
        x, y, since = nx, ny, since + STEP

wedge = 2 * math.pi / N_TRUNKS
for i in range(N_TRUNKS):
    lo = i * wedge + rng.uniform(-0.04, 0.04)
    hi = (i + 1) * wedge
    th = (lo + hi) / 2
    grow(1.5 * math.cos(th), 1.5 * math.sin(th), th, lo, hi, R_MAX - 4 - rng.uniform(0, 7))

# ---- smooth SDF of half-capsules (round section, flat bottom on z=0) ----
ext = DIAMETER / 2 + 14
Z_SCALE = 0.72          # <1 makes the round section taller than a half-circle
zmax = R_TRUNK / Z_SCALE + 2.0
xs = np.arange(-ext, ext, RES)
zs = np.arange(0, zmax, RES)
nx_, nz = len(xs), len(zs)
D = np.full((nx_, nx_, nz), 10.0, dtype=np.float32)

def smin(a, b, k):
    h = np.clip(0.5 + 0.5 * (b - a) / k, 0, 1)
    return b * (1 - h) + a * h - k * h * (1 - h)

for (x0, y0, x1, y1, r0, r1) in segs:
    rmax = max(r0, r1) + BLEND + 0.5
    i0 = max(int((min(x0, x1) - rmax + ext) / RES), 0)
    i1 = min(int((max(x0, x1) + rmax + ext) / RES) + 2, nx_)
    j0 = max(int((min(y0, y1) - rmax + ext) / RES), 0)
    j1 = min(int((max(y0, y1) + rmax + ext) / RES) + 2, nx_)
    X = xs[i0:i1, None, None]; Y = xs[None, j0:j1, None]; Z = zs[None, None, :]
    dx, dy = x1 - x0, y1 - y0
    L2 = dx * dx + dy * dy or 1e-9
    t = np.clip(((X - x0) * dx + (Y - y0) * dy) / L2, 0, 1)
    px, py = X - (x0 + t * dx), Y - (y0 + t * dy)
    r = r0 + (r1 - r0) * t
    d = np.sqrt(px * px + py * py + (Z * Z_SCALE) ** 2) - r
    D[i0:i1, j0:j1, :] = smin(D[i0:i1, j0:j1, :], d.astype(np.float32), BLEND)

# center hub dome
X = xs[:, None, None]; Y = xs[None, :, None]; Z = zs[None, None, :]
hub = np.sqrt(X**2 + Y**2 + (Z * Z_SCALE * 0.95) ** 2) - 3.4
D = smin(D, hub.astype(np.float32), 1.5)

D[:, :, 0] = np.where(D[:, :, 0] < 0, -1.0, D[:, :, 0])  # exact flat base
# close the flat bottom: add a slab below z=0 that is outside
vol = np.concatenate([np.full((nx_, nx_, 1), 1.0, np.float32), D,
                      np.full((nx_, nx_, 1), 1.0, np.float32)], axis=2)
verts, faces, _, _ = measure.marching_cubes(vol, 0.0, spacing=(RES, RES, RES))
verts[:, 0] -= ext; verts[:, 1] -= ext; verts[:, 2] -= RES
import fast_simplification
v2, f2 = fast_simplification.simplify(verts.astype(np.float32), faces.astype(np.int64), target_reduction=0.75)
mesh = trimesh.Trimesh(v2, f2, process=True)
mesh.merge_vertices()
parts = mesh.split(only_watertight=False)
mesh = max(parts, key=lambda m: len(m.faces))   # keep the single connected piece
mesh.update_faces(mesh.nondegenerate_faces())
mesh.remove_unreferenced_vertices()
import pymeshfix
fv, ff = pymeshfix.clean_from_arrays(np.ascontiguousarray(mesh.vertices, dtype=np.float64), np.ascontiguousarray(mesh.faces, dtype=np.int32))  # clear decimation non-manifolds
mesh = trimesh.Trimesh(fv, ff, process=True)
mesh.fix_normals()
if mesh.volume < 0: mesh.invert()
mesh.apply_translation([-mesh.bounds[0][0] - mesh.extents[0] / 2,
                        -mesh.bounds[0][1] - mesh.extents[1] / 2, -mesh.bounds[0][2]])
print("piece faces", sorted(len(p.faces) for p in parts)[-3:]); print("segments", len(segs), "pieces", len(parts), "watertight", mesh.is_watertight,
      "extents mm", np.round(mesh.extents, 2), "volume cm3", round(mesh.volume / 1000, 1),
      "faces", len(mesh.faces))
mesh.export("coral_burst.stl")
mesh.export("coral_burst.3mf")
