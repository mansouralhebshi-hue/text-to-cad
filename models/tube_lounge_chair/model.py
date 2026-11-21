"""Tube Lounge Chair: 800 mm wide chromed-tube lounge chair, split for a Creality K1C.

The chair is five Ø100 tubes side by side (front and rear arches with legs, three
seat tubes), two stacked Ø100 back cushions and a centre ball foot. A K1C prints
220 x 220 x 250 mm, so every tube is cut into segments perpendicular to its axis.
Each segment prints standing on a cut face with no supports (bends are cut so no
piece leans past 42 degrees). Each cut has a Ø10 dowel hole and a Ø6 anti-rotation
pin hole. Neighbouring tubes meet on flat glue strips.

Run: python model.py  ->  parts/*.stl, chair_assembled.stl, chair_scale_1to4.stl
"""
import math, os, json
import numpy as np
import trimesh
from manifold3d import Manifold, Mesh

W = 800.0            # overall width (mm)
R = 50.0             # tube radius
GAP = 92.0           # axis spacing (tubes overlap 8 mm -> 39 mm wide glue strips)
SEAT_Z = 330.0       # seat tube axis height (seat top 380)
FILLET = 110.0       # bend radius at the arch corners
SEG = 128            # circle resolution
BED = (210.0, 210.0, 240.0)   # K1C 220x220x250 minus margin
MAX_TILT = math.radians(42)
DOWEL_R, DOWEL_D = 5.2, 22.0  # Ø10.4 hole, 22 mm deep each side (Ø10 x 40 dowel)
PIN_R, PIN_D, PIN_OFF = 3.15, 15.0, 28.0  # Ø6 x 28 pin

FOOT_X = W / 2 - R

def fillet_path(pts, r, n=40):
    """Polyline with circular fillets at interior corners -> dense list of points."""
    pts = [np.array(p, float) for p in pts]
    out = [pts[0]]
    for i in range(1, len(pts) - 1):
        a, b, c = pts[i - 1], pts[i], pts[i + 1]
        u, v = (a - b) / np.linalg.norm(a - b), (c - b) / np.linalg.norm(c - b)
        ang = math.acos(np.clip(u @ v, -1, 1))
        tlen = r / math.tan(ang / 2)
        p1, p2 = b + u * tlen, b + v * tlen
        bis = (u + v) / np.linalg.norm(u + v)
        ctr = b + bis * r / math.sin(ang / 2)
        a1 = math.atan2(*(p1 - ctr)[::-1]); a2 = math.atan2(*(p2 - ctr)[::-1])
        d = (a2 - a1 + math.pi) % (2 * math.pi) - math.pi
        out.append(p1)
        for k in range(1, n):
            t = a1 + d * k / n
            out.append(ctr + r * np.array([math.cos(t), math.sin(t)]))
        out.append(p2)
    out.append(pts[-1])
    return out

def densify(pts, step=4.0):
    res = [pts[0]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = max(1, int(np.linalg.norm(b - a) / step))
        res += [a + (b - a) * k / n for k in range(1, n + 1)]
    return np.array(res)

def xz_to_3d(p2, y):
    return np.array([[p[0], y, p[1]] for p in p2])

# ---------------- paths ----------------
arch2d = densify(fillet_path([(-FOOT_X, 40), (-FOOT_X + 50, SEAT_Z), (FOOT_X - 50, SEAT_Z), (FOOT_X, 40)], FILLET))
mid2d = arch2d[arch2d[:, 1] >= 250]          # seat tubes curl down partway into the bend
BACK_HALF = 130.0
back2d = densify([np.array([-BACK_HALF, 0.0]), np.array([BACK_HALF, 0.0])])

ys = [-2 * GAP, -GAP, 0.0, GAP, 2 * GAP]
BACK_Z1 = SEAT_Z + GAP
BACK_Z2 = BACK_Z1 + GAP
POST_TOP = SEAT_Z - R + 10                   # flat glue face on the bottom of the centre tube

tubes = {
    "front": dict(path=xz_to_3d(arch2d, ys[0]), split=0.0, feet=True),
    "seat1": dict(path=xz_to_3d(mid2d, ys[1]), split=70.0, feet=False),
    "seat2": dict(path=xz_to_3d(mid2d, ys[2]), split=-70.0, feet=False),
    "seat3": dict(path=xz_to_3d(mid2d, ys[3]), split=70.0, feet=False),
    "rear":  dict(path=xz_to_3d(arch2d, ys[4]), split=0.0, feet=True),
    "back1": dict(path=np.array([[p[0], ys[4], BACK_Z1] for p in back2d]), split=-50.0, feet=False),
    "back2": dict(path=np.array([[p[0], ys[4], BACK_Z2] for p in back2d]), split=50.0, feet=False),
}

# ---------------- helpers ----------------
def sphere(c, r=R):
    return Manifold.sphere(r, SEG).translate(tuple(c))

def tube_along(pts, r=R, nseg=SEG, ncap=24):
    """Swept circular tube with hemispherical end caps, as one clean manifold mesh."""
    pts = np.asarray(pts, float)
    tans = np.array([tangent(pts, i) for i in range(len(pts))])
    # rotation-minimising frames
    e1 = np.cross(tans[0], [0, 1.0, 0] if abs(tans[0][1]) < 0.9 else [1.0, 0, 0]); e1 /= np.linalg.norm(e1)
    frames = []
    for t in tans:
        e1 = e1 - (e1 @ t) * t; e1 /= np.linalg.norm(e1)
        frames.append((e1, np.cross(t, e1)))
    ang = np.linspace(0, 2 * np.pi, nseg, endpoint=False)
    ca, sa = np.cos(ang), np.sin(ang)
    rings = []
    def ring(c, f, rr):
        return c + rr * (ca[:, None] * f[0] + sa[:, None] * f[1])
    # start cap (pole first), body rings, end cap
    phis = np.linspace(0, np.pi / 2, ncap + 1)[1:-1]
    for ph in phis:                                   # start cap rings
        rings.append(ring(pts[0] - tans[0] * r * np.cos(ph), frames[0], r * np.sin(ph)))
    for c, f in zip(pts, frames):
        rings.append(ring(c, f, r))
    for ph in phis[::-1]:                             # end cap rings
        rings.append(ring(pts[-1] + tans[-1] * r * np.cos(ph), frames[-1], r * np.sin(ph)))
    V = np.vstack([pts[0] - tans[0] * r] + rings + [pts[-1] + tans[-1] * r])
    nr = len(rings); F = []
    base = lambda k: 1 + k * nseg
    j = np.arange(nseg); jn = (j + 1) % nseg
    F += [np.column_stack([np.zeros(nseg, int), base(0) + jn, base(0) + j])]
    for k in range(nr - 1):
        a, b = base(k), base(k + 1)
        F += [np.column_stack([a + j, a + jn, b + jn]), np.column_stack([a + j, b + jn, b + j])]
    last = len(V) - 1; a = base(nr - 1)
    F += [np.column_stack([np.full(nseg, last), a + j, a + jn])]
    F = np.vstack(F)
    m = Manifold(Mesh(vert_properties=V.astype(np.float32), tri_verts=F.astype(np.uint32)))
    if m.volume() < 0 or m.is_empty():
        m = Manifold(Mesh(vert_properties=V.astype(np.float32), tri_verts=F[:, ::-1].astype(np.uint32)))
    assert not m.is_empty(), m.status()
    return m

def manifold3d_op(name):
    import manifold3d
    return getattr(manifold3d.OpType, name)

def frame(t):
    t = t / np.linalg.norm(t)
    ref = np.array([0, 1.0, 0]) if abs(t[1]) < 0.9 else np.array([1.0, 0, 0])
    e1 = np.cross(ref, t); e1 /= np.linalg.norm(e1)
    e2 = np.cross(t, e1)
    return e1, e2, t

def cyl_between(p, axis, r, half_len):
    """Cylinder of radius r centred on p along axis, total length 2*half_len."""
    e1, e2, t = frame(axis)
    c = Manifold.cylinder(2 * half_len, r, r, 48, True)
    m = np.column_stack([e1, e2, t, p])        # 3x4: rotate z->axis, then translate
    return c.transform(m)

def box(lo, hi):
    lo, hi = np.array(lo, float), np.array(hi, float)
    return Manifold.cube(tuple(hi - lo)).translate(tuple(lo))

def tangent(path, i):
    a, b = path[max(i - 1, 0)], path[min(i + 1, len(path) - 1)]
    return (b - a) / np.linalg.norm(b - a)

def inplane_normal(t):
    n = np.array([-t[2], 0.0, t[0]]) if abs(t[1]) < 0.9 else np.array([0, 0, 1.0])
    return n / np.linalg.norm(n)

# ---------------- segmentation ----------------
def stations_for_half(path, i0, step_dir, origin_is_foot):
    """Greedy cut indices walking from i0 in step_dir until the path end."""
    cuts, i, end = [], i0, (len(path) - 1 if step_dir > 0 else 0)
    first = True
    while i != end:
        t0 = tangent(path, i) * step_dir
        e1, e2, tz = (np.eye(3) if (first and origin_is_foot) else frame(t0))
        if first and origin_is_foot:
            e1, e2, tz = np.array([1., 0, 0]), np.array([0, 1., 0]), np.array([0, 0, 1.])
        best, j = None, i
        while j != end:
            j += step_dir
            seg = path[min(i, j):max(i, j) + 1]
            loc = (seg - path[i]) @ np.column_stack([e1, e2, tz])
            ext = loc.max(0) - loc.min(0)
            tilt = math.acos(np.clip(tangent(path, j) * step_dir @ t0, -1, 1))
            zext = ext[2] + R + (R if j == end else 0) + (40 if (first and origin_is_foot) else 0)
            if ext[0] + 2 * R > BED[0] or ext[1] + 2 * R > BED[1] or zext > BED[2] or \
               (tilt > MAX_TILT and not (first and origin_is_foot)) or \
               (first and origin_is_foot and tilt > math.radians(60)):
                break
            best = j
        if best is None:
            raise RuntimeError("cannot segment")
        cuts.append(best)
        i, first = best, False
    return cuts

def segment(name, spec):
    path = spec["path"]
    isplit = int(np.argmin(np.abs(path[:, 0] - spec["split"])))
    pieces = []   # (i_start, i_end, travel_dir, orient_identity)
    if spec["feet"]:
        # walk from each foot inward to the split
        for start, d in ((0, 1), (len(path) - 1, -1)):
            sub_end = isplit
            idx = start; first = True
            sub = path[:isplit + 1] if d > 0 else path[isplit:][::-1]
            cuts = stations_for_half(sub, 0, 1, True)
            prev = 0
            for c in cuts:
                a, b = (prev, c) if d > 0 else (len(path) - 1 - prev, len(path) - 1 - c)
                pieces.append((a, b, d, prev == 0))
                prev = c
    else:
        for d in (1, -1):
            sub = path[isplit:] if d > 0 else path[:isplit + 1][::-1]
            cuts = stations_for_half(sub, 0, 1, False)
            prev = 0
            for c in cuts:
                a, b = (isplit + prev, isplit + c) if d > 0 else (isplit - prev, isplit - c)
                pieces.append((a, b, d, False))
                prev = c
    return pieces

# ---------------- build ----------------
os.makedirs("parts", exist_ok=True)
for f in os.listdir("parts"):
    os.remove(os.path.join("parts", f))

def flatten(m, name):
    """Glue-strip trims where this tube meets its neighbours."""
    big = 2000
    if name in ("front", "seat1", "seat2", "seat3", "rear"):
        k = ["front", "seat1", "seat2", "seat3", "rear"].index(name)
        y = ys[k]
        if k > 0:
            m = m.trim_by_plane((0, 1, 0), y - GAP / 2)
        if k < 4:
            m = m.trim_by_plane((0, -1, 0), -(y + GAP / 2))
    if name == "rear":
        m = m - box((-BACK_HALF - R - 5, ys[4] - 80, BACK_Z1 - GAP / 2), (BACK_HALF + R + 5, ys[4] + 80, 600))
    if name == "seat2":
        m = m - box((-45, -60, 0), (45, 60, POST_TOP))
    if name == "back1":
        m = m.trim_by_plane((0, 0, 1), BACK_Z1 - GAP / 2).trim_by_plane((0, 0, -1), -(BACK_Z1 + GAP / 2))
    if name == "back2":
        m = m.trim_by_plane((0, 0, 1), BACK_Z2 - GAP / 2)
    return m

# joint dowels between components (vertical)
joint_holes = []
for x in (-90.0, 90.0):
    joint_holes.append(cyl_between(np.array([x, ys[4], BACK_Z1 - GAP / 2]), np.array([0, 0, 1.]), DOWEL_R, DOWEL_D))
    joint_holes.append(cyl_between(np.array([x, ys[4], BACK_Z2 - GAP / 2]), np.array([0, 0, 1.]), DOWEL_R, DOWEL_D))
joint_holes.append(cyl_between(np.array([0, 0, POST_TOP]), np.array([0, 0, 1.]), DOWEL_R, DOWEL_D))

def cut_holes(path, idx, d):
    t = tangent(path, idx)
    p = path[idx]
    n = inplane_normal(t)
    return [cyl_between(p, t, DOWEL_R, DOWEL_D), cyl_between(p + n * PIN_OFF, t, PIN_R, PIN_D)]

parts, assembled, centres = [], [], {}
for name, spec in tubes.items():
    path = spec["path"]
    pcs = segment(name, spec)
    cut_idx = set()
    for a, b, d, foot in pcs:
        for i in (a, b):
            if 0 < i < len(path) - 1:
                cut_idx.add(i)
    holes = [h for i in cut_idx for h in cut_holes(path, i, 1)]
    for n_, (a, b, d, foot) in enumerate(pcs):
        lo, hi = min(a, b), max(a, b)
        ext = int(60 / 4)
        keep = [k for k in range(max(lo - ext, 0), min(hi + ext, len(path) - 1) + 1)
                if k not in (lo, hi) or k in (0, len(path) - 1)]
        m = tube_along(path[keep])   # no vertex ring may sit exactly on a cut plane
        if lo > 0:
            m = m.trim_by_plane(tuple(tangent(path, lo)), float(tangent(path, lo) @ path[lo]))
        if hi < len(path) - 1:
            th = -tangent(path, hi)
            m = m.trim_by_plane(tuple(th), float(th @ path[hi]))
        if spec["feet"]:
            m = m.trim_by_plane((0, 0, 1), 0.0)
        m = flatten(m, name)
        for h in holes + joint_holes:
            m = m - h
        assembled.append(m)
        bbm = np.array(m.bounding_box()); centres_pending = (bbm[:3] + bbm[3:]) / 2
        # print orientation: start face (toward the origin of travel) down
        if foot:
            Rm = np.eye(3)
        else:
            s = a
            t0 = tangent(path, s) * d
            e1, e2, tz = frame(t0)
            Rm = np.vstack([e1, e2, tz])
        pm = m.transform(np.column_stack([Rm, np.zeros(3)]))
        bb = np.array(pm.bounding_box())
        pm = pm.translate(tuple(-(bb[:3] + bb[3:]) / 2 * np.array([1, 1, 0]) - np.array([0, 0, bb[2]])))
        side = "L" if (path[a][0] + path[b][0]) / 2 < 0 else "R"
        parts.append((f"{name}_{side}{n_ + 1}", pm))
        centres[f"{name}_{side}{n_ + 1}"] = centres_pending.tolist()

# centre post: ball foot + column, two pieces
post = (sphere((0, 0, 40), 45) + Manifold.cylinder(POST_TOP - 40, 30, 30, SEG).translate((0, 0, 40))).trim_by_plane((0, 0, 1), 0)
split_z = 165.0
ph = [cyl_between(np.array([0, 0, split_z]), np.array([0, 0, 1.]), DOWEL_R, DOWEL_D),
      cyl_between(np.array([0, 18.0, split_z]), np.array([0, 0, 1.]), PIN_R, PIN_D)]
for h in ph + joint_holes:
    post = post - h
for nm, pc in (("post_1", post.trim_by_plane((0, 0, -1), -split_z)), ("post_2", post.trim_by_plane((0, 0, 1), split_z))):
    assembled.append(pc)
    bb = np.array(pc.bounding_box()); centres[nm] = ((bb[:3] + bb[3:]) / 2).tolist()
    parts.append((nm, pc.translate((0, 0, -bb[2]))))

def to_tri(m):
    mm = m.to_mesh()
    return trimesh.Trimesh(np.array(mm.vert_properties)[:, :3], np.array(mm.tri_verts), process=False)

report, total_vol, total_area = [], 0, 0
for nm, pm in parts:
    tm = to_tri(pm)
    if not tm.is_watertight:   # knife-edge seams where two glue trims meet: re-mesh them
        import pymeshfix
        v, f = pymeshfix.clean_from_arrays(np.ascontiguousarray(tm.vertices, dtype=np.float64),
                                           np.ascontiguousarray(tm.faces, dtype=np.int32),
                                           joincomp=True, remove_smallest_components=False)
        tm = trimesh.Trimesh(v, f)
    ext = tm.extents
    ok = ext[0] <= 220 and ext[1] <= 220 and ext[2] <= 250
    report.append((nm, np.round(ext, 1).tolist(), round(tm.volume / 1000, 1), tm.is_watertight, ok))
    total_vol += tm.volume; total_area += tm.area
    tm.export(f"parts/{nm}.stl")

# one-piece solid (no cuts/holes) for preview and the 1:4 desk model
solid = [tube_along(spec["path"]) for spec in tubes.values()]
solid.append(sphere((0, 0, 40), 45) + Manifold.cylinder(POST_TOP - 30, 30, 30, SEG).translate((0, 0, 40)))
whole = Manifold.batch_boolean(solid, manifold3d_op("Add")).trim_by_plane((0, 0, 1), 0.0)
wt = to_tri(whole)
wt.export("chair_assembled.stl")
small = to_tri(whole.scale((0.25, 0.25, 0.25)))
small.export("chair_scale_1to4.stl")

# dowels / pins to print (or use Ø10x40 hardwood dowels and Ø6x28 pins)
dw = to_tri(Manifold.cylinder(41, 4.9, 4.9, 48))
dw.export("parts/zz_dowel_10x41_print_xN.stl")
pn = to_tri(Manifold.cylinder(28, 2.85, 2.85, 32))
pn.export("parts/zz_pin_6x28_print_xN.stl")

for r_ in report:
    print(r_)
print("parts", len(parts), "all fit:", all(r[4] for r in report), "all watertight:", all(r[3] for r in report))
print("assembled extents", np.round(wt.extents, 1), "watertight", wt.is_watertight)
wall = 1.8   # 4 walls x 0.45
shell = total_area * wall
inner = max(total_vol - shell, 0)
for inf in (0.10, 0.15):
    g = (shell + inner * inf) / 1000 * 1.24
    print(f"PLA estimate @ {int(inf*100)}% infill: {g/1000:.1f} kg")
json.dump({"parts": {r[0]: dict(size_mm=r[1], volume_cm3=float(r[2]), centre_assembled=centres[r[0]]) for r in report}},
          open("parts/parts.json", "w"), indent=1)
