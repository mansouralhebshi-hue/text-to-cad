"""Varsity Figure: 160 mm articulated designer-toy figure for a Creality K1C.

Traced from the front, side and back sheets. The figure has 15 points of
articulation, all on printed pins:
  neck swivel . shoulders x2 . elbows x2 . wrists x2 . waist swivel
  hips x2 . knees x2 . ankles x2 (+ separate eye pins)
Every part prints on a flat face, oriented for the bed. Pin axes along X are
printed vertically (perfectly round holes). Any horizontal hole is a teardrop,
so it prints clean without supports. Colours are separate parts (skin / black /
white / denim), with optional white and beard detail bodies for a Creality CFS
multicolour print or for paint pens.

Run: python model.py
  parts/*.stl                     print-ready, one colour each (details merged in)
  multicolor/*.3mf                parts with colour-detail bodies split for CFS
  figure_assembled.glb            coloured assembly for viewing
"""
import math, os, json
import numpy as np
import trimesh
from manifold3d import Manifold, Mesh, CrossSection, OpType

SEG = 64
TIGHT, TURN = 0.08, 0.30      # hole oversize: press-fit / rotating friction
OUT = "parts"

# ----------------------------------------------------------------- primitives
def sph(c, r, seg=SEG):
    return Manifold.sphere(r, seg).translate(tuple(map(float, c)))

def hull(ms):
    return Manifold.batch_hull(list(ms))

def union(ms):
    ms = [m for m in ms if m is not None]
    return Manifold.batch_boolean(ms, OpType.Add) if len(ms) > 1 else ms[0]

def box(lo, hi):
    lo, hi = np.asarray(lo, float), np.asarray(hi, float)
    return Manifold.cube(tuple(hi - lo)).translate(tuple(lo))

def rbox(lo, hi, r, seg=32):
    lo, hi = np.asarray(lo, float) + r, np.asarray(hi, float) - r
    assert np.all(hi >= lo - 1e-9), f"rbox smaller than 2r: {lo - r} {hi + r} r={r}"
    return hull(sph((x, y, z), r, seg) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2]))

def frame(a, up=None):
    a = np.asarray(a, float); a /= np.linalg.norm(a)
    if up is not None:
        v = np.asarray(up, float) - (np.asarray(up, float) @ a) * a
        if np.linalg.norm(v) > 1e-6:
            v /= np.linalg.norm(v); return np.cross(v, a), v, a
    ref = np.array([0, 0, 1.0]) if abs(a[2]) < 0.9 else np.array([1.0, 0, 0])
    u = np.cross(ref, a); u /= np.linalg.norm(u)
    return u, np.cross(a, u), a

def place(m, p, axis, up=None):
    """Map local z -> axis, local y -> up (projected), origin -> p."""
    e1, e2, e3 = frame(axis, up)
    return m.transform(np.column_stack([e1, e2, e3, np.asarray(p, float)]))

def cyl(p0, p1, r, seg=SEG):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    L = np.linalg.norm(p1 - p0)
    return place(Manifold.cylinder(L, r, r, seg), p0, p1 - p0)

def hole(p, axis, d, depth_neg, depth_pos, up, clip=None):
    """Hole along axis from -depth_neg to +depth_pos around p. If axis is not
    parallel to the print-up direction, it is a teardrop pointing up, so it
    prints without support. `clip` truncates the teardrop tip (mm from axis)."""
    r = d / 2
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    L = depth_neg + depth_pos
    if up is None or abs(a @ (np.asarray(up, float) / np.linalg.norm(up))) > 0.97:
        cs = CrossSection.circle(r, 48)
    else:
        k = r * math.sqrt(2)
        tri = CrossSection([[(-r * math.sqrt(0.5), r * math.sqrt(0.5)), (r * math.sqrt(0.5), r * math.sqrt(0.5)), (0, k)]])
        cs = CrossSection.circle(r, 48) + hull_cs(tri, r)
        if clip is not None:
            cs = cs - CrossSection.square((4 * r, 4 * r)).translate((-2 * r, clip))
    m = cs.extrude(L).translate((0, 0, -depth_neg))
    return place(m, p, a, up)

def hull_cs(tri, r):
    return CrossSection.batch_hull([tri, CrossSection.circle(r * 0.999, 48).scale((0.72, 0.72))])

def rot_x(m, deg, p):
    """Rotate about an X-parallel axis through p (positive swings -z toward +y)."""
    c, s_ = math.cos(math.radians(deg)), math.sin(math.radians(deg))
    R = np.array([[1, 0, 0], [0, c, -s_], [0, s_, c]]); p = np.asarray(p, float)
    return m.transform(np.column_stack([R, p - R @ p]))

def swept_slot(x0, x1, p, r, w, L, a0, a1, step=4):
    """Clearance for a hinge tongue (round head r, stem half-width w, length L below
    the axis) swept through its whole arc, built as one clean 2-D fan in the Y-Z
    plane and extruded along X."""
    stem = CrossSection.square((2 * w, L)).translate((-w, -L))
    fan = CrossSection.circle(r, 64)
    for a in np.arange(a0, a1 + 1e-6, step):
        fan = fan + stem.rotate(float(a))
    m = fan.extrude(x1 - x0)
    return m.transform(np.array([[0, 0, 1, x0], [1, 0, 0, p[1]], [0, 1, 0, p[2]]], float))

def slab(axis_i, lo, hi):
    l, h = [-1e3] * 3, [1e3] * 3
    l[axis_i], h[axis_i] = lo, hi
    return box(l, h)

def cs_poly(pts):
    pts = [tuple(map(float, p)) for p in pts]
    area = sum(a[0] * b[1] - b[0] * a[1] for a, b in zip(pts, pts[1:] + pts[:1]))
    return CrossSection([pts if area > 0 else pts[::-1]])

def front_decal(cs_xz, shape_out, shape_in, y0=-60, y1=5):
    """2-D shape drawn in front view (x, z), projected onto a body as a raised skin."""
    m = cs_xz.extrude(y1 - y0).translate((0, 0, y0))          # (x, z_as_y, y_as_z)
    m = m.transform(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0]], float))   # drawn as seen from the front
    return (shape_out - shape_in) ^ m

# ----------------------------------------------------------------- body plan (mm)
# x = figure's left (+), y = back (+), front faces -y, z up. 160 mm tall.
NECK_Z, SHOULDER_Z, ELBOW_Z, HIP_Z, KNEE_Z, ANKLE_Z = 106.7, 95.8, 86.0, 49.5, 42.0, 15.5
KNUCKLE_GAP = 0.4
ANKLE_Y = 3.0
TORSO_BOT, TORSO_TOP = 57.9, 106.7
LEG_X = 10.3            # leg centreline
ARM_X0, ARM_X1 = 20.3, 33.8
ARM_XC = (ARM_X0 + ARM_X1) / 2

parts = {}        # name -> dict(body, details{name:(manifold,colour)}, colour, up, count)

def add(name, body, colour, up, count=1, details=None, note=""):
    parts[name] = dict(body=body, colour=colour, up=np.asarray(up, float), count=count,
                       details=details or {}, note=note)

# ================================================================= TORSO
def torso_shape(d=0.0):
    pts = []
    for x in (-1, 1):
        for y in (-1, 1):
            pts.append(sph((x * (19.9 - 2.5), y * (14.0 - 2.5), TORSO_BOT + 2.5), 2.5 + d))
            pts.append(sph((x * (19.3 - 4.0), y * (14.0 - 4.0), 80.0), 4.0 + d))
            pts.append(sph((x * (17.15 - 7.0), y * (13.2 - 7.0), TORSO_TOP - 7.0), 7.0 + d))
    return hull(pts)

T_UP = (0, -1, 0)                          # torso prints lying on its back
torso = torso_shape()
shoulder_details = []
# white details (raised 0.6 mm skins)
skin6 = (torso_shape(0.6), torso)
hem = (torso - torso_shape(-0.7)) ^ (slab(2, 58.6, 59.7) + slab(2, 61.3, 62.4))   # inlaid flush
collar = Manifold.cylinder(1.8, 8.6, 8.6, SEG).translate((0, 0, TORSO_TOP - 0.6)) - \
         Manifold.cylinder(4, 5.4, 5.4, SEG).translate((0, 0, TORSO_TOP - 1))
collar_stripe = Manifold.cylinder(0.6, 8.9, 8.9, SEG).translate((0, 0, TORSO_TOP + 0.3)) - \
                Manifold.cylinder(4, 5.4, 5.4, SEG).translate((0, 0, TORSO_TOP - 1))
shirt = front_decal(cs_poly([(-5.2, 63.0), (-0.6, 63.0), (-0.6, 104.0), (-5.2, 104.0)]), *skin6)
v_neck = front_decal(cs_poly([(-6.5, 106.9), (6.5, 106.9), (0.0, 98.0)]), torso_shape(0.9), torso)
buttons = union([sph((-2.6, -14.0, z), 1.15, 24) for z in (99.0, 90.0, 81.0, 72.0)]) - torso
def D_letter():
    outer = CrossSection.circle(4.6, 64).scale((0.78, 1.0)) ^ CrossSection.square((10, 10)).translate((0, -5))
    outer = outer + CrossSection.square((1.8, 9.2)).translate((-1.8, -4.6))
    inner = CrossSection.circle(2.7, 64).scale((0.62, 1.0)) ^ CrossSection.square((10, 10)).translate((0, -5))
    inner = inner + CrossSection.square((0.6, 5.4)).translate((-0.6, -2.7))
    serif = CrossSection.square((4.0, 1.0)).translate((-3.0, -4.6)) + CrossSection.square((4.0, 1.0)).translate((-3.0, 3.6))
    return (outer - inner + serif).translate((11.0, 88.0))
d_letter = front_decal(D_letter(), *skin6)
welts = [front_decal(cs_poly([(-0.75, -4.2), (0.75, -4.2), (0.75, 4.2), (-0.75, 4.2)]).rotate(-s * 38).translate((s * 12.6, 69.5)), *skin6)
         for s in (-1, 1)]
white = union([hem, collar - torso, collar_stripe - torso, shirt, v_neck, buttons, d_letter] + welts)
# black hem rib between stripes is the torso itself; collar gets a black stripe
# joints
torso_cut = [
    hole((0, 0, 64.9 - 4.0), (0, 0, 1), 19.8, 8.5, 4.0, T_UP, clip=11.0),               # waist cavity
    hole((0, 0, 64.9), (0, 0, 1), 5 + TURN, 0.5, 7.5, T_UP),                             # waist pin (turns)
    hole((0, 0, NECK_Z), (0, 0, 1), 4.5 + TIGHT, 8.0, 1.0, T_UP),                        # neck pin (fixed)
]
for s in (-1, 1):
    torso_cut.append(hole((s * 17.6, 0, SHOULDER_Z), (s, 0, 0), 4 + TIGHT, 7.5, 2.0, T_UP))  # shoulder pin (fixed)
torso_body = torso - union(torso_cut) - hem
white = white - union(torso_cut)
add("torso", torso_body, "black", T_UP, details={"torso_white": (white, "white")},
    note="lie on its back; front details face up")

# ================================================================= HEAD
HEAD_C = np.array([0, 0, 131.0])
def head_blob(d=0.0):
    return hull([sph((0, 0.5, 137.0), 16.7 + d), sph((0, -0.5, 124.0), 15.5 + d)])
neck = Manifold.cylinder(112.0 - (NECK_Z + 0.2), 4.5, 4.5, SEG).translate((0, 0, NECK_Z + 0.2))
EAR = [np.array([s * 16.4, 2.0, 126.0]) for s in (-1, 1)]
ears = union([cyl(e - [s * 1.9, 0, 0], e + [s * 1.9, 0, 0], 4.8) -
              cyl(e + [s * 1.0, 0, 0], e + [s * 2.5, 0, 0], 2.6) for s, e in zip((-1, 1), EAR)])
head = union([head_blob(), neck, ears])
# beard (black detail skin)
beard_out = cs_poly([(-18.8, 129.5), (-18.8, 120), (-16.8, 113), (-12.2, 108.0), (-6, 105.6), (0, 105.2), (6, 105.6),
                     (12.2, 108.0), (16.8, 113), (18.8, 120), (18.8, 129.5), (15.4, 127.0), (12.0, 121.0),
                     (9.2, 118.6), (6.6, 121.6), (2.0, 122.2), (-2.0, 122.2), (-6.6, 121.6), (-9.2, 118.6),
                     (-12.0, 121.0), (-15.4, 127.0)])
mouth = cs_poly([(-4.9, 118.4), (-3.9, 116.2), (3.9, 116.2), (4.9, 118.4)])
beard = front_decal(beard_out - mouth, head_blob(0.9), head, y0=-60, y1=4.0) - ears
# eyes: holes for black eye pins
eye_pts = []
for s in (-1, 1):
    hit = head_blob().ray_cast((s * 7.7, -60, 128.4), (s * 7.7, 0, 128.4))[0].position
    eye_pts.append(np.array(hit))
head_cut = [hole((0, 0, NECK_Z + 0.2), (0, 0, 1), 4.5 + TURN, 1.0, 8.0, (0, 0, 1))]
for p in eye_pts:
    head_cut.append(hole(p, (0, 1, 0), 3.0 + TIGHT, 1.0, 3.5, (0, 0, 1)))
add("head", head - union(head_cut) - beard, "skin", (0, 0, 1),
    details={"head_beard": (beard - union(head_cut), "black")},
    note="stand on the neck; tree supports under chin and ears")
add("eye_pin", union([Manifold.cylinder(3.6, 1.5, 1.5, 32), sph((0, 0, 3.6), 1.55, 32) ^ slab(2, 3.6, 9)]),
    "black", (0, 0, 1), count=2, note="press into the eye holes, dome out")

# ================================================================= HAIR (cornrows + hanging braids)
rng = np.random.default_rng(3)
beads, anchors = [], []
def surf(direction):
    d = np.asarray(direction, float); d /= np.linalg.norm(d)
    h = head_blob().ray_cast(tuple(HEAD_C), tuple(HEAD_C + d * 60))
    return np.array(h[-1].position) if h else None
ROWS = np.radians(np.linspace(-62, 62, 11))
for phi in ROWS:
    th0 = math.radians(78 - 34 * (abs(phi) / math.radians(62)) ** 2)
    th1 = math.radians(238)
    last, acc, row_pts = None, 0, []
    for th in np.linspace(th0, th1, 400):
        dirn = np.array([math.sin(phi), -math.cos(phi) * math.cos(th), math.cos(phi) * math.sin(th) * 1.0])
        p = surf(dirn)
        if p is None:
            continue
        if last is not None:
            acc += np.linalg.norm(p - last)
        last = p
        rb = 2.55 if len(row_pts) % 2 == 0 else 2.3
        if not row_pts or acc >= 0.78 * (rb + 2.3):
            if all(np.linalg.norm(p - e) > 7.0 for e in EAR):
                c = p + (p - HEAD_C) / np.linalg.norm(p - HEAD_C) * rb * 0.35
                row_pts.append((c, rb)); acc = 0
    beads += row_pts
    # hanging braid from the nape down to just above the collar
    c, rb = row_pts[-1]
    out = np.array([c[0] - 0.0, c[1] - HEAD_C[1], 0.0]); out /= max(np.linalg.norm(out), 1e-6)
    z = c[2]; k = 0
    while True:
        k += 1
        rb2 = 2.5 - 0.06 * k
        z -= 0.74 * 2 * rb2
        if z - rb2 < NECK_Z + 1.2:
            break
        beads.append((np.array([c[0] + out[0] * 0.4 * k, c[1] + out[1] * 0.25 * k, z]), rb2))
bead_ms = [sph(c, r, 28) for c, r in beads]
shell_mask = union([sph(c, r + 1.6, 20) for c, r in beads if c[2] > 112])
hair = union(bead_ms + [(head_blob(1.3) - head_blob()) ^ shell_mask]) - head_blob(0.12) - (ears + Manifold.sphere(0.01))
hair = hair - (head_blob(1.4) ^ front_decal(beard_out, head_blob(1.4), Manifold(), y0=-60, y1=6.0)) \
            - head_blob(0.12) - union([cyl(e - [s_ * 2.4, 0, 0], e + [s_ * 2.4, 0, 0], 5.3) for s_, e in zip((-1, 1), EAR)])
add("hair", hair, "black", (0, 0, -1),
    note="print crown-down with tree supports; glue onto the head")

# ================================================================= ARMS
for s, side in ((-1, "R"), (1, "L")):          # R = figure's right (image left)
    X = lambda x: s * x
    up = (s, 0, 0)                               # outer face up, flat inner face on the bed
    xr = sorted((X(ARM_X0), X(ARM_X1)))
    sl = slab(0, *xr)
    xc = X(ARM_XC)
    # --- upper arm: rounded shoulder, full-width elbow knuckle with a hidden slot
    RK = 7.4
    ua = hull([sph((xc, 0, 98.0), 7.2), cyl((xr[0], 0, ELBOW_Z), (xr[1], 0, ELBOW_Z), RK)]) ^ sl ^ slab(1, -7.4, 7.4)
    ring = (cyl((X(ARM_X1), 0, SHOULDER_Z), (X(ARM_X1 + 0.7), 0, SHOULDER_Z), 5.4) -
            cyl((X(ARM_X1 - 1), 0, SHOULDER_Z), (X(ARM_X1 + 1), 0, SHOULDER_Z), 4.0)) + \
           cyl((X(ARM_X1), 0, SHOULDER_Z), (X(ARM_X1 + 0.35), 0, SHOULDER_Z), 3.3)
    ua = ua + ring
    slot_a = swept_slot(xc - 3.1, xc + 3.1, (0, 0, ELBOW_Z), 4.4, 4.4, 26.0, -112, 0)  # elbow folds forward
    ua = ua - slot_a - hole((X(ARM_X0), 0, SHOULDER_Z), (s, 0, 0), 4 + TURN, 1, 8.0, up) \
            - hole((xc, 0, ELBOW_Z), (1, 0, 0), 3 + TIGHT, 9, 9, up)
    add(f"upper_arm_{side}", ua, "black", up)
    # --- lower arm: sleeve cuff, tongue rising into the elbow knuckle, wrist socket below
    la_top = ELBOW_Z - RK - KNUCKLE_GAP
    la = rbox((xr[0] - 0.5 * (s < 0), -8.0, 62.3), (xr[1] + 0.5 * (s > 0), 8.0, la_top), 2.2)
    la = la + cyl((xc - 2.8, 0, ELBOW_Z), (xc + 2.8, 0, ELBOW_Z), 4.0) + box((xc - 2.8, -4.0, la_top - 1), (xc + 2.8, 4.0, ELBOW_Z))
    bevel = place(Manifold.cube((60, 20, 20), True), (xc, -8.0 - 7.07 + 3.2, la_top + 7.07 - 3.2), (0, -1, 1), up=(0, 1, 1))
    la = la - bevel
    la = la - hole((xc, 0, ELBOW_Z), (1, 0, 0), 3 + TURN, 5, 5, up) \
            - hole((xc, 0, 62.3), (0, 0, 1), 3.5 + TIGHT, 1, 7.0, up)
    add(f"lower_arm_{side}", la, "black", up)
    # --- hand: wrist + C-claw, flat both faces, swivels on the wrist pin
    cs = CrossSection.circle(7.0, 64) - CrossSection.circle(3.3, 48)
    gap_dir = math.pi if s > 0 else 0.0           # opening faces the body
    w = math.radians(32)
    wedge = cs_poly([(0, 0), (12 * math.cos(gap_dir - w), 12 * math.sin(gap_dir - w)),
                     (12 * math.cos(gap_dir + w), 12 * math.sin(gap_dir + w))])
    c_shape = (cs - wedge).offset(-1.0, circular_segments=24)
    c_m = c_shape.extrude(5.0).translate((0, 0, -2.5)).minkowski_sum(Manifold.sphere(1.0, 12))
    c_m = c_m.transform(np.array([[1, 0, 0, xc + s * 2.2], [0, 0, 1, 0], [0, 1, 0, 49.8]], float))
    wrist = cyl((xc, 0, 55.0), (xc, 0, 62.0), 3.5)
    hand = (c_m + wrist) - hole((xc, 0, 62.0), (0, 0, 1), 3.5 + TURN, 6.0, 1, (0, -1, 0))
    add(f"hand_{side}", hand, "skin", (0, -1, 0))

# ================================================================= HIPS (waist swivel + hip axle carrier)
hips = union([rbox((-1.6, -9.0, 44.0), (1.6, 9.0, 59.0), 1.2),
              Manifold.cylinder(6.5, 9.5, 9.5, SEG).translate((0, 0, 58.0))])
hips = hips - hole((0, 0, HIP_Z), (1, 0, 0), 4 + TIGHT, 3, 3, (0, 0, -1)) \
            - hole((0, 0, 64.5), (0, 0, 1), 5 + TIGHT, 6.0, 1, (0, 0, -1))
add("hips", hips, "denim", (0, 0, -1), note="print disc-down")

# ================================================================= LEGS
for s, side in ((-1, "R"), (1, "L")):
    xc = s * LEG_X
    up = (1, 0, 0)
    tx = sorted((s * 2.0, s * 18.6))
    sl = slab(0, *tx)
    # thigh: hip barrel + full-depth knee knuckle with a hidden slot (straight jeans look)
    RKN = 11.0
    th = hull([cyl((tx[0], 0, HIP_Z), (tx[1], 0, HIP_Z), 8.0), cyl((tx[0], 0, KNEE_Z), (tx[1], 0, KNEE_Z), RKN),
               rbox((tx[0], -11.0, 46.0), (tx[1], 11.0, 50.0), 2.0)]) ^ sl ^ slab(1, -11.0, 11.0)
    slot_k = swept_slot(xc - 3.1, xc + 3.1, (0, 0, KNEE_Z), 4.9, 4.9, 32.0, 0, 112)   # knee folds back
    th = th - slot_k - hole((0, 0, HIP_Z), (1, 0, 0), 4 + TURN, 25, 25, up) \
            - hole((xc, 0, KNEE_Z), (1, 0, 0), 3 + TIGHT, 12, 12, up)
    add(f"thigh_{side}", th, "denim", up)
    # shin: rolled cuff with a tongue up into the knee, ankle tongue below
    cuff_top = KNEE_Z - RKN - KNUCKLE_GAP
    cuff = rbox((xc - 10.0, -12.0, 22.6), (xc + 10.0, 12.0, 27.4), 2.4) + \
           rbox((xc - 9.7, -11.6, 25.8), (xc + 9.7, 11.6, cuff_top), 2.2)
    cuff = cuff + cyl((xc - 2.8, 0, KNEE_Z), (xc + 2.8, 0, KNEE_Z), 4.5) + box((xc - 2.8, -4.5, cuff_top - 1), (xc + 2.8, 4.5, KNEE_Z))
    cuff = cuff + cyl((xc - 2.7, ANKLE_Y, ANKLE_Z), (xc + 2.7, ANKLE_Y, ANKLE_Z), 4.0) + \
           box((xc - 2.7, ANKLE_Y - 4.0, ANKLE_Z), (xc + 2.7, ANKLE_Y + 4.0, 23.5))
    kbevel = place(Manifold.cube((60, 20, 20), True), (xc, 11.6 + 7.07 - 3.6, cuff_top + 7.07 - 3.6), (0, 1, 1), up=(0, -1, 1))
    cuff = cuff - kbevel                                                          # lets the knee fold further
    cuff = cuff - hole((xc, 0, KNEE_Z), (1, 0, 0), 3 + TURN, 5, 5, up) \
                - hole((xc, ANKLE_Y, ANKLE_Z), (1, 0, 0), 3 + TURN, 5, 5, up)
    cuff = cuff ^ slab(0, xc - 9.3, xc + 9.3)
    add(f"shin_{side}", cuff, "denim", up)
    # shoe: black upper with an ankle fork, white toe cap + ankle discs, separate white sole
    xs = xc + s * 2.9
    def shoe_shape(d=0.0):
        return hull([sph((xs + x, y, z), 3.0 + d) for x, y, z in
                     [(-8.5, 8.5, 17.5), (8.5, 8.5, 17.5), (-8.5, 8.5, 8.4), (8.5, 8.5, 8.4),
                      (-8.5, -18.5, 8.4), (8.5, -18.5, 8.4), (-6.5, -17.0, 11.6), (6.5, -17.0, 11.6),
                      (-7.5, 1.5, 19.0), (7.5, 1.5, 19.0)]]) ^ slab(2, 5.4, 40)
    shoe = shoe_shape()
    slot = box((xc - 3.05, ANKLE_Y - 6.0, ANKLE_Z - 4.4), (xc + 3.05, ANKLE_Y + 7.0, 40))
    pinh = hole((xc, ANKLE_Y, ANKLE_Z), (1, 0, 0), 3 + TIGHT, 14, 14, (0, 0, 1))
    discs = union([cyl((xs + sx * 11.4, ANKLE_Y, ANKLE_Z), (xs + sx * 12.6, ANKLE_Y, ANKLE_Z), 3.7) -
                   cyl((xs + sx * 12.1, ANKLE_Y, ANKLE_Z), (xs + sx * 13.0, ANKLE_Y, ANKLE_Z), 1.9)
                   for sx in (-1, 1)])
    inlay = shoe_shape(0.5) - shoe_shape(-0.3)          # details sink 0.3 mm in: no coincident faces
    toe = inlay ^ slab(1, -40, -13.5) ^ slab(2, 5.4, 40)
    stripe = inlay ^ union([
        place(Manifold.cube((30, 2.2, 30), True), (xs, -5.0, 12.5), (0, 0, 1), up=(0, math.sin(0.6), math.cos(0.6))),
        place(Manifold.cube((30, 2.2, 30), True), (xs, -9.5, 12.5), (0, 0, 1), up=(0, math.sin(0.6), math.cos(0.6)))]) ^ \
        slab(2, 7.0, 40) ^ slab(1, -14, 0)
    shoe_white = union([toe, discs, stripe]) - slot - pinh
    add(f"shoe_{side}", shoe - slot - pinh - discs - toe - stripe, "black", (0, 0, 1),
        details={f"shoe_{side}_white": (shoe_white, "white")})
    sole = rbox((xs - 11.7, -21.8, 0.0), (xs + 11.7, 11.8, 5.4 + 2.0), 2.0) ^ slab(2, 0, 5.4)
    add(f"sole_{side}", sole, "white", (0, 0, 1), note="glue under the shoe")

# ================================================================= PINS (print lying, flat-sided)
def pin(d, L, ch=0.4):
    """Flat-sided pin with chamfered lead-ins, as one convex hull (prints lying flat)."""
    def prof(r):
        return CrossSection.circle(r, 48) ^ CrossSection.square((2 * d, d * 0.82)).translate((-d, -d * 0.41))
    core = prof(d / 2).extrude(L - 2 * ch).translate((0, 0, ch))
    ends = prof(d / 2 - ch).extrude(L)
    return place(hull([core, ends]), (0, 0, 0), (1, 0, 0), up=(0, 0, 1))      # lying along x, flats down
PINS = {  # name: (diameter, length, count)
    "pin_neck_4.5x15": (4.5, 15.0, 1), "pin_shoulder_4x17": (4.0, 17.0, 2), "pin_elbow_3x13": (3.0, 13.0, 2),
    "pin_wrist_3.5x12": (3.5, 12.0, 2), "pin_waist_5x13": (5.0, 13.0, 1), "pin_hip_4x37": (4.0, 37.0, 1),
    "pin_knee_3x16": (3.0, 16.0, 2), "pin_ankle_3x23": (3.0, 23.0, 2),
}
for n, (d, L, c) in PINS.items():
    add(n, pin(d, L), "white", (0, 0, 1), count=c)

# ================================================================= FIT TEST (print this first)
coupon = rbox((-22, -7, 0), (22, 7, 6), 1.0)
for i, (d, tol) in enumerate([(3.0, TIGHT), (3.0, TURN), (4.0, TIGHT), (4.0, TURN), (5.0, TIGHT), (5.0, TURN)]):
    coupon = coupon - Manifold.cylinder(10, (d + tol) / 2, (d + tol) / 2, 48).translate((-17.5 + i * 7, 0, -1))
add("fit_test_coupon", coupon, "white", (0, 0, 1),
    note="holes left to right: 3 tight, 3 turn, 4 tight, 4 turn, 5 tight, 5 turn; test with the matching pins")

# ================================================================= EXPORT
COLOURS = {"skin": (184, 116, 63), "black": (34, 34, 36), "white": (238, 238, 232), "denim": (111, 134, 181)}
def tri(m):
    mm = m.to_mesh()
    return trimesh.Trimesh(np.array(mm.vert_properties)[:, :3].astype(np.float64), np.array(mm.tri_verts), process=False)

def stl_safe(t):
    """STL stores loose triangles; coplanar booleans can leave sub-micron slivers that
    become non-manifold once a slicer merges vertices. Re-mesh those, checking volume."""
    import io, pymeshfix
    b = io.BytesIO(); t.export(b, file_type="stl"); b.seek(0)
    r = trimesh.load(b, file_type="stl")
    if r.is_watertight:
        return r
    v, f = pymeshfix.clean_from_arrays(np.ascontiguousarray(r.vertices, dtype=np.float64),
                                       np.ascontiguousarray(r.faces, dtype=np.int32),
                                       joincomp=True, remove_smallest_components=False)
    fixed = trimesh.Trimesh(v, f)
    if not (fixed.is_watertight and abs(fixed.volume - t.volume) / t.volume < 0.005):
        raise RuntimeError(f"repair changed the part: {t.volume:.1f} -> {fixed.volume:.1f}, bodies {r.body_count}, wt {fixed.is_watertight}")
    return fixed

def to_print(m, up):
    e1, e2, e3 = frame(up)
    R = np.vstack([e1, e2, e3])                      # up -> +z
    pm = m.transform(np.column_stack([R, np.zeros(3)]))
    lo, hi = np.array(pm.bounding_box()[:3]), np.array(pm.bounding_box()[3:])
    return pm.translate(tuple(-np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]]))), np.column_stack([R, np.zeros(3)])

os.makedirs(OUT, exist_ok=True); os.makedirs("multicolor", exist_ok=True)
for f in os.listdir(OUT): os.remove(os.path.join(OUT, f))
scene = trimesh.Scene()
report = []
for name, p in parts.items():
    merged = union([p["body"]] + [d for d, _ in p["details"].values()])
    pm, R = to_print(merged, p["up"])
    t = stl_safe(tri(pm))
    ext = t.extents
    report.append(dict(part=name, colour=p["colour"], qty=p["count"], size_mm=[round(float(e), 1) for e in ext],
                       watertight=bool(t.is_watertight), volume_cm3=round(t.volume / 1000, 2), note=p["note"]))
    t.export(f"{OUT}/{name}{'_x' + str(p['count']) if p['count'] > 1 else ''}.stl")
    if p["details"]:                                   # colour-split 3MF for CFS
        sc = trimesh.Scene()
        lo = np.array(merged.transform(R).bounding_box()[:3]); hi = np.array(merged.transform(R).bounding_box()[3:])
        sh = -np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2, lo[2]])
        for nm, (m, col) in [(name, (p["body"], p["colour"]))] + list(p["details"].items()):
            tm_ = tri(m.transform(R).translate(tuple(sh)))
            tm_.visual.face_colors = list(COLOURS[col]) + [255]
            sc.add_geometry(tm_, node_name=f"{nm}__{col}", geom_name=f"{nm}__{col}")
        sc.export(f"multicolor/{name}_multicolor.3mf")
    if name.startswith("pin_") or name == "eye_pin":
        continue
    for nm, (m, col) in [(name, (p["body"], p["colour"]))] + list(p["details"].items()):
        tm_ = tri(m); tm_.visual.face_colors = list(COLOURS[col]) + [255]
        scene.add_geometry(tm_, node_name=nm, geom_name=nm)
for p in eye_pts:
    e = tri(sph(p + np.array([0, -0.3, 0]), 1.55, 32)); e.visual.face_colors = list(COLOURS["black"]) + [255]
    scene.add_geometry(e)
scene.export("figure_assembled.glb")
json.dump(report, open(f"{OUT}/parts.json", "w"), indent=1)
for r in report:
    flag = "" if r["watertight"] and max(r["size_mm"][:2]) < 215 and r["size_mm"][2] < 245 else "  <-- CHECK"
    print(f'{r["part"]:22s} {r["colour"]:6s} x{r["qty"]}  {r["size_mm"]}  {r["volume_cm3"]} cm3{flag}')
tot = trimesh.util.concatenate([g for g in scene.geometry.values()])
print("assembled extents", np.round(tot.extents, 1), "parts", sum(r["qty"] for r in report))
