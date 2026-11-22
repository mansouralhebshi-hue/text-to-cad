"""Gem Bag: faceted diamond handbag sculpture, 190 mm wide, for a Creality K1C.

The silhouette and facet seams are traced from the back view. The thickness comes
from the side views: 66 mm at the girdle, tapering to a 10 mm handle and a narrow
bottom point. The bag is mirror-symmetric front to back, so it is split on its
mid-plane into two identical flat-backed halves. They print face-up with no
supports and glue together on three Ø5 alignment pins.

Run: python model.py -> gem_bag_half.stl (print x2), gem_bag_full.stl (preview), pins
"""
import numpy as np
import triangle
import trimesh
from manifold3d import Manifold, Mesh

WIDTH = 190.0
S = WIDTH / 760.0                       # reference px -> mm
def P(px, py):
    return ((px - 1335) * S, (870 - py) * S)

# --- outline (px from the back view) and its half-thickness h (mm) ---
outer = [("OTL", 1240, 105, 5.0), ("GL", 955, 455, 4.0), ("BP", 1335, 870, 2.0),
         ("GR", 1715, 455, 4.0), ("OTR", 1430, 105, 5.0)]
hole = [("TL", 1268, 130, 5.0), ("L", 1188, 243, 9.0), ("A", 1240, 350, 22.0),
        ("B", 1430, 350, 22.0), ("R", 1482, 243, 9.0), ("TR", 1402, 130, 5.0)]
inner = [("E", 1180, 452, 31.0), ("C", 1490, 452, 31.0), ("D", 1335, 700, 15.0),
         ("PC", 1335, 470, 33.0)]          # pentagon centre: 66 mm thick at the girdle
pts = outer + hole + inner
idx = {n: i for i, (n, *_ ) in enumerate(pts)}
V2 = np.array([P(px, py) for _, px, py, _ in pts])
H = np.array([h for *_, h in pts])

ring = lambda names: [(idx[a], idx[b]) for a, b in zip(names, names[1:] + names[:1])]
segs = ring([n for n, *_ in outer]) + ring([n for n, *_ in hole])
# facet seams: pentagon A-B-C-D-E, girdle seams, centre-bottom seam
segs += [(idx["B"], idx["C"]), (idx["C"], idx["D"]), (idx["D"], idx["E"]), (idx["E"], idx["A"]),
         (idx["E"], idx["GL"]), (idx["C"], idx["GR"]), (idx["D"], idx["BP"])]
holept = np.mean([P(px, py) for _, px, py, _ in hole], axis=0)
T = triangle.triangulate(dict(vertices=V2, segments=np.array(segs), holes=[holept]), "p")
assert len(T["vertices"]) == len(V2), "triangulator added points"
tri = T["triangles"]

# --- closed half solid: faceted face at z = h, flat back at z = 0 ---
n = len(V2)
V = np.vstack([np.column_stack([V2, H]), np.column_stack([V2, np.zeros(n)])])
F = [tri, tri[:, ::-1] + n]
edges = {}
for t in tri:                                   # boundary edges appear once
    for a, b in ((t[0], t[1]), (t[1], t[2]), (t[2], t[0])):
        edges[(a, b)] = edges.get((a, b), 0) + 1
for (a, b), _ in edges.items():
    if (b, a) not in edges:                     # boundary edge, oriented as in the top face
        F.append(np.array([[b, a, a + n], [b, a + n, b + n]]))
F = np.vstack(F)
m = Manifold(Mesh(vert_properties=V.astype(np.float32), tri_verts=F.astype(np.uint32)))
if m.volume() < 0:
    m = Manifold(Mesh(vert_properties=V.astype(np.float32), tri_verts=F[:, ::-1].astype(np.uint32)))
assert not m.is_empty() and m.volume() > 0, m.status()

solid = m
# alignment pin holes on the mid-plane (Ø5 pins, 5 mm deep each side)
PIN_R, PIN_D = 2.65, 5.5
for px, py in ((1335, 470), (1080, 470), (1590, 470), (1335, 640)):
    x, y = P(px, py)
    m = m - Manifold.cylinder(PIN_D, PIN_R, PIN_R, 48).translate((x, y, -0.01))

def tm(mf):
    mm = mf.to_mesh()
    return trimesh.Trimesh(np.array(mm.vert_properties)[:, :3], np.array(mm.tri_verts), process=False)

half = tm(m)
half.apply_translation(-half.bounds[0])
half.export("gem_bag_half.stl")
full = tm(solid + solid.mirror((0, 0, 1)))
full.export("gem_bag_full.stl")
tm(Manifold.cylinder(10.0, 2.45, 2.45, 32)).export("gem_bag_pin_5x10_print_x4.stl")
print("half", np.round(half.extents, 1), "watertight", half.is_watertight, "vol cm3", round(half.volume / 1000, 1))
print("full", np.round(full.extents, 1), "watertight", full.is_watertight)
