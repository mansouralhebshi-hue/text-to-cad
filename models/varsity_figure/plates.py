"""Pack parts/*.stl onto 220 x 220 K1C plates, one filament colour per plate."""
import json, re, glob, os
import numpy as np, trimesh
BED, GAP = 205.0, 5.0
rep = {r["part"]: r for r in json.load(open("parts/parts.json"))}
os.makedirs("plates", exist_ok=True)
for f in glob.glob("plates/*"): os.remove(f)
by_col = {}
for path in sorted(glob.glob("parts/*.stl")):
    base = os.path.basename(path)[:-4]
    name = re.sub(r"_x\d+$", "", base)
    r = rep[name]
    for k in range(r["qty"]):
        by_col.setdefault(r["colour"], []).append((name, trimesh.load(path)))
summary = {}
for col, items in by_col.items():
    items.sort(key=lambda t: -t[1].extents[1])
    plates, sc, x, y, row_h = [], trimesh.Scene(), 0.0, 0.0, 0.0
    for i, (name, m) in enumerate(items):
        w, h = m.extents[0], m.extents[1]
        if x + w > BED:
            x, y, row_h = 0.0, y + row_h + GAP, 0.0
        if y + h > BED:
            plates.append(sc); sc, x, y, row_h = trimesh.Scene(), 0.0, 0.0, 0.0
        mm = m.copy(); mm.apply_translation([x - m.bounds[0][0] + 7.5, y - m.bounds[0][1] + 7.5, -m.bounds[0][2]])
        sc.add_geometry(mm, node_name=f"{name}_{i}")
        x += w + GAP; row_h = max(row_h, h)
    plates.append(sc)
    for j, p in enumerate(plates):
        fn = f"plates/plate_{col}_{j + 1}.3mf"; p.export(fn)
        summary[fn] = len(p.geometry)
print(summary)
