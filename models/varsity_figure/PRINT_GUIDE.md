# Varsity Figure — print & assembly guide (Creality K1C)

160 mm articulated figure · 34 parts · 4 plates (one per colour) · ~100 g PLA

## 0. Print the fit test first (10 min)
`fit_test_coupon` + one 3 mm, 4 mm and 5 mm pin.
- **TIGHT** holes: pin should press in firmly and not turn.
- **TURN** holes: pin should rotate with smooth friction (holds a pose).
Too loose → set `TURN = 0.25` in `model.py`; too tight → `0.35`. Re-run `python model.py`.

## 1. Plates (all fit one K1C bed each)
| Plate | Filament | Parts |
|---|---|---|
| `plate_black_1.3mf` | Matte black PLA | torso, hair, upper/lower arms, shoes, eye pins |
| `plate_denim_1.3mf` | Denim / pastel blue PLA | hips, thighs, shins |
| `plate_skin_1.3mf` | Tan / brown PLA | head, hands |
| `plate_white_1.3mf` | White PLA | soles, all joint pins, fit-test coupon |

Every part is already oriented: pin axes print vertically (perfect circles), any sideways hole is a teardrop (no support needed).

## 2. Settings (Creality Print / OrcaSlicer, K1C 0.4 nozzle)
- Layer **0.12 mm** (0.16 mm acceptable) · 3 walls · 20 % gyroid · seam: rear/aligned
- Supports **only for `head` and `hair`**: tree supports, build-plate + model, 50° threshold
  - `head` stands on its neck; `hair` prints crown-down
- Pins: 100 % infill, 4 walls, print slowly (outer wall 60 mm/s) — they are the skeleton
- Brim 3 mm on the thighs, shins and pins

## 3. Details (choose one)
- **Paint (single-colour printer):** white paint pen (Posca PC-1M) on the raised D letter, shirt strip, buttons,
  pocket welts, collar, hem stripes (inlaid flush), shoe toe caps and stripes; black for the beard (raised).
- **Creality CFS multicolour:** load `multicolor/torso_multicolor.3mf`, `head_multicolor.3mf`,
  `shoe_L/R_multicolor.3mf` — each is one object with the detail bodies split; assign filaments per body.

## 4. Assembly (pins: TIGHT end first, a drop of CA glue on that end only)
| Joint | Pin | Glue/press into | Turns in |
|---|---|---|---|
| Neck swivel | 4.5×15 | torso | head |
| Shoulders | 4×17 (×2) | torso sides | upper arm |
| Elbows | 3×13 (×2) | through upper-arm knuckle | forearm tongue |
| Wrists | 3.5×12 (×2) | forearm | hand |
| Waist swivel | 5×13 | hips disc | torso |
| Hips | 4×37 (one axle) | hips plate (centre) | both thighs |
| Knees | 3×16 (×2) | through thigh knuckle | shin tongue |
| Ankles | 3×23 (×2) | through shoe | shin tongue |

Order: soles → shoes; shins into shoes (ankle pins); shins into thighs (knee pins);
thighs + hips on the hip axle; hips disc into torso (waist pin); arms; hands; neck pin + head; glue hair cap on
(align the parting rows with the forehead); press eye pins in.

## 5. Range of motion (verified by collision sweep in CAD)
| Joint | Range |
|---|---|
| Neck, waist, wrists | 360° swivel |
| Shoulders | 360° |
| Elbows | 105° forward bend |
| Knees | 95° backward bend |
| Hips | +55° / −50° |
| Ankles | ±20–25° tilt |

Tip: if a joint is too loose after use, a coat of clear nail varnish on the pin restores friction.
