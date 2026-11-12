"""Tydzień 10: poznajemy model SO-101 w MuJoCo (stawy, siłowniki, chwytak, zasięg)."""

import argparse
from pathlib import Path

import mujoco
import numpy as np

import rl_journey

DEFAULT_SCENE = Path(rl_journey.__file__).parent / "assets" / "so101" / "scene.xml"

parser = argparse.ArgumentParser()
parser.add_argument("--scene", type=Path, default=DEFAULT_SCENE)
parser.add_argument("--samples", type=int, default=20_000)
args = parser.parse_args()

model = mujoco.MjModel.from_xml_path(str(args.scene))
data = mujoco.MjData(model)


def name(obj, i):
    return mujoco.mj_id2name(model, obj, i) or f"(bez nazwy {i})"


print(
    f"stawy: {model.njnt}, siłowniki: {model.nu}, ciała: {model.nbody}, punkty (site): {model.nsite}"
)

print("\nSTAWY (zakres w stopniach)")
for j in range(model.njnt):
    lo, hi = np.rad2deg(model.jnt_range[j])
    limited = "tak" if model.jnt_limited[j] else "nie"
    print(
        f"  {j}: {name(mujoco.mjtObj.mjOBJ_JOINT, j):16s} {lo:7.1f} … {hi:6.1f}   limit: {limited}"
    )

print("\nSIŁOWNIKI")
for a in range(model.nu):
    joint = name(mujoco.mjtObj.mjOBJ_JOINT, model.actuator_trnid[a, 0])
    kp = model.actuator_gainprm[a, 0]
    lo, hi = model.actuator_ctrlrange[a]
    f_lo, f_hi = model.actuator_forcerange[a]
    print(
        f"  {a}: {name(mujoco.mjtObj.mjOBJ_ACTUATOR, a):16s} → staw {joint:16s} "
        f"kp = {kp:7.1f}   sterowanie {lo:6.2f} … {hi:5.2f}   siła {f_lo:5.2f} … {f_hi:4.2f}"
    )

sites = [name(mujoco.mjtObj.mjOBJ_SITE, s) for s in range(model.nsite)]
bodies = [name(mujoco.mjtObj.mjOBJ_BODY, b) for b in range(model.nbody)]
print("\nPUNKTY (site):", sites)
print("CIAŁA:", bodies)

# Chwytak: punkt (site) z „grip” w nazwie, a jeśli go nie ma, ostatnie ciało w łańcuchu
grip_site = next((s for s, n in enumerate(sites) if "grip" in n.lower()), None)
if grip_site is not None:
    tip_label = f"site {sites[grip_site]}"

    def tip():
        return data.site_xpos[grip_site].copy()
else:
    tip_label = f"ciało {bodies[-1]}"

    def tip():
        return data.xpos[model.nbody - 1].copy()


# Poza zerowa: wszystkie kąty 0
mujoco.mj_forward(model, data)
p0 = tip()
print(f"\nchwytak = {tip_label}; przy wszystkich kątach 0: {np.round(p0 * 100, 1)} cm")

# Sprawdzian znanej pozy: obrót podstawy (staw 0) o 90° przesuwa chwytak po okręgu
pan = 0
anchor, axis = data.xanchor[pan].copy(), data.xaxis[pan].copy()


def height_radius(p):
    """Wysokość wzdłuż osi podstawy i odległość od tej osi."""
    r = p - anchor
    h = r @ axis
    return h, r - h * axis


h0, flat0 = height_radius(p0)
data.qpos[model.jnt_qposadr[pan]] = np.pi / 2
mujoco.mj_forward(model, data)
h90, flat90 = height_radius(tip())
r0, r90 = np.linalg.norm(flat0), np.linalg.norm(flat90)
print(
    f"\nobrót staw 0 ({name(mujoco.mjtObj.mjOBJ_JOINT, pan)}) o 90°, oś {np.round(axis, 2)}:"
)
print(f"  wysokość wzdłuż osi:  {h0 * 100:6.2f} → {h90 * 100:6.2f} cm")
print(f"  odległość od osi:     {r0 * 100:6.2f} → {r90 * 100:6.2f} cm")
if r0 > 1e-3:
    angle = np.rad2deg(np.arccos(np.clip(flat0 @ flat90 / (r0 * r90), -1, 1)))
    print(f"  kąt obrotu chwytaka:  {angle:6.1f}°")

# Zasięg: losowe pozy w zakresach stawów (bez chwytaka), mierzymy położenie chwytaka
rng = np.random.default_rng(0)
arm = [
    j
    for j in range(model.njnt)
    if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE
    and "grip" not in name(mujoco.mjtObj.mjOBJ_JOINT, j).lower()
]
heights, radii = [], []
for _ in range(args.samples):
    data.qpos[:] = 0
    for j in arm:
        lo, hi = model.jnt_range[j] if model.jnt_limited[j] else (-np.pi, np.pi)
        data.qpos[model.jnt_qposadr[j]] = rng.uniform(lo, hi)
    mujoco.mj_kinematics(model, data)
    h, flat = height_radius(tip())
    heights.append(h)
    radii.append(np.linalg.norm(flat))
heights, radii = np.array(heights), np.array(radii)
print(
    f"\nzasięg ({args.samples} losowych póz, stawy: {[name(mujoco.mjtObj.mjOBJ_JOINT, j) for j in arm]}):"
)
print(
    f"  odległość od osi podstawy: do {radii.max() * 100:.1f} cm (95% póz do {np.percentile(radii, 95) * 100:.1f} cm)"
)
print(
    f"  wysokość wzdłuż osi:       od {heights.min() * 100:.1f} do {heights.max() * 100:.1f} cm"
)
