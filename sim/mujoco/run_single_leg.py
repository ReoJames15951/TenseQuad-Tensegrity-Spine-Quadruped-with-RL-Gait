"""Single-leg five-bar + SEA validation driver.

Run from this directory (sim/mujoco):

    python run_single_leg.py --mode workspace   # reachable set + FK-vs-sim check
    python run_single_leg.py --mode bounce      # drop test on the springed leg
    python run_single_leg.py --mode bounce --k 20 --T 4   # spring stiffness sweep

--render opens a passive viewer during bounce (requires glfw, installed with mujoco).
"""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

import leg_utils as leg

HERE = Path(__file__).parent


def build(k_s: float = 40.0, d_s: float = 0.3, gantry: bool = True, weld_ms: float = 1.0):
    """Compile the model with spring stiffness/damping override and optional
    gantry root (z-slider) so the single leg performs a clean vertical bounce
    instead of tipping over. weld_ms sets the weld equality timeconstant."""
    xml = (HERE / "fivebar_leg.xml").read_text(encoding="utf-8")
    xml = xml.replace('stiffness="40"', f'stiffness="{k_s}"')
    xml = xml.replace('damping="0.3"', f'damping="{d_s}"')
    xml = xml.replace('solref="0.001 1"', f'solref="{weld_ms * 1e-3:g} 1"')
    if gantry:
        xml = xml.replace(
            '<freejoint name="root"/>',
            '<joint name="root" type="slide" axis="0 0 1" limited="true" '
            'range="-0.15 0.5" damping="0.1"/>',
        )
    return mujoco.MjModel.from_xml_string(xml)


def _setup(model, data):
    mujoco.mj_resetData(model, data)
    r1 = int(model.jnt("r1").qposadr[0])
    j1 = int(model.jnt("j1").qposadr[0])
    r2 = int(model.jnt("r2").qposadr[0])
    j2 = int(model.jnt("j2").qposadr[0])
    return r1, j1, r2, j2


# Elbow closure is baked into the coupler body frames (euler y = -/+0.2761),
# so FK joint angles must be written with these offsets in the model.
JC1_BAKE = 0.2761  # add to FK joint angle for jc1
JC2_BAKE = -0.2761  # add to FK joint angle for jc2


def mode_workspace(n: int = 61):
    model = build()
    data = mujoco.MjData(model)
    r1, j1, r2, j2 = _setup(model, data)

    m = leg.workspace_metrics(n)
    print("== reachable set (FK) ==")
    for k, v in m.items():
        print(f"  {k:<16} {v:>10.4f}" if isinstance(v, float) else f"  {k:<16} {v}")
    for d in (0.05, 0.08, 0.11, 0.14):
        r = leg.swing_at_depth(d)
        if np.isfinite(r[0]) and np.isfinite(r[1]):
            print(f"  swing at depth {d:5.3f}: x in [{r[0]:+.3f}, {r[1]:+.3f}] m")
        else:
            print(f"  swing at depth {d:5.3f}: outside reachable set")

    # FK-vs-sim cross-check: set input joints + analytically-closed elbows, then
    # verify the weld keeps both coupler tips coincident and matching the FK.
    th = np.linspace(-0.9, 0.9, 13)
    errs = []
    tip_gap = []
    for t1 in th:
        for t2 in th:
            _, _, f_fk, jc1, jc2 = leg.fk_full(t1, t2)
            if f_fk is None:
                continue
            data.qpos[r1] = t1
            data.qpos[j1] = t1
            data.qpos[r2] = t2
            data.qpos[j2] = t2
            data.qpos[int(model.jnt("jc1").qposadr[0])] = jc1 + JC1_BAKE
            data.qpos[int(model.jnt("jc2").qposadr[0])] = jc2 + JC2_BAKE
            mujoco.mj_forward(model, data)
            hip = data.body("hip").xpos
            fx, fz = data.site("s_foot").xpos[0] - hip[0], data.site("s_foot").xpos[2] - hip[2]
            errs.append(np.hypot(fx - f_fk[0], fz - f_fk[1]))
            tip_gap.append(
                np.linalg.norm(
                    data.xpos[model.body("tipC1").id] - data.xpos[model.body("tipC2").id]
                )
            )
    errs = np.array(errs)
    tip_gap = np.array(tip_gap)
    print("\n== FK vs MuJoCo loop closure ==")
    print(f"  samples      {errs.size}")
    print(f"  max foot err {errs.max() * 1e3:.3f} mm   mean {errs.mean() * 1e3:.3f} mm")
    print(f"  weld tip gap {tip_gap.max() * 1e3:.3f} mm (constraint slack at fk/forward)")


def mode_bounce(
    T: float,
    k_s: float,
    d_s: float,
    render: bool,
    drop: float,
    bend: float = 0.0,
    weld_ms: float = 1.0,
):
    model = build(k_s=k_s, d_s=d_s, gantry=True, weld_ms=weld_ms)
    data = mujoco.MjData(model)
    _, _, _, _ = _setup(model, data)
    mid1 = int(model.actuator("m1").id)
    mid2 = int(model.actuator("m2").id)
    r1 = int(model.jnt("r1").qposadr[0])
    r2 = int(model.jnt("r2").qposadr[0])
    j1 = int(model.jnt("j1").qposadr[0])
    j2 = int(model.jnt("j2").qposadr[0])
    base = model.body("base").id
    tip1 = model.body("tipC1").id
    tip2 = model.body("tipC2").id

    # start bowed: both proximal links at +bend keeps the mechanism away from
    # the straight-posture singularity so the springs see real vertical loads.
    data.qpos[r1] = data.qpos[j1] = bend
    data.qpos[r2] = data.qpos[j2] = bend
    _, _, _, jc1, jc2 = leg.fk_full(bend, bend)
    data.qpos[int(model.jnt("jc1").qposadr[0])] = jc1 + JC1_BAKE
    data.qpos[int(model.jnt("jc2").qposadr[0])] = jc2 + JC2_BAKE
    Kp, Kd = 300.0, 25.0  # rotor hold PD around +/-bend setpoint
    dt = model.opt.timestep
    n = int(T / dt)

    data.qpos[0] += drop  # extra drop height above the XML clearance
    mujoco.mj_forward(model, data)
    z0 = data.xpos[base][2]  # world base height at drop start

    viewer = mujoco.viewer.launch_passive(model, data) if render else None

    np.arange(n) * dt
    base_z = np.empty(n)
    spring = np.empty(n)
    j_angles = np.empty((n, 2))
    hold_err = np.empty(n)
    weld_err = np.empty(n)
    try:
        for i in range(n):
            qr1 = data.sensor("sp_r1").data[0]
            qr2 = data.sensor("sp_r2").data[0]
            qj1 = data.sensor("sp_j1").data[0]
            qj2 = data.sensor("sp_j2").data[0]
            qvr1 = data.sensor("sv_r1").data[0]
            qvr2 = data.sensor("sv_r2").data[0]
            data.ctrl[mid1] = -Kp * (qr1 - bend) - Kd * qvr1
            data.ctrl[mid2] = -Kp * (qr2 - bend) - Kd * qvr2
            mujoco.mj_step(model, data)
            base_z[i] = data.xpos[base][2]
            spring[i] = k_s * max(abs(qr1 - qj1), abs(qr2 - qj2))
            j_angles[i] = (qj1, qj2)
            hold_err[i] = max(abs(qr1 - bend), abs(qr2 - bend))
            weld_err[i] = np.linalg.norm(data.xpos[tip1] - data.xpos[tip2])
            if viewer is not None:
                viewer.sync()
    finally:
        if viewer is not None:
            viewer.close()

    # bounce frequency from stance minima spacing after the first rebound.
    z = base_z.copy()
    mins, lo = [], z.max()
    for i in range(len(z)):
        if z[i] < lo:
            lo = z[i]
        elif z[i] - lo > 0.5 * abs(z.min() - lo) and z[i] > lo:
            mins.append(i)
            lo = z[i]
    periods = np.diff(np.asarray(mins)) * dt if len(mins) > 2 else np.array([])
    bounce_f = 1.0 / periods.mean() if periods.size else np.nan

    rest = z[len(z) // 3 :]
    droop = rest.mean()
    z0 = z[0]

    print(
        f"== SEA bounce (k_s={k_s:g} N-m/rad, bend {bend:+.2f} rad, base {model.opt.timestep * 1e3:g} ms) =="
    )
    print(f"  peak spring tension     {abs(spring).max():6.2f} N")
    print(f"  mean link deflection    {np.degrees(abs(j_angles).mean()):6.2f} deg")
    print(f"  rotor hold error (max)  {np.degrees(hold_err.max()):6.2f} deg")
    print(f"  min base height         {z.min():6.3f} m")
    print(f"  settled base height     {droop:6.3f} m   (sag {(z0 - droop) * 1e3:6.1f} mm)")
    print(f"  bounce frequency        {bounce_f:6.2f} Hz   (est. from stance minima)")
    print(f"  weld loop error (max)   {weld_err.max() * 1e6:8.1f} um")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["workspace", "bounce"], default="bounce")
    p.add_argument("--n", type=int, default=61, help="workspace grid resolution")
    p.add_argument("--T", type=float, default=3.0, help="bounce duration (s)")
    p.add_argument("--k", type=float, default=40.0, help="spring stiffness N-m/rad")
    p.add_argument("--d", type=float, default=0.3, help="spring damping N-m-s/rad")
    p.add_argument("--drop", type=float, default=0.0, help="extra drop height (m)")
    p.add_argument(
        "--bend", type=float, default=0.3, help="rotor-hold setpoint (rad), bows the leg"
    )
    p.add_argument("--weld-ms", type=float, default=1.0, help="weld equality timeconstant (ms)")
    p.add_argument("--render", action="store_true")
    a = p.parse_args()

    if a.mode == "workspace":
        mode_workspace(a.n)
    else:
        mode_bounce(a.T, a.k, a.d, a.render, a.drop, a.bend, a.weld_ms)


if __name__ == "__main__":
    main()
