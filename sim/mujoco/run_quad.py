"""Quadruped gait driver.

Run from this directory (sim/mujoco):

    python run_quad.py --mode static             # drop + settle, check stance
    python run_quad.py --mode walk               # stabilized open-loop crawl
    python run_quad.py --mode walk --render      # with viewer

The `walk` mode layers a simple stance-adjustment loop (height + roll/pitch)
on an open-loop crawl foot trajectory. To keep the springs from absorbing the
commanded motion (planted feet resist position control), the stance sweep is
deliberately small; the output demonstrates attitude/height stabilization on
the compliant legs. Sustained forward trotting is expected from the learned
policy (PPO, roadmap W10-14), not from this hand-tuned planner.
"""

from __future__ import annotations

import argparse

import mujoco
import numpy as np

import leg_utils as leg
import quad_model

LEGS = ["FL", "FR", "RL", "RR"]
# Trot phase pairs: (FL,RR) and (FR,RL)
TROT_PHASE = {"FL": 0.0, "FR": 0.5, "RL": 0.5, "RR": 0.0}
# Crawl: one leg at a time (1/4 duty stagger)
CRAWL_PHASE = {"FL": 0.0, "FR": 0.25, "RL": 0.5, "RR": 0.75}

# Gait parameters
STANCE_DEPTH = 0.188   # hip-frame z (down) of foot at nominal stance (m)
STRIDE       = 0.08    # total stance sweep (m)
STEP_HEIGHT  = 0.03    # foot lift above stance during swing (m)
GAIT_FREQ    = 1.5     # steps per second per leg


def _foot_target(phase: float, t: float, freq: float = GAIT_FREQ, duty: float = 0.5):
    """Compute desired foot position in hip frame (x, z_down) for one leg.

    duty = fraction of the gait period spent in stance (3-4 feet on ground).
    """
    phi = (phase + freq * t) % 1.0   # normalized phase [0, 1)
    if phi < duty:  # stance: foot sweeps backward at ground level
        sweep = phi / duty
        x = STRIDE / 2 * (1 - 2 * sweep)
        z = STANCE_DEPTH
    else:  # swing: foot lifts and sweeps forward
        sweep = (phi - duty) / (1.0 - duty)
        x = -STRIDE / 2 + STRIDE * sweep
        z = STANCE_DEPTH - STEP_HEIGHT * np.sin(np.pi * sweep)
    return x, z


def mode_static(T: float):
    """Drop the robot, settle for T seconds, and check stance metrics."""
    model, data = quad_model.build_quad()
    data.qpos[2] += 0.05  # slight drop
    mujoco.mj_forward(model, data)

    n = int(T / model.opt.timestep)
    trunk = model.body("trunk").id

    for _ in range(n):
        mujoco.mj_step(model, data)

    # metrics
    z = data.xpos[trunk][2]
    ncon = data.ncon
    foot_contacts = {}
    for t in LEGS:
        pad_id = model.geom(f"{t}foot_pad").id
        foot_contacts[t] = any(
            data.contact[i].geom1 == pad_id or data.contact[i].geom2 == pad_id
            for i in range(data.ncon)
        )
    quat = data.sensor("trunk_quat").data.copy()
    # extract pitch from quaternion (small-angle approx)
    pitch = 2 * np.arctan2(quat[3], quat[0])  # ~2*angle

    print(f"== static stance (T={T:.1f}s) ==")
    print(f"  trunk height     {z:.4f} m")
    print(f"  pitch            {np.degrees(pitch):+.3f} deg")
    print(f"  foot contacts    {sum(foot_contacts.values())}/4  {foot_contacts}")
    print(f"  ncon             {ncon}")


def _euler_from_quat(q):
    """Roll (x), pitch (y), yaw (z) angles from wxyz quaternion."""
    w, a, b, c = q
    roll = np.arctan2(2 * (w * a + b * c), 1 - 2 * (a * a + b * b))
    pitch = np.arcsin(np.clip(2 * (w * b - c * a), -1, 1))
    yaw = np.arctan2(2 * (w * c + a * b), 1 - 2 * (b * b + c * c))
    return roll, pitch, yaw


def mode_walk(T: float, render: bool, gait: str = "trot", freq: float = GAIT_FREQ,
              duty: float = 0.5, stable: bool = True):
    """Open-loop gait with rotor PD tracking IK targets.

    With ``stable=True`` a simple stance-adjustment loop (attitude + height)
    is layered on top: each leg's commanded stance depth reacts to roll/pitch
    and trunk-height error so an open-loop gait can be sustained.
    """
    model, data = quad_model.build_quad()
    dt = model.opt.timestep
    n = int(T / dt)
    phase_map = TROT_PHASE if gait == "trot" else CRAWL_PHASE

    Kph, Krh = 0.05, 0.05   # pitch/roll feedback gain (m depth per rad lean)
    Kh = 0.4                # height-hold gain (m depth per m height error)
    z_ref = 0.212           # target trunk height (just above spring sag)

    # build per-leg actuator and joint maps
    act = {}
    jr = {}   # rotor qposadr
    jj = {}   # link qposadr
    for t in LEGS:
        act[t] = (int(model.actuator(f"{t}m1").id),
                   int(model.actuator(f"{t}m2").id))
        jr[t] = (int(model.jnt(f"{t}r1").qposadr[0]),
                  int(model.jnt(f"{t}r2").qposadr[0]))
        jj[t] = (int(model.jnt(f"{t}j1").qposadr[0]),
                  int(model.jnt(f"{t}j2").qposadr[0]))

    trunk_id = model.body("trunk").id
    Kp, Kd = 300.0, 25.0

    viewer = mujoco.viewer.launch_passive(model, data) if render else None

    # recordings
    t_arr = np.arange(n) * dt
    trunk_z = np.empty(n)
    trunk_pitch = np.empty(n)
    trunk_x = np.empty(n)
    spring_defl = np.empty((n, 8))  # per-leg, per-motor
    weld_err = np.empty(n)
    foot_contacts = np.zeros((n, 4), dtype=bool)

    tip_ids = [model.body(f"{t}tipC1").id for t in LEGS]
    tip_ids2 = [model.body(f"{t}tipC2").id for t in LEGS]
    pad_ids = [model.geom(f"{t}foot_pad").id for t in LEGS]

    for i in range(n):
        t = t_arr[i]
        if stable:
            roll, pitch, _ = _euler_from_quat(data.sensor("trunk_quat").data)
            z_err = z_ref - data.xpos[trunk_id][2]
        # compute desired rotor angles for each leg via IK
        for li, tag in enumerate(LEGS):
            px, pz = _foot_target(phase_map[tag], t, freq=freq, duty=duty)
            if stable:
                # roll/pitch/height stance adjustment
                dz = Kh * z_err
                if tag in ("FL", "FR"):
                    dz -= Kph * pitch          # nose-up: retract front
                else:
                    dz += Kph * pitch          #          extend rear
                if tag in ("FL", "RL"):
                    dz -= Krh * roll
                else:
                    dz += Krh * roll
                pz += dz
            th1, th2 = leg.ik(px, pz)
            if np.isnan(th1):
                th1, th2 = 0.0, 0.0  # unreachable → hold zero

            # rotor PD
            r1_idx, r2_idx = act[tag]
            qr1 = data.sensor(f"{tag}sp_r1").data[0]
            qr2 = data.sensor(f"{tag}sp_r2").data[0]
            qvr1 = data.sensor(f"{tag}sv_r1").data[0]
            qvr2 = data.sensor(f"{tag}sv_r2").data[0]
            data.ctrl[r1_idx] = -Kp * (qr1 - th1) - Kd * qvr1
            data.ctrl[r2_idx] = -Kp * (qr2 - th2) - Kd * qvr2

            # spring deflection
            qj1 = data.sensor(f"{tag}sp_j1").data[0]
            qj2 = data.sensor(f"{tag}sp_j2").data[0]
            spring_defl[i, li*2] = qr1 - qj1
            spring_defl[i, li*2+1] = qr2 - qj2

        mujoco.mj_step(model, data)

        trunk_z[i] = data.xpos[trunk_id][2]
        trunk_x[i] = data.xpos[trunk_id][0]
        _, trunk_pitch[i], _ = _euler_from_quat(data.sensor("trunk_quat").data)
        weld_err[i] = max(
            np.linalg.norm(data.xpos[tip_ids[j]] - data.xpos[tip_ids2[j]])
            for j in range(4)
        )
        for j, pad_id in enumerate(pad_ids):
            foot_contacts[i, j] = any(
                data.contact[k].geom1 == pad_id or data.contact[k].geom2 == pad_id
                for k in range(data.ncon)
            )

        if viewer is not None:
            viewer.sync()
    if viewer is not None:
        viewer.close()

    # summary metrics
    z0 = trunk_z[0]
    z_final = trunk_z[-1]
    x_final = trunk_x[-1]
    x_min = trunk_x.min()
    pitch_max = np.degrees(np.abs(trunk_pitch)).max()
    pitch_rms = np.degrees(np.sqrt(np.mean(trunk_pitch**2)))
    spring_rms = np.degrees(np.sqrt(np.mean(spring_defl**2, axis=0)))
    contact_rate = foot_contacts.mean(axis=0)

    print(f"== {gait} walk (T={T:.1f}s, freq={freq} Hz, stride={STRIDE} m, duty={duty}) ==")
    print(f"  trunk height      {trunk_z.mean():.4f} m  (start {z0:.4f}, end {z_final:.4f})")
    print(f"  x displacement    {x_final:+.3f} m  (range {x_min:.3f} to {trunk_x.max():.3f})")
    print(f"  pitch max         {pitch_max:.2f} deg   rms {pitch_rms:.2f} deg")
    print(f"  spring defl rms   {np.mean(spring_rms):.2f} deg  (per-leg per-motor: "
          f"{' '.join(f'{v:.1f}' for v in spring_rms)})")
    print(f"  weld loop max     {weld_err.max()*1e3:.1f} mm")
    print(f"  foot contact rate {' '.join(f'{t}:{c:.0%}' for t,c in zip(LEGS, contact_rate))}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", choices=["static", "walk"], default="static")
    p.add_argument("--T", type=float, default=3.0, help="duration (s)")
    p.add_argument("--gait", choices=["trot", "crawl"], default="crawl")
    p.add_argument("--freq", type=float, default=0.5, help="gait frequency (Hz)")
    p.add_argument("--duty", type=float, default=0.75, help="stance duty (0.5 trot, 0.75 crawl)")
    p.add_argument("--openloop", action="store_true", help="disable stance feedback")
    p.add_argument("--render", action="store_true")
    a = p.parse_args()

    if a.mode == "static":
        mode_static(a.T)
    else:
        mode_walk(a.T, a.render, a.gait, a.freq, a.duty, stable=not a.openloop)


if __name__ == "__main__":
    main()
