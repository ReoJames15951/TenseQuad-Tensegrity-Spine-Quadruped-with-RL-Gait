"""Gymnasium-free quadruped RL environment over the MuJoCo SEA quad model.

Policy clock: 50 Hz.  Each policy step runs 10 PD substeps at 500 Hz (the
model timestep).  The robot is proprioceptive-only (no motion capture, no
feet-on-ground from MuJoCo — contacts are derived from foot site height).

Action space is a low-level *gait template* (8 dims, see ``_decode_action``):
stance torque blend on each rotor axis, stride/lift amplitudes, stance depth,
diagonal-pair phase, and cadence.  This keeps the pain threshold low — a
planted SEA foot cannot be position-servoed (springs saturate), so stance
torque is what actually propels; the policy tunes where the robot lives on
that manifold.

Reward drives forward +x speed, upright attitude, commanded height, with
action-magnitude/rate penalties.  Episodes terminate on crash or reorient.

Domain randomization (per env, per episode, reset-scoped): floor friction,
trunk mass, spring stiffness/damping, motor-gain scale, control delay,
action and observation noise, small initial drop/tilt/jitter.
"""

from __future__ import annotations

import json
import pathlib

import mujoco
import numpy as np

import leg_utils as leg
import quad_model

_HERE = pathlib.Path(__file__).resolve().parent  # module resource dir (sea_center.json)

LEGS = ["FL", "FR", "RL", "RR"]
N_SUB = 10  # 50 Hz policy / 500 Hz PD
PD_KP, PD_KD = 300.0, 25.0
KP_ST = 300.0  # firm stance-axis stiffness (verified-stable hold)
KP_LD = 300.0  # firm load-axis stiffness (height)
TAU1, TAU2 = 1.6, 1.0  # stance torque magnitude (Nm) on m1/m2 when blended
DUTY = 0.80  # stance fraction per leg
G_FREQ_LO, G_FREQ_HI = 0.40, 0.80
# Diagonal-trot phase: opposite legs swing together (FL+RR, then FR+RL).
# The sequential 0/0.25/0.5/0.75 crawl winds the feet in a rotary pattern
# around the COM, so the robot circles instead of marching; the diagonal
# (trot) phase cancels that yaw and walks straight.
CRAWL_PHASE = {"FL": 0.00, "FR": 0.50, "RL": 0.50, "RR": 0.00}
# Sequential-crawl order (opposite rotary senses) used by the phase-bias turn.
_FORWARD_CRAWL = {"FL": 0.00, "FR": 0.25, "RL": 0.50, "RR": 0.75}
_REVERSE_CRAWL = {"FL": 0.00, "FR": 0.75, "RL": 0.50, "RR": 0.25}
TERM_Z_LO, TERM_Z_HI = 0.15, 0.35
TERM_TILT = 0.6
MAX_STEPS = 500  # 10 s episode

OBS_DIM = 57  # base 52 + 4 x per-leg hip-yaw joint position + world heading

# Active hip-yaw straightening servo: the four symmetric sagittal legs give
# no yaw authority, so tripod reactions compound into a steady turn; these
# small yaw motors counter that yaw error PD-style to keep the heading.
# A differential base on left vs right is the open-loop turning input:
# |turn| > 0 biases the yaw setpoints oppositely and produces real (if
# gait-modulated) heading rates -- the reliably commandable turn mechanism.
KP_YAW, KD_YAW = 60.0, 2.0

# Anti-yaw reward weights (litmus for reward-exploitation vs mechanical
# coupling): heading error (world yaw vs the commanded heading, squared)
# and instantaneous yaw rate. The heading command is the turning input --
# the policy learned world-yaw control to zero out this penalty.
RYAW_A, RYAW_G = 1.0, 0.15


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


# stance-torque blend ceiling for u0 (mapped by stride gate and clutch).
# 0.30 is the position-PD-dominant regime (stable, backward-biased); raising
# it empowers the +x propulsor, safe once the one-way clutch shapes it.
_LAM1_SCALE = 0.30


def _euler_from_quat(q):
    w, a, b, c = q
    roll = np.arctan2(2 * (w * a + b * c), 1 - 2 * (a * a + b * b))
    pitch = np.arcsin(np.clip(2 * (w * b - c * a), -1, 1))
    yaw = np.arctan2(2 * (w * c + a * b), 1 - 2 * (b * b + c * c))
    return roll, pitch, yaw


def _decode_action(u, stride_max=0.05, stride_min=0.0, gait_dir=1.0):
    """Map the 8-dim action in [-1,1] to gait parameters (residual form).

    Each parameter is centered on an expert prior (the hand-validated crawl
    that already walks in place steadily) and the action is a bounded
    residual the policy learns on top of it::

        param = clip(base + scale*u, lo, hi)

    stride_max caps the stroke length (curriculum knob: 0 = standing balance
    only).  u0: stance-torque blend on the knee axis (0 = rigid position-PD
    hold, 1 = pure +x stance torque; the torque is the forward propulsor, the
    PD was winning at blend<=0.3).  u2/u3: stride and
    foot-lift amplitude residuals.  u4: stance depth bias (+-0.02 m, height).
    u5: front(+)/rear(-) torque balance (pitch authority).
    u6: left(+)/right(-) torque balance (roll authority).
    u7: cadence residual (0.4..0.8 Hz).

    Legs crawl on staggered phases (CRAWL_PHASE) so >=3 feet stay down and
    the robot is statically stable at every instant.  Returns lam1, lam2,
    stride, lift, depth, freq, and per-leg m1 torque scales (4,).
    """
    stride = float(np.clip(0.05 + 0.07 * u[2], stride_min, stride_max))
    # stance torque is gated by stride fraction: at stride 0 (balance
    # curriculum) the torque is inert, keeping the position-PD hold safe;
    # it blends in as the walk builds (tested ceiling: blend*0.30 keeps the
    # position-PD regime stable; full blend destabilizes the stance knee --
    # the +x propulsor exists, but its strong regime is not a stable attractor).
    lam_gate = (stride / stride_max) if stride_max > 0.0 else 0.0
    lam1 = float(np.clip(u[0] * _LAM1_SCALE, 0.0, 1.0) * lam_gate)
    lam2 = 0.0  # load-axis stance torque drives the robot BACKWARD; disabled
    lift = float(np.clip(0.02 + 0.06 * np.clip(u[3], 0, 1), 0.0, 0.08))
    depth = float(0.188 + np.clip(u[4], -1, 1) * 0.02)
    fb = float(np.clip(u[5], -1, 1))  # +front -rear
    lr = float(np.clip(u[6], -1, 1))  # +left -right
    freq = float(G_FREQ_LO + np.clip(0.5 + u[7], 0, 1) * (G_FREQ_HI - G_FREQ_LO))
    scale = {}
    for t in LEGS:
        front = 1.0 + 0.25 * fb if t in ("FL", "FR") else 1.0 - 0.25 * fb
        left = 1.0 + 0.25 * lr if t in ("FL", "RL") else 1.0 - 0.25 * lr
        scale[t] = float(np.clip(0.5 * (front + left), 0.1, 1.9))
    return lam1, lam2, stride, lift, depth, freq, scale


class QuadGaitEnv:
    def __init__(
        self,
        seed=0,
        dr=True,
        stride_max=0.05,
        speed_target=0.15,
        yaw0=0.0,
        gait_dir=-1.0,
        stride_min=0.0,
        heading_cmd=0.0,
        turn=0.0,
    ):
        self._rng = np.random.default_rng(seed)
        self._dr_on = dr
        self.stride_max = float(stride_max)
        self.speed_target = float(speed_target)
        self.stride_min = float(stride_min)
        self.heading_cmd = float(heading_cmd)
        self.turn = float(turn)
        self._yaw0 = float(yaw0)
        self._gait_dir = gait_dir
        self._clutch = True
        self.model, self.data = quad_model.build_quad()
        self.dt = self.model.opt.timestep
        self._cache_ids()
        self._dr = {}
        self._steps = 0
        self._u_prev = np.zeros(8)
        self._clock = 0.0
        self.reset()

    # ---- model/name caches -------------------------------------------------
    def _cache_ids(self):
        m = self.model
        self.trunk_id = m.body("trunk").id
        self.floor_id = int(m.geom("floor").id)
        self.act = {t: (int(m.actuator(f"{t}m1").id), int(m.actuator(f"{t}m2").id)) for t in LEGS}
        self.act_yaw = {t: int(m.actuator(f"{t}yaw").id) for t in LEGS}
        self.jnt_yaw = {
            t: (int(m.jnt(f"{t}yaw").qposadr[0]), int(m.jnt(f"{t}yaw").dofadr[0])) for t in LEGS
        }
        self.tend_id = {
            spring: int(m.tendon(f"{t}{spring}").id)
            for t in LEGS
            for spring in ("spring1", "spring2")
        }

    # ---- reset -------------------------------------------------------------
    def _draw_dr(self):
        rng = self._rng
        if not self._dr_on:
            return {
                "mu_gain": 1.0,
                "k_s": 1.0,
                "d_s": 1.0,
                "friction": 1.0,
                "mass": 1.0,
                "delay": 0,
                "act_noise": 0.0,
                "obs_noise": 0.0,
                "drop": 0.0,
                "tilt": np.zeros(2),
                "jitter": np.zeros(8),
            }
        return {
            "mu_gain": rng.uniform(0.9, 1.1),  # motor gain
            "k_s": rng.uniform(0.8, 1.2),  # spring stiffness scale
            "d_s": rng.uniform(0.2, 0.45),  # spring damping scale
            "friction": rng.uniform(0.5, 1.3),  # floor friction
            "mass": rng.uniform(0.9, 1.1),  # trunk mass scale
            "delay": int(rng.integers(0, 3)),  # control delay (policy steps)
            "act_noise": rng.uniform(0.0, 0.04),  # action noise
            "obs_noise": rng.uniform(0.0, 0.02),  # obs noise (std, normalized)
            "drop": rng.uniform(0.0, 0.03),  # extra initial height
            "tilt": rng.uniform(-0.05, 0.05, size=2),  # init roll/pitch
            "jitter": rng.uniform(-0.01, 0.01, size=8),
        }

    # SEA (kappa_s, d_s) DR center: measured bench pair (sec. 11 A1/B1 rows) via
    # sea_center.json, CAD 40.0/0.3 fallback. Never re-seed DR with CAD numbers.
    _SEA_CENTER = None  # (k_c, d_c) tuple, resolved once on first use

    @classmethod
    def _sea_center(cls):
        if cls._SEA_CENTER is not None:
            return cls._SEA_CENTER
        p = _HERE / "sea_center.json"
        try:
            j = json.loads(p.read_text(encoding="utf-8"))
            k, d = float(j["k_s"]), float(j["d_s"])
            if not (k > 0 and d > 0):
                raise ValueError("sea_center.json needs positive k_s, d_s")
        except (OSError, KeyError, ValueError):
            k, d = 40.0, 0.3  # CAD fallback
        cls._SEA_CENTER = (k, d)
        return cls._SEA_CENTER

    def _apply_dr(self):
        m, _d = self.model, self.data
        k = self._dr
        k_c, d_c = self._sea_center()  # measured SEA pair (sec. 11)
        m.geom_friction[self.floor_id, 0] = k["friction"]
        m.body(self.trunk_id).mass = 2.0 * k["mass"]
        for tid in self.tend_id.values():
            m.tendon_stiffness[tid] = k_c * k["k_s"]
            m.tendon_damping[tid] = d_c * k["d_s"]

    def reset(self):
        self._dr = self._draw_dr()
        self._apply_dr()
        m, d = self.model, self.data
        mujoco.mj_resetData(m, d)
        roll, pitch = self._dr.get("tilt", (0.0, 0.0))
        # compose the reset tilt (roll,pitch) with the fixed starting yaw:
        # q = q_yaw * q_tilt, applied in the air about the world z axis.
        cr, sr, cp, sp = np.cos(roll / 2), np.sin(roll / 2), np.cos(pitch / 2), np.sin(pitch / 2)
        cy, sy = np.cos(self._yaw0 / 2), np.sin(self._yaw0 / 2)
        # q_tilt = (cp*cr, -sp*sr, sp*cr, cp*sr) in (w,x,y,z)
        w1, x1, y1, z1 = cp * cr, -sp * sr, sp * cr, cp * sr
        w2, _x2, y2, _z2 = cy, 0.0, 0.0, sy
        qw = w2 * w1 - y2 * y1  # z-axis only: x=z=0 parts
        qx = w2 * x1 + y2 * z1
        qy = w2 * y1 - y2 * x1
        qz = w2 * z1 + y2 * w1
        d.qpos[0:3] = (0.0, 0.0, 0.226 + self._dr.get("drop", 0.0))
        d.qpos[3:7] = (qw, qx, qy, qz)
        for i, t in enumerate(LEGS):
            for joint in ("r1", "r2"):
                d.qpos[int(m.jnt(f"{t}{joint}").qposadr[0])] = self._dr.get("jitter", np.zeros(8))[
                    i * 2 + (joint == "r2")
                ]
        mujoco.mj_forward(m, d)
        self._steps = 0
        self._u_prev = np.zeros(8)
        self._u_hist = [np.zeros(8)] * (self._dr.get("delay", 0) + 1)
        self._clock = 0.0
        self._ep_vx_sum = 0.0
        self._ep_z_sum = 0.0
        self._z0 = d.xpos[self.trunk_id][2]
        self._xprev = d.xpos[self.trunk_id][0]
        return self._get_obs()

    # ---- observation ---------------------------------------------------------
    def _get_obs(self):
        d = self.data
        o = []
        roll, pitch, _ = _euler_from_quat(d.sensor("trunk_quat").data)
        o += [roll, pitch]
        o += list(d.sensor("gyro").data)
        o.append(d.xpos[self.trunk_id][2])
        for t in LEGS:
            o.append(d.sensor(f"{t}sp_r1").data[0])
            o.append(d.sensor(f"{t}sp_j1").data[0])
            o.append(d.sensor(f"{t}sp_r2").data[0])
            o.append(d.sensor(f"{t}sp_j2").data[0])
            o.append(d.sensor(f"{t}sv_r1").data[0])
            o.append(d.sensor(f"{t}sv_r2").data[0])
            o.append(1.0 if d.sensor(f"{t}foot_pos").data[2] < 0.02 else 0.0)
        for t in LEGS:
            o.append(d.sensor(f"{t}sp_r1").data[0] - d.sensor(f"{t}sp_j1").data[0])
            o.append(d.sensor(f"{t}sp_r2").data[0] - d.sensor(f"{t}sp_j2").data[0])
        for t in LEGS:
            o.append(d.sensor(f"{t}sp_yaw").data[0])
        o.append((_euler_from_quat(d.sensor("trunk_quat").data)[2] + np.pi) % (2 * np.pi) - np.pi)
        o += list(self._u_prev)
        o += [np.sin(2 * np.pi * self._clock), np.cos(2 * np.pi * self._clock)]
        obs = np.asarray(o, dtype=np.float64)
        n = self._dr.get("obs_noise", 0.0)
        if n > 0.0:
            obs = obs + self._rng.normal(0.0, n, size=obs.shape) * np.maximum(np.abs(obs), 1.0)
        return obs

    # ---- step ----------------------------------------------------------------
    def step(self, action):
        m, d = self.model, self.data
        action = np.clip(np.asarray(action, dtype=np.float64), -1.0, 1.0)
        act_noise = self._dr.get("act_noise", 0.0)
        gain = self._dr.get("mu_gain", 1.0)

        # applied command lags by `delay` policy steps (zero-order hold per step)
        u = self._u_hist[0]
        if act_noise > 0.0:
            u = u + self._rng.normal(0.0, act_noise, size=8)
        lam1, lam2, stride, lift, depth, freq, tscale = _decode_action(
            u, self.stride_max, self.stride_min
        )
        gd = self._gait_dir

        # per-leg foot targets for this policy step (diagonal trot).
        # gait_dir = -1 (default, ratchet-forward): stance sweeps the foot
        # forward (+x) so the toothed pad rides free while the body is
        # propelled +x; +1 sweeps backward (stock crawl, backward cruise).
        # The pad cleat must face the same side as gait_dir points.
        # Phase-bias turn: sweep the per-leg phase from the diagonal trot
        # (yaw-cancelling) toward one of the two sequential crawls, which
        # orbit the feet in opposite rotary senses -- a strong, phase-locked
        # heading input keyed to the direction of self.turn.
        tgt = {}
        for t in LEGS:
            if self.turn > 0.0:
                ph = (1.0 - np.clip(self.turn, 0, 1)) * CRAWL_PHASE[t] + np.clip(
                    self.turn, 0, 1
                ) * _FORWARD_CRAWL[t]
            elif self.turn < 0.0:
                ph = (1.0 - np.clip(-self.turn, 0, 1)) * CRAWL_PHASE[t] + np.clip(
                    -self.turn, 0, 1
                ) * _REVERSE_CRAWL[t]
            else:
                ph = CRAWL_PHASE[t]
            phi = (ph + freq * self._clock) % 1.0
            stance = phi < DUTY
            s = (phi / DUTY) if stance else (phi - DUTY) / (1.0 - DUTY)
            x = (gd * stride / 2 * (1 - 2 * s)) if stance else (-gd * stride / 2 + gd * stride * s)
            z = depth if stance else depth - lift * np.sin(np.pi * s)
            th1, th2 = leg.ik(x, z)
            if np.isnan(th1):
                th1, th2 = 0.0, 0.0
            # one-way thrust clutch: the stance torque (the +x propulsor)
            # engages the front half of the sweep (foot ahead of hip, s<0.5)
            # and eases linearly to zero behind the hip -- the forward-recoil
            # shape of a directional spring.  Load-bearing PD is untouched.
            clutch = 1.0
            if self._clutch:
                clutch = 1.0 if s < 0.5 else max(0.0, 1.0 - (s - 0.5) / 0.5)
            tgt[t] = (th1, th2, stance, tscale[t], clutch)

        # PD substeps at the model rate
        for _ in range(N_SUB):
            for t in LEGS:
                th1, th2, stance, ts, clutch = tgt[t]
                r1i, r2i = self.act[t]
                qr1 = d.sensor(f"{t}sp_r1").data[0]
                qr2 = d.sensor(f"{t}sp_r2").data[0]
                qv1 = d.sensor(f"{t}sv_r1").data[0]
                qv2 = d.sensor(f"{t}sv_r2").data[0]
                if stance:
                    d.ctrl[r1i] = (1 - lam1) * (
                        -KP_ST * gain * (qr1 - th1) - PD_KD * qv1
                    ) + lam1 * TAU1 * ts * clutch
                    d.ctrl[r2i] = (1 - lam2) * (
                        -KP_LD * gain * (qr2 - th2) - PD_KD * qv2
                    ) + lam2 * TAU2
                else:
                    d.ctrl[r1i] = -PD_KP * gain * (qr1 - th1) - PD_KD * qv1
                    d.ctrl[r2i] = -PD_KP * gain * (qr2 - th2) - PD_KD * qv2
                # hip-yaw straightening servo: torque the yaw joint toward 0
                yi, di = self.jnt_yaw[t]
                d.ctrl[self.act_yaw[t]] = -KP_YAW * d.qpos[yi] - KD_YAW * d.qvel[di]
            mujoco.mj_step(m, d)

        self._steps += 1
        self._clock += self.dt * N_SUB

        # state
        x = d.xpos[self.trunk_id][0]
        z = d.xpos[self.trunk_id][2]
        vx = (x - self._xprev) / (self.dt * N_SUB)
        self._xprev = x
        roll, pitch, _ = _euler_from_quat(d.sensor("trunk_quat").data)

        self._ep_vx_sum += vx
        self._ep_z_sum += z

        # reward: modest speed target (endurance crawl); the velocity bonus
        # only counts while the trunk is truly flat (within ~0.12 rad), and
        # rocking angular velocity is penalized directly.  Surging in a
        # near-fall earns no progress bonus, so the policy cannot trade
        # stability for an instant speed payout.
        upright = (abs(roll) < 0.12) and (abs(pitch) < 0.12)
        # asymmetric speed reward: backward cruise is steeply penalized so the
        # policy cannot settle from the balance attractor into the �-x� gait
        # (identical fore/aft legs make the world sign otherwise ambiguous).
        r_vx = np.clip(vx / 0.03, -5.0, 2.0) if upright else -0.4
        # anti-yaw reward: penalize world heading error (wrapped) and the
        # instantaneous body yaw rate, so the policy must pay symmetric
        # counter-torques if straight forward motion is only reachable along
        # the curved limit cycle.
        yaw = _wrap(_euler_from_quat(d.sensor("trunk_quat").data)[2] - self.heading_cmd)
        gz = d.sensor("gyro").data[2]
        r_yaw = -RYAW_A * yaw * yaw - RYAW_G * gz * gz
        # pitch target centered nose-down (-0.07 rad): defines "forward" on
        # the otherwise fore/aft-symmetric body and pulls the cruised posture
        # toward the +x direction.
        r_att = -0.50 * (roll * roll + (pitch - (-0.07)) ** 2)
        gx, gy = d.sensor("gyro").data[0], d.sensor("gyro").data[1]
        r_w = -0.01 * (gx * gx + gy * gy)
        r_h = -0.05 * abs(z - 0.21)
        r_u = -0.005 * np.mean(np.abs(action))
        r_du = -0.02 * np.mean(np.abs(action - self._u_prev))
        reward = r_vx + r_yaw + r_att + r_w + r_h + r_u + r_du
        self._u_prev = action.copy()
        self._u_hist = [*self._u_hist[1:], action.copy()]

        terminated = (
            z < TERM_Z_LO
            or z > TERM_Z_HI
            or abs(roll) > TERM_TILT
            or abs(pitch) > TERM_TILT
            or not np.isfinite(reward)
        )
        if terminated:
            reward -= 25.0
        truncated = self._steps >= MAX_STEPS
        obs = self._get_obs()
        if not np.isfinite(obs).all():
            terminated = True
            obs = obs * 0.0
        info = {}
        if terminated or truncated:
            info["ep_vx"] = self._ep_vx_sum / self._steps
            info["ep_z"] = self._ep_z_sum / self._steps
            info["ep_len"] = self._steps
        return obs, reward, terminated, truncated, info


class VecQuadGait:
    def __init__(
        self,
        n=8,
        seed=0,
        dr=True,
        stride_max=0.05,
        speed_target=0.15,
        yaw0=0.0,
        gait_dir=-1.0,
        stride_min=0.0,
        heading_cmd=0.0,
        turn=0.0,
    ):
        self.stride_max, self.speed_target = stride_max, speed_target
        self.envs = [
            QuadGaitEnv(
                seed=seed + i,
                dr=dr,
                stride_max=stride_max,
                speed_target=speed_target,
                yaw0=yaw0,
                gait_dir=gait_dir,
                stride_min=stride_min,
                heading_cmd=heading_cmd,
                turn=turn,
            )
            for i in range(n)
        ]
        self.n = n
        self.metrics = []

    def reset(self, reset_indices=None):
        if reset_indices is None:
            reset_indices = range(self.n)
        obs = np.stack([self.envs[i].reset() for i in reset_indices])
        return obs

    def step(self, actions):
        obss, rews, terms, truns = [], [], [], []
        reset_idxs = []
        for i, env in enumerate(self.envs):
            o, r, t, tu, info = env.step(actions[i])
            obss.append(o)
            rews.append(r)
            terms.append(t)
            truns.append(tu)
            if info:
                self.metrics.append(info)
                reset_idxs.append(i)
        if reset_idxs:
            ob = self.reset(reset_idxs)
            for i, idx in enumerate(reset_idxs):
                obss[idx] = ob[i]
        return (
            np.stack(obss),
            np.asarray(rews),
            np.asarray(terms, dtype=bool),
            np.asarray(truns, dtype=bool),
        )
