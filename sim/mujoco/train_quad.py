"""Train (or evaluate) a PPO policy that makes the SEA quadruped trot.

Run from this directory (sim/mujoco)::

    python train_quad.py --iters 80                    # train fresh
    python train_quad.py --iters 100 --load ckpt.npz   # resume
    python train_quad.py --eval-only --load ckpt.npz   # evaluate policy
    python train_quad.py --eval-only --load ckpt.npz --render

Policy clock 50 Hz; the env executes 10 PD substeps (500 Hz) per action.

Log format per iteration::

    it  mean_a  ep_len  ep_vx  ep_z  | eval_vx eval_z eval_pitch
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

import gait_env
import ppo
import run_quad as rq

HERE = Path(__file__).parent
DEFAULT_CKPT = HERE / "quad_ppo.npz"


def evaluate(
    agent,
    n_envs=3,
    T=8.0,
    seed=99,
    render=False,
    stride_max=0.05,
    speed_target=0.15,
    gait_dir=-1.0,
    stride_min=0.0,
    heading_cmd=0.0,
    turn=0.0,
):
    """Run deterministic episodes, return avg metrics + per-env final x."""
    envs = gait_env.VecQuadGait(
        n=n_envs,
        seed=seed,
        dr=False,
        stride_max=stride_max,
        speed_target=speed_target,
        gait_dir=gait_dir,
        stride_min=stride_min,
        heading_cmd=heading_cmd,
        turn=turn,
    )
    obs = envs.reset()
    np.zeros(n_envs)
    speeds, zs, pitches = [], [], []
    yaw_start = np.array(
        [
            np.degrees(gait_env._euler_from_quat(e.data.sensor("trunk_quat").data)[2])
            for e in envs.envs
        ]
    )
    steps = int(T / (envs.envs[0].dt * gait_env.N_SUB))
    viewer = None
    if render:
        import mujoco

        viewer = mujoco.viewer.launch_passive(envs.envs[0].model, envs.envs[0].data)
    for _ in range(steps):
        a = agent.act_eval(obs)
        obs, _rew, _term, _trunc = envs.step(a)
        vx = np.array(
            [(e.data.xpos[e.trunk_id][0] - e._xprev) / (e.dt * gait_env.N_SUB) for e in envs.envs]
        )
        z = np.array([e.data.xpos[e.trunk_id][2] for e in envs.envs])
        for e in envs.envs:
            _r, p, _ = rq._euler_from_quat(e.data.sensor("trunk_quat").data)
            pitches.append(abs(p))
        speeds.append(vx)
        zs.append(z)
        if render and viewer is not None:
            viewer.sync()
    if render and viewer is not None:
        viewer.close()
    speeds = np.array(speeds)
    zs = np.array(zs)
    final_x = np.array([e.data.xpos[e.trunk_id][0] for e in envs.envs])
    yaw_drift = (
        np.array(
            [
                np.degrees(gait_env._euler_from_quat(e.data.sensor("trunk_quat").data)[2])
                for e in envs.envs
            ]
        )
        - yaw_start
    )
    return {
        "vx": float(speeds.mean()),
        "z": float(zs.mean()),
        "z_min": float(zs.min()),
        "pitch_max": float(np.degrees(np.max(pitches))),
        "dist": float(final_x.mean()),
        "yaw_drift": float(yaw_drift.mean()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--iters", type=int, default=80)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--horizon", type=int, default=64, help="policy steps/rollout")
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dr", type=int, default=1, help="domain randomization on/off")
    ap.add_argument(
        "--stride-max",
        type=float,
        default=0.05,
        help="max stride stroke (0 = stand-balance curriculum)",
    )
    ap.add_argument(
        "--speed-target", type=float, default=0.15, help="walk speed reward target (m/s)"
    )
    ap.add_argument(
        "--gait-dir",
        type=float,
        default=-1.0,
        help="stance-sweep sense; must match the ratchet-pad sign",
    )
    ap.add_argument(
        "--stride-min",
        type=float,
        default=0.0,
        help="warm-up stride floor: force stride while the ratchet gait entrains",
    )
    ap.add_argument(
        "--heading-cmd",
        type=float,
        default=0.0,
        help="commanded world heading (rad) tracked by the anti-yaw reward; turning input",
    )
    ap.add_argument(
        "--turn",
        type=float,
        default=0.0,
        help="open-loop differential yaw-base on left vs right feet (rad)",
    )
    ap.add_argument("--eval-every", type=int, default=10)
    ap.add_argument("--save", type=Path, default=DEFAULT_CKPT)
    ap.add_argument("--load", type=Path, default=None)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--render", action="store_true")
    args = ap.parse_args()

    chk = ppo.Agent(gait_env.OBS_DIM, 8, hidden=args.hidden, lr=args.lr, seed=args.seed)
    if args.load and args.load.exists():
        npz = np.load(args.load)
        chk.set_params({k: npz[k] for k in npz.files})
        if "obs_mean" in npz.files:
            chk.obs_mean = npz["obs_mean"]
        if "obs_var" in npz.files:
            chk.obs_var = npz["obs_var"]
        print(f"[load] {args.load}")

    if args.eval_only:
        res = evaluate(
            chk,
            T=8.0,
            render=args.render,
            stride_max=args.stride_max,
            speed_target=args.speed_target,
            gait_dir=args.gait_dir,
            stride_min=args.stride_min,
            heading_cmd=args.heading_cmd,
            turn=args.turn,
        )
        print(
            f"[eval] vx={res['vx']:+.3f} m/s  z={res['z']:.3f} m "
            f"(min {res['z_min']:.3f})  pitch_max={res['pitch_max']:.1f} deg  "
            f"dist={res['dist']:+.3f} m  yaw={res['yaw_drift']:+.1f}deg"
        )
        return

    envs = gait_env.VecQuadGait(
        n=args.n_envs,
        seed=args.seed,
        dr=bool(args.dr),
        stride_max=args.stride_max,
        speed_target=args.speed_target,
        gait_dir=args.gait_dir,
        stride_min=args.stride_min,
    )
    H, N = args.horizon, args.n_envs
    obs = envs.reset()

    n_dims = obs.shape[1]
    it_rew = np.zeros(args.iters)
    t0 = time.time()
    for it in range(args.iters):
        buf_obs = np.zeros((H + 1, N, n_dims))
        buf_act = np.zeros((H, N, 8))
        buf_rew = np.zeros((H, N))
        buf_done = np.zeros((H, N), dtype=bool)
        buf_trunc = np.zeros((H, N), dtype=bool)
        buf_logp = np.zeros((H, N))
        buf_val = np.zeros((H, N))

        for t in range(H):
            buf_obs[t] = obs
            a, logp, v = chk.act(obs)
            obs2, rew, done, trunc = envs.step(a)
            buf_act[t] = a
            buf_logp[t] = logp
            buf_val[t] = v
            buf_rew[t] = rew
            buf_done[t] = done
            buf_trunc[t] = trunc
            obs = obs2.copy()
        buf_obs[H] = obs
        chk.update_obs_stats(buf_obs.reshape(-1, n_dims))
        _, _, v_last = chk.act(buf_obs[H], sample=False)

        advs, rets = chk.gae(buf_rew, buf_done, buf_trunc, buf_val, v_last)
        chk.train(
            buf_obs[:-1].reshape(-1, n_dims),
            buf_act.reshape(-1, 8),
            rets.reshape(-1),
            advs.reshape(-1),
            buf_logp.reshape(-1),
        )

        it_rew[it] = buf_rew.mean()
        ep_vx = ep_z = ep_len = None
        if envs.metrics:
            ep_vx = float(np.mean([m["ep_vx"] for m in envs.metrics]))
            ep_z = float(np.mean([m["ep_z"] for m in envs.metrics]))
            ep_len = float(np.mean([m["ep_len"] for m in envs.metrics]))
            envs.metrics.clear()

        line = f"it {it:3d}  R {it_rew[it]:+.3f}  {'' if ep_len is None else f'len {ep_len:5.0f}  vx {ep_vx:+.3f}  z {ep_z:.3f}'}"
        if (it + 1) % args.eval_every == 0:
            res = evaluate(
                chk,
                T=6.0,
                stride_max=args.stride_max,
                speed_target=args.speed_target,
                gait_dir=args.gait_dir,
                stride_min=args.stride_min,
            )
            line += f"  |  eval vx {res['vx']:+.3f}  z {res['z']:.3f}  pitch {res['pitch_max']:.1f}deg  yaw {res['yaw_drift']:+.1f}deg"
        print(line, flush=True)

        if (it + 1) % 20 == 0:
            args.save.parent.mkdir(parents=True, exist_ok=True)
            np.savez(args.save, **chk._params(), obs_mean=chk.obs_mean, obs_var=chk.obs_var)
            print(f"    saved {args.save}")

    args.save.parent.mkdir(parents=True, exist_ok=True)
    np.savez(args.save, **chk._params(), obs_mean=chk.obs_mean, obs_var=chk.obs_var)
    print(f"\n[done] {args.save} in {time.time() - t0:.0f}s")
    res = evaluate(
        chk,
        T=8.0,
        stride_max=args.stride_max,
        speed_target=args.speed_target,
        gait_dir=args.gait_dir,
        stride_min=args.stride_min,
        heading_cmd=args.heading_cmd,
        turn=args.turn,
    )
    print(
        f"[final eval] vx={res['vx']:+.3f} m/s  z={res['z']:.3f} m "
        f"(min {res['z_min']:.3f})  pitch_max={res['pitch_max']:.1f} deg  "
        f"dist={res['dist']:+.3f} m  yaw={res['yaw_drift']:+.1f}deg"
    )


if __name__ == "__main__":
    main()
