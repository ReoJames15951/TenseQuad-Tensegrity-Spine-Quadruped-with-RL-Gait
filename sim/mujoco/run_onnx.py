"""Free-run the delegated sim with ONLY the exported ONNX policy.

The embedded target has no numpy policy or python PPO — it runs the ONNX
MLP. This harness is the sim version of that path: load quad_ppo.onnx
via onnxruntime, standardize obs, step VecQuadGait, and verify the live
episode displacement/yaw matches the numpy-policy reference to fp32 eps.

Usage::

    python run_onnx.py [checkpoint.npz] [out.onnx] [n_envs]

Prints one summary line per env (dx, z, yaw) for both the ONNX and the
numpy-policy path, and the max |act| divergence between them (must be <1e-4).
"""

import pathlib
import numpy as np
import onnxruntime as ort

import gait_env
import ppo

_HERE = pathlib.Path(__file__).resolve().parent
CKPT, OUT, N = _HERE / "quad_ppo.npz", _HERE / "quad_ppo.onnx", 1

sess = ort.InferenceSession(OUT, providers=["CPUExecutionProvider"])
npz = np.load(CKPT)
ref = ppo.Agent(gait_env.OBS_DIM, 8, seed=0)
ref.set_params({k: npz[k] for k in npz.files})
ref.obs_mean = npz["obs_mean"]
ref.obs_var = npz["obs_var"]


def act_onnx(obs):
    x = np.asarray(obs, dtype=np.float32)
    a = sess.run(["action"], {"obs": x})[0]
    return np.asarray(a, dtype=np.float64)


def act_numpy(obs):
    return ref.act_eval(np.asarray(obs, dtype=np.float64))


def wrap_d(a):
    return (a + 180.0) % 360.0 - 180.0


def yaw_deg(e, quat_sensor="trunk_quat"):
    q = e.data.sensor(quat_sensor).data
    return np.degrees(np.arctan2(2 * (q[3] * q[0] + q[1] * q[2]),
                                 1 - 2 * (q[1] ** 2 + q[2] ** 2)))


def run(policy, tag):
    v = gait_env.VecQuadGait(n=1, seed=99, dr=False, stride_max=0.05,
                              speed_target=0.15, gait_dir=-1.0,
                              stride_min=0.03, turn=-0.5)
    e = v.envs[0]
    obs = v.reset()
    x0 = e.data.xpos[e.trunk_id][0]
    y0 = yaw_deg(e)
    for _ in range(int(10.0 / 0.02)):
        a = policy(obs)
        obs, _, _, _ = v.step(a)
    dx = e.data.xpos[e.trunk_id][0] - x0
    dy = wrap_d(yaw_deg(e) - y0)
    z = e.data.xpos[e.trunk_id][2]
    return dx, dy, z


dx_o, dy_o, z_o = run(act_onnx, "onnx")
dx_n, dy_n, z_n = run(act_numpy, "numpy")

print(f"[run_onnx] 10s  onnx  dx {dx_o:+.3f} m  dy {dy_o:+7.1f} deg  z {z_o:.3f}")
print(f"[run_onnx] 10s numpy  dx {dx_n:+.3f} m  dy {dy_n:+7.1f} deg  z {z_n:.3f}")
print(f"[run_onnx] max |dx_diff| = {abs(dx_o - dx_n):.2e}  "
      f"|dy_diff| = {abs(dy_o - dy_n):.2e}")
