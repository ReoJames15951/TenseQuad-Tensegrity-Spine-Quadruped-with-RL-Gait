import numpy as np

import gait_env
import ppo

npz = np.load("quad_ppo.npz")
chk = ppo.Agent(gait_env.OBS_DIM, 8, seed=0)
chk.set_params({k: npz[k] for k in npz.files})
chk.obs_mean = npz["obs_mean"]
chk.obs_var = npz["obs_var"]


def wrap(a):
    return (a + 180) % 360 - 180


for t in (-1.0, -0.5, 0.0, 0.5, 1.0):
    v = gait_env.VecQuadGait(
        n=1,
        seed=99,
        dr=False,
        stride_max=0.05,
        speed_target=0.15,
        gait_dir=-1.0,
        stride_min=0.03,
        turn=t,
    )
    e = v.envs[0]
    obs = v.reset()
    x0 = e.data.xpos[e.trunk_id][0]
    q0 = e.data.sensor("trunk_quat").data
    n = int(10.0 / 0.02)
    for _i in range(n):
        obs, _, _, _ = v.step(chk.act_eval(obs))
    q = e.data.sensor("trunk_quat").data
    yaw = np.degrees(np.arctan2(2 * (q[3] * q[0] + q[1] * q[2]), 1 - 2 * (q[1] ** 2 + q[2] ** 2)))
    yaw0 = np.degrees(
        np.arctan2(2 * (q0[3] * q0[0] + q0[1] * q0[2]), 1 - 2 * (q0[1] ** 2 + q0[2] ** 2))
    )
    dy = wrap(yaw - yaw0)
    dx = e.data.xpos[e.trunk_id][0] - x0
    mode = f"{dy / 10:+5.1f}deg/s" if abs(dy) > 8 else "straight"
    print(
        f"turn {t:+4.1f} | 10s {mode} | dx {dx:+.3f} m  yaw {dy:+.1f}deg  z {e.data.xpos[e.trunk_id][2]:.3f}"  # noqa: E501
    )
