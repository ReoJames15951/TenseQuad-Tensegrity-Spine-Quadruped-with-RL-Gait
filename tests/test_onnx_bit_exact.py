"""ONNX-vs-numpy bit-exactness gate (the deployment contract).

run_onnx.run(policy, tag) rebuilds the SAME seeded episode (VecQuadGait
n=1, seed=99, 10 s) and calls `policy(obs)` every control step.  Replaying
the episode with the exported ONNX graph (act_onnx) and with the numpy
reference policy (act_numpy) must land the trunk on the *identical* spot:
the ONNX export is what ships to embedded, and any drift here means the
.gitattributes-tracked quad_ppo.onnx no longer represents quad_ppo.npz.
"""

from __future__ import annotations

import run_onnx


def test_onnx_export_reproduces_numpy_policy_bit_exact() -> None:
    dx_o, dy_o, z_o = run_onnx.run(run_onnx.act_onnx, "onnx")
    dx_n, dy_n, z_n = run_onnx.run(run_onnx.act_numpy, "numpy")
    assert dx_o == dx_n, f"trunk x: onnx {dx_o} vs numpy {dx_n}"
    assert dy_o == dy_n, f"trunk yaw: onnx {dy_o} vs numpy {dy_n}"
    assert z_o == z_n, f"trunk z: onnx {z_o} vs numpy {z_n}"
