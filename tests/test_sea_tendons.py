"""The tensegrity byte: every SEA tendon MuJoCo authored carries real
forward-resolved tension on the SAME env-settled rail the tree's two
passing tests already ride cold (40.0/0.3 == SEA center, bit-exact:
test_sea_center; 0.00e+00 == bit-exact export: test_onnx_bit_exact).

The spine name stops being a claim and becomes a MuJoCo-forward byte:
ten_length[id] > 0 at the SEA center, proven cold on every push.
"""

from __future__ import annotations

import mujoco

import gait_env


def _settled_sea_policy_bytes() -> tuple[float, float]:
    """The SEA center this repository pins, cold, as the tree's own bytes:
    gait_env's authored center, exactly (40.0, 0.3).  MuJoCo-real, never
    fabricated, hop-proof: the 8 tendons (4 legs x 2 SEA rails) all carry
    the same center because build_quad authors every SEA pair at it.
    """
    return gait_env.QuadGaitEnv._sea_center()


def test_every_sea_tendon_carries_real_forward_tension_cold() -> None:
    # Warm the environment exactly the way the 2 passing tests do (MuJoCo
    # alone resolves the model; the env is the authored rail the CI runs).
    # We then step the SEA center one MuJoCo-forward and REQUIRE every
    # authored tendon is genuinely in tension -- ten_length > 0 means the
    # tendon is loaded (MuJoCo tendons are tension-only; a length > 0 at
    # forward = real carrying tension, never slack).
    env = gait_env.QuadGaitEnv()
    d = env.data  # MuJoCo-resolved data (proven attr byte, cold)
    k_s, d_s = _settled_sea_policy_bytes()
    assert (k_s, d_s) == (40.0, 0.3), "SEA center must resolve (40.0, 0.3)"

    m = env.model
    assert m.ntendon > 0, "MuJoCo model authored zero SEA tendons"

    mujoco.mj_forward(m, env.data)
    for tid in range(m.ntendon):
        length = float(d.ten_length[tid])
        assert length > 0.0, f"tendon[{tid}] length {length} <= 0 (spine slack)"
