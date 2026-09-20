"""The SEA-tendon byte -- resolved cold, MuJoCo-honest, never a named claim.

This rail does NOT assert "8 tendons == (40.0, 0.3)" and does NOT assert
"every ten_length > 0" -- cold MuJoCo 3.x resolves tendons 6-7 to
(39.3607, 0.1168) and ten_length to FINITE REAL *signed* lengths, so both
of those claims would be, byte-honestly, fabricated.  A byte is not a
claim; this rail pins only what MuJoCo cold-resolves truthfully:

  * m.ntendon == 8          (MuJoCo authored 8 SEA tendon elements)
  * every resolved tendon length  is FINITE REAL (never NaN/Inf)
  * every resolved tendon velocity is FINITE REAL (never NaN/Inf)
  * the env's auth-SEA center still falls back to the authored CAD pair
    (40.0, 0.3) -- the honest default until a bench measurement exists.

"Tensegrity-Spine" stops being a name the day MuJoCo resolves the spine
byte-cold, and not one byte before.  This test is that cold.day.
"""

from __future__ import annotations

import math

import mujoco

import gait_env


def test_every_authored_sea_tendon_resolves_finite_real_cold() -> None:
    env = gait_env.QuadGaitEnv()
    env.reset()
    m, d = env.model, env.data

    assert m.ntendon == 8, f"ntendon = {m.ntendon} != 8 (the 4x2 SEA spine)"

    mujoco.mj_forward(m, d)  # MuJoCo resolves the model, cold, byte-for-byte

    for tid in range(m.ntendon):
        length = float(d.ten_length[tid])
        vel = float(d.ten_velocity[tid])
        assert math.isfinite(length), f"tendon[{tid}] length {length} not finite"
        assert math.isfinite(vel), f"tendon[{tid}] velocity {vel} not finite"


def test_sea_center_still_falls_back_to_authored_cad_pair_cold() -> None:
    gait_env.QuadGaitEnv._SEA_CENTER = None  # never trust a stale cache
    k_s, d_s = gait_env.QuadGaitEnv._sea_center()
    assert (k_s, d_s) == (40.0, 0.3), "SEA center must resolve (40.0, 0.3)"
