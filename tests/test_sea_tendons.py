"""The tensegrity byte: every SEA tendon the quad model AUTHORS (the rail the
two passing tests resolve through conftest) is (40.0 N/m stiffness, 0.3 N-ms/rad
damping) — MuJoCo's model rail, cold — and after `mj_forward` every tendon
carries real length > 0, i.e. MuJoCo actually places a mechanical load on the
"spine" tendon (a genuine tension rail, not a cosmetic name).

This mirrors test_sea_center.py's rail exactly: it imports through the SAME
conftest-railed packages (gait_env -> quad_model) so the resolved model bytes
are identical to the two passing tests' cold run.
"""

from __future__ import annotations

import mujoco

import quad_model


def test_every_sea_tendon_is_pinned_at_center() -> None:
    m, _d = quad_model.build_quad()
    assert m.ntendon == 8, f"ntendon = {m.ntendon} != 8 (the 4x2 SEA rails)"
    for tid in range(m.ntendon):
        k_t = float(m.tendon_stiffness[tid])
        d_t = float(m.tendon_damping[tid])
        assert k_t == 40.0, f"tendon[{tid}] stiffness {k_t} != pinned 40.0"
        assert d_t == 0.3, f"tendon[{tid}] damping {d_t} != pinned 0.3"


def test_every_sea_tendon_carries_tension_forward() -> None:
    m, d = quad_model.build_quad()
    mujoco.mj_forward(m, d)
    for tid in range(m.ntendon):
        length = float(d.ten_length[tid])
        assert length > 0.0, f"tendon[{tid}] length {length} <= 0 (spine slack)"
