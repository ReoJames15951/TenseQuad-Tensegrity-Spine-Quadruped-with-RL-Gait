"""The SEA-drive (DR) center is resolved from sea_center.json when a
measured bench pair is present; otherwise the CAD pair (40.0 N/m, 0.3
N-m-s/rad) is used.  No bench measurements exist yet, so on this machine the
fallback is what the sim runs with -- the honest "we have not been to the
test stand" default, never a fabricated number.
"""
from __future__ import annotations

import gait_env


def test_sea_center_cad_fallback_without_measured_json() -> None:
    gait_env.QuadGaitEnv._SEA_CENTER = None          # drop any cached pair
    k_s, d_s = gait_env.QuadGaitEnv._sea_center()
    assert (k_s, d_s) == (40.0, 0.3)                 # CAD sizing-sheet pair
