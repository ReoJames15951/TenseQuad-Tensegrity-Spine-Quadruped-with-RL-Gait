"""Five-bar leg kinematics helpers for the SEA quadruped (mirror of fivebar_leg.xml).

All planar math happens in the sagittal (x, z) plane, z up.  Joint angles are
measured from the downward vertical, positive = forward swing (+x).
Provisional geometry targets -- re-validate against the rig before CAD freeze.
"""

from __future__ import annotations

import numpy as np

# Provisional target geometry (must match fivebar_leg.xml).
BASE = 0.060  # hip mount separation (m), "ground link" of the five-bar
L1 = 0.090    # proximal input link (m)
L2 = 0.110    # coupler link (m)
LIM_R = 1.2   # rad, rotor / spring-loaded link joint limits
LIM_C = 1.6   # rad, coupler elbow joint limits


def fk(theta1: float, theta2: float, base: float = BASE,
       L1: float = L1, L2: float = L2):
    """Forward kinematics of the crossed five-bar.

    Returns (E1, E2, F): proximal-link tips and the foot point as (x, z)
    arrays, or None for F when the pose is singular (degenerate / leg fully
    extended).  The chosen apex is the deeper one (smaller z).  Positive joint
    angles swing the foot toward -x (matches fivebar_leg.xml).
    """
    E1 = np.array([-base / 2 - L1 * np.sin(theta1), -L1 * np.cos(theta1)])
    E2 = np.array([base / 2 - L1 * np.sin(theta2), -L1 * np.cos(theta2)])
    d = float(np.linalg.norm(E2 - E1))
    if d <= 1e-9 or d > 2 * L2:  # collapsed elbow or stretch singularity
        return E1, E2, None
    mid = (E1 + E2) / 2.0
    h = float(np.sqrt(max(0.0, L2 ** 2 - (d / 2.0) ** 2)))
    n = np.array([E2[1] - E1[1], -(E2[0] - E1[0])]) / d
    f_up = mid + h * n
    f_dn = mid - h * n
    return E1, E2, f_up if f_up[1] < f_dn[1] else f_dn


def depth(theta1: float, theta2: float, **kw) -> float:
    """Foot depth below the hip line (m)."""
    _, _, f = fk(theta1, theta2, **kw)
    return -f[1] if f is not None else np.inf


def _rot_y_inv(theta):
    """Inverse of the MuJoCo +y rotation acting on an (x, z) plane vector."""
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]])


def fk_full(theta1: float, theta2: float, **kw) -> tuple:
    """Input-joint FK including passive elbow closure angles.

    Returns (E1, E2, F, jc1, jc2) where jc are the coupler joint angles (about
    +y in each proximal link's own frame) that place each coupler tip exactly
    on the foot point F.  These are the values to write into the MuJoCo joints
    jc1/jc2 to reach a *consistent* five-bar configuration for a given
    (j1, j2) pair.  F is None when singular.
    """
    E1, E2, F = fk(theta1, theta2, **kw)
    if F is None:
        return E1, E2, None, np.nan, np.nan
    v1 = _rot_y_inv(theta1).dot(F - E1)
    v2 = _rot_y_inv(theta2).dot(F - E2)
    jc1 = np.arctan2(-v1[0], -v1[1])
    jc2 = np.arctan2(-v2[0], -v2[1])
    return E1, E2, F, jc1, jc2


def ik(fx: float, fz: float, base: float = BASE,
       L1: float = L1, L2: float = L2):
    """Inverse kinematics for the crossed five-bar.

    Given a desired foot position (fx, fz) in the hip mount frame
    (z down), returns (theta1, theta2) in radians, or (nan, nan)
    when unreachable.
    """
    a2 = base / 2.0
    m1 = np.array([-a2, 0.0])
    m2 = np.array([a2, 0.0])
    f  = np.array([fx, fz])
    results = []
    for mi in (m1, m2):
        d_vec = f - mi
        d = float(np.linalg.norm(d_vec))
        if d < 1e-9 or d > L1 + L2 or d < abs(L1 - L2):
            return np.nan, np.nan
        l = (L1*L1 - L2*L2 + d*d) / (2.0 * d)
        l_clip = max(-L1, min(L1, l))
        h_sq = L1*L1 - l_clip*l_clip
        if h_sq < 0.0:
            return np.nan, np.nan
        h = float(np.sqrt(h_sq))
        u = d_vec / d
        perp = np.array([u[1], -u[0]])
        if mi is m1:
            E = mi + l_clip * u + h * perp      # crossed branch
        else:
            E = mi + l_clip * u - h * perp      # mirrored crossed branch
        results.append(E)
    E1, E2 = results
    th1 = np.arctan2(-(E1[0] - m1[0]), -(E1[1] - m1[1]))
    th2 = np.arctan2(-(E2[0] - m2[0]), -(E2[1] - m2[1]))
    return float(th1), float(th2)


def workspace_scan(n: int = 61, lim_r: float = LIM_R, **kw):
    th = np.linspace(-lim_r, lim_r, n)
    F = np.full((n, n, 2), np.nan)
    sing = np.zeros((n, n), dtype=bool)
    for i, t1 in enumerate(th):
        for j, t2 in enumerate(th):
            _, _, f = fk(t1, t2, **kw)
            if f is None:
                sing[i, j] = True
            else:
                F[i, j] = f
    return F, sing, th


def workspace_metrics(n: int = 61, **kw) -> dict:
    """Bounding box and singularity margin metrics for the reachable set."""
    F, sing, _ = workspace_scan(n, **kw)
    valid = F[~np.isnan(F[:, :, 0])]
    d = -valid[:, 1]                      # depths
    stretch = 2 * kw.get("L2", L2)
    return {
        "n_valid": valid.shape[0],
        "n_singular": int(sing.sum()),
        "reach_max": float(valid[:, 1].max() if valid.size else np.nan),   # hip z offset
        "x_min": float(valid[:, 0].min()),
        "x_max": float(valid[:, 0].max()),
        "z_min": float(valid[:, 1].min()),
        "z_max": float(valid[:, 1].max()),
        "depth_median": float(np.median(d)),
        "stretch_margin": stretch,
    }


def swing_at_depth(d_target: float, n: int = 61, **kw) -> tuple[float, float]:
    """Max forward/back x extent of poses whose depth is within 5% of target."""
    if d_target <= 0:
        return np.nan, np.nan
    F, _, _ = workspace_scan(n, **kw)
    d = -F[:, :, 1]
    band = (np.abs(d - d_target) / d_target) < 0.05
    xs = F[:, :, 0][band]
    return (float(xs.min()), float(xs.max())) if xs.size else (np.nan, np.nan)


if __name__ == "__main__":
    m = workspace_metrics()
    print("workspace_metrics:")
    for k, v in m.items():
        print(f"  {k:<16} {v:>12.4f}" if isinstance(v, float) else f"  {k:<16} {v}")
    for d in (0.05, 0.08, 0.11, 0.14):
        r = swing_at_depth(d)
        print(f"  swing at depth {d:5.3f}: x in [{r[0]:+.3f}, {r[1]:+.3f}]")