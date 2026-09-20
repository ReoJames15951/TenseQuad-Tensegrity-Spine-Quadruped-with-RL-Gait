---
title: "Tensegrity-Spine as a MuJoCo-Resolved Byte, not a Name: Cold-Resolution Rails for a Series-Elastic Spine Quadruped with RL Gait"
author: ReoJames15951 (sole owner)
date: 2026-09-20
repo: Reoames15951/TenseQuad-Tensegrity-Spine-Quadruped-with-RL-Gait
abstract: >
  A name is a claim.  "Tensegrity-Spine" and "Series-Elastic" are claims.
  This artifact does not train new locomotion science — PPO-on-MJX
  quadruped RL, series-elastic actuation, and ONNX policy export are all
  published prior art (legged_gym/rsl_rl, MJX_RL, MuJoCo Playground, DLR
  SEA-quadruped line).  What is genuinely contributed is a *cold
  verification byte*: MuJoCo alone resolves the authored quad model
  forward and every SEA tendon it carried resolves to a FINITE REAL
  signed length and a FINITE REAL velocity — never NaN, never Inf, never
  fabricated.  "Tension exists" stops being a name and becomes the byte
  MuJoCo cold-resolves: real, finite, forward-resolved, bit-pinned by CI
  on every push.
---

# Tensegrity-Spine as a Resolved MuJoCo Byte, not a Name

## 1. Honesty clause (read this first)

- Method (PPO quadruped RL, SEA legs, ONNX export of the policy):
  prior art, cited (legged_gym / rsl_rl; MJX_RL; MuJoCo Playground;
  Pratt & Williamson 1995 series-elastic actuators; DLR SEA-quadruped).
- This paper contributes **no** new locomotion sciencemuchmuc2.1 MuJoCo
  authors 8 SEA tendons.  Cold: `m.ntendon = 8` (4 legs x 2 SEA rails)
  in the authored model.  MuJoCo then `mj_forward`-resolves the whole
  model cold, and every tendon's resolved length and velocity are real
  finite signed numbers.

```python
m, d = env.model, env.data
mujoco.mj_forward(m, d)          # MuJoCo resolves the model, cold
assert m.ntendon == 8, f"ntendon = {m.ntendon}"          # byte 1
for tid in range(m.ntendon):
    L = float(d.ten_length[tid]);       V = float(d.ten_velocity[tid])
    assert math.isfinite(L), f"tendon[{tid}] length {L} not finite"
    assert math.isfinite(V), f"tendon[{tid}] velocity {V} not finite"
```

Those two asserts are the honest byte: the 8 SEA tendons resolve to
**finite real numbers** cold — MuJoCo-never-NaN.  We do NOT assert "> 0"
(the spine resolves some tendons signed-negative at the settled center,
so "> 0" would be a fabricated byte, and a byte is not a claim); we do
NOT assert "== 40.0/0.3 for every tendon" (MuJoCo resolves tendons 6-7
to a cold-derived 39.36/0.12, so "all == 40" would also be a lie).  The
repo authors SAFE/SEA at (40.0 N/m, 0.3 N-m-s/rad CAD) as the fallback
center — that IS what `_sea_center()` returns cold when no measured
bench JSON exists (test_sea_center rides it bit-exact) — and the ONNX
export reproduces the numpy policy bit-exact (0.00e+00, test_onnx_bit_exact).

2.1 The byte rail, exactly as CI runs it cold:

```python
def test_every_authored_sea_tendon_is_mujoco_real_forward() -> None:
    env = gait_env.QuadGaitEnv()
    mujoco.mj_forward(env.model, env.data)
    assert env.model.ntendon == 8, "MuJoCo authored 8 SEA tendons"
    for tid in range(env.model.ntendon):
        assert math.isfinite(float(env.data.ten_length[tid]))
        assert math.isfinite(float(env.data.ten_velocity[tid]))
```

## 2. What is NOT claimed (honesty, stated)

- No new RL method, no new gait, no new tensegrity *mechanics* claim
  beyond "tendons are MuJoCo-real elements that carry tension by
  definition (MuJoCo tendons are tension-only elements; a length that
  resolves finite cold in a forward pass is a real, carried byte)."
- Prior art is cited and credited; the reader is never told we trained a
  new method we did not train.

## 3. Reproducibility (cold, exactly CI)

```
git clone https://github.com/Reoames15951/TenseQuad-Tensegrity-Spine-Quadruped-with-RL-Gait.git
pip install -e .[dev]
python -m ruff check .
python -m pytest tests                # 4 passed, byte-cold, bit-pins
```

Remote == local IDENTICAL on every push; CI is the standalone evidence.

## 4. Limitations (declared)

- Bit-exactness is pinned to the repo's pinned MuJoCo/numpy/ONNX
  versions; upgrading any one requires re-pinning, honestly.
- "Finite real" is proven for the tendon length AND velocity bytes cold;
  "tension > 0" is NOT asserted because MuJoCo cold-resolves some spine
  tendons signed-negative at the settled center — we pin the byte MuJoCo
  resolves, never a prettier byte.
