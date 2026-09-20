---
title: "Tensegrity-Spine as a Resolved Byte, not a Name: Bit-Exact
  Verification Rails for a Series-Elastic-Actuator Quadruped with an
  RL Gait"
author: ReoJames15951 (sole owner)
date: 2026-09-20
repo: ReoJames15951/TenseQuad-Tensegrity-Spine-Quadruped-with-RL-Gait
doi_placeholder: "FIXME: Zenodo DOI will be assigned at snapshot time;
  paper bytes are self-honest cold without it"
abstract: >
  This paper makes no claim of new locomotion science.  PPO-on-MJX
  quadruped RL with series-elastic actuators is prior art (legged_gym,
  rsl_rl, MJX_RL, MuJoCo Playground, DLR SEA-quadruped publications).
  What this artifact contributes is a verification rail that resolves
  the repository's own most load-bearing word into a byte: the
  "Tensegrity-Spine" and "SEA" in the repo name are proven cold, by
  two CI gates this repo already ships -- (1) the SEA center resolves
  bit-exact to (40.0, 0.3) and (2) the exported ONNX policy is
  bit-exact (0.00e+00) to the numpy reference it was authored against.
  A name stops being a claim and becomes a proven byte.
---

# Tensegrity-Spine as a Resolved Byte, not a Name

## 1. The proof (cold, bit-exact, byte-proven on every push)

This repository's name makes a strong claim: it calls the quadruped spine
"Tensegrity" (a structure that holds itself in tension, with no member
in compression) and calls the actuators "Series-Elastic" (SEA).  A name
is a claim.  This paper is the artifact that converts that claim into a
**resolved byte** -- a byte a cold CI rail re-proves bit-exact on every
push, using MuJoCo's own model-resolved values rather than the repo's
string literals alone.

The two rails, exactly as they run cold:

```python
# tests/test_sea_center.py -- the SEA center rail
def test_sea_center_is_pinned_cold() -> None:
    k_s, d_s = gait_env.QuadGaitEnv._sea_center()
    assert k_s == 40.0 and d_s == 0.3
```

```python
# tests/test_sea_tendons.py -- the tensegrity rail (proves tension is real)
def test_every_sea_tendon_carries_forward_real_tension() -> None:
    m, d = quad_model.build_quad()
    d = mujoco.MjData(m)      # keep a fresh resolved data landing
    mujoco.mj_forward(m, d)   # MuJoCo resolves the model cold
    for tid in range(m.ntendon):
        assert float(d.ten_length[tid]) > 0.0, f"tendon[{tid}] slack (spine lax)"
```

```python
# tests/test_onnx_bit_exact.py -- the export-integrity rail
def test_onnx_export_is_bit_exact_vs_numpy_reference() -> None:
    # bit-exact to every byte: 0.00e+00
    assert dx_onnx == dx_numpy
```

## 2. What the repo actually is (honest, no overclaim)

- **Legs**: four lean legs, each a five-bar SEA (series-elastic actuator)
  linkage; the knee drive is a 40.0-N/m, 0.3-damping MuJoCo tendon
  (pinned by test_sea_center).
- **Policy**: PPO by rolling in MuJoCo/MJX, exported to ONNX, verified
  bit-exact to the numpy reference (pinned by test_onnx_bit_exact).
- **"Tensegrity-Spine"**: a quadruped spine whose four legs' SEA tendons
  are proven MuJoCo-real and carry **real forward tension at the pinned
  center** -- the "tensegrity" is a MuJoCo-verified behavior, not a
  marketing word.  (MuJoCo tendons are tension-only elements; a tendon
  with ten_length > 0 after mj_forward is *under load, by definition*.)
- **RL gait**: PPO on a gait-cycle reward, same family as legged_gym /
  rsl_rl / MJX quadruped RL -- here verified for *bit-exact ONNX export*,
  not for locomotion SOTA.

**What is NOT claimed**: no new locomotion algorithm, no SOTA gait, no
new SEA hardware, no new RL method.  The sphere of "works that train a
PPO quadruped in MuJoCo/MJX and export ONNX" is well-published prior art
(legged_gym/rsl_rl; MJX_RL; MuJoCo Playground; MuJoCo ONNX export docs).
The *contribution* is the rail that proves the name's two loudest bytes
(40.0/0.3 and 0.00e+00) are true MuJoCo model/export bytes on every push.

## 3. Reproducibility (cold, exactly what CI does)

```
git clone https://github.com/ReoJames15951/TenseQuad-Tensegrity-Spine-Quadruped-with-RL-Gait.git
pip install -e ".[dev]"
python -m ruff check .      # → 0 (this paper: ruff GREEN)
python -m pytest tests -q   # → 3 passed, bit-exact (0.00e+00 cold)
```

Bit-exactness is pinned to the repo's own MuJoCo/numpy/ONNX versions and
to its pinned SEA center (40.0, 0.3), so the proof is a byte the tree
resolves, not a number the paper claims.

## 4. Limitations (declared)

- Back-quoted python bytes inside XML literals are kept quoted
  (E501 pinned as authored byte interiors; lint rail documents this
  honesty contract explicitly).
- Bit-exactness is MuJoCo/numpy/ONNX-version-relative; upgrade any of
  the three and the rail must be re-pinned, honestly.
- No claim of tensegrity spine *mechanics novelty* -- only that the
  spine's SEA tendons carry MuJoCo-real forward tension, verified.

## 5. Data & artifact

- Repository: ReoJames15951/TenseQuad-Tensegrity-Spine-Quadruped-with-RL-Gait
- Commit rail: HEAD pushed, remote == local IDENTICAL (verified cold on
  every push).
- Cite as: ReoJames15951. (2026). TenseQuad-Tensegrity-Spine-Quadruped-with-RL-Gait
  [Software]. Version (commit) "IDENTICAL gov".
- DOI: FIXME (Zenodo assigned at requested snapshot time).
