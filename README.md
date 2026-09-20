# SeriesQuad — simulation-first SEA quadruped locomotion stack

[![CI — tests](https://github.com/ReoJames15951/TenseQuad-Tensegrity-Spine-Quadruped-with-RL-Gait/actions/workflows/tests.yml/badge.svg)](https://github.com/ReoJames15951/TenseQuad-Tensegrity-Spine-Quadruped-with-RL-Gait/actions/workflows/tests.yml)
[![CI — lint](https://github.com/ReoJames15951/TenseQuad-Tensegrity-Spine-Quadruped-with-RL-Gait/actions/workflows/tests.yml/badge.svg)](https://github.com/ReoJames15951/TenseQuad-Tensegrity-Spine-Quadruped-with-RL-Gait/actions/workflows/tests.yml)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A quadruped locomotion research stack built MuJoCo-first: five-bar (SEA)
legs, a pure-numpy PPO policy, and an ONNX export path that is **bit-exact**
against the numpy reference policy.

## What this is

An end-to-end quadruped locomotion research harness:

- **Model.** A four-legged, series-elastic five-bar walker defined in MuJoCo
  XML (`quad_model.py`, `fivebar_leg.xml`). Each leg is a crossed five-bar
  linkage with a tendon-driven series-elastic actuator — the same
  architecture used in compliant, energy-recycling single-leg rigs.
- **Policy.** A PPO agent implemented **in pure numpy** (`ppo.py`) with no
  deep-learning framework dependency. It learns a gait template (stride
  length, stance torque blend, foot lift, cadence) that keeps the compliant
  platform upright and moving forward.
- **Deployment path.** The trained policy is exported to ONNX
  (`export_onnx.py`) and run through `onnxruntime` (`run_onnx.py`). This
  mirrors what runs on embedded hardware: a standalone model artifact with no
  numpy/Python counterpart at inference time.
- **Domain randomization.** `gait_env.py` randomizes friction, trunk mass,
  spring stiffness/damping, delays, and noise per episode so the learned
  policy is robust across a family of physical builds. The DR center for the
  series-elastic actuator (spring stiffness `k_s`, damping `d_s`) defaults to
  the CAD/best-estimate pair `(40.0 N/m, 0.3 N·m·s/rad)` and is meant to be
  overridden from a measured bench value via a `sea_center.json` drop-in file.

## Why it matters

Compliant legs make the same policy transfer across a *range* of physical
spring stiffnesses only if the sim's DR distribution is centered near the
real robot's measured values. This repo makes that loop explicit and
verifiable: train in MuJoCo, export to ONNX, verify the export is bit-exact,
then identify the real actuator center on a bench and drop it in without
touching the policy. See `docs/architecture.md` and `docs/research.md`.

## Verified results

The following is the only hard result we currently claim — it is reproduced
live by `tests/test_onnx_bitexact.py`:

| Check | Result | Command |
|---|---|---|
| ONNX policy vs numpy reference, max policy-output diff at 10 s sim | **`0.00e+00`** (bit-exact) | `python run_onnx.py` |
| SEA center default (no `sea_center.json`) | `(40.0 N/m, 0.3 N·m·s/rad)` | `tests/test_sea_center.py` |

Everything else (march distance, yaw drift, leg workspace bounds) is a
**pending measurement** — see `docs/research.md`. We deliberately do not
publish numbers we have not measured or reproduced.

## Repository layout

```text
seriesquad/
├── docs/                 # design, install, usage, development, research
├── sim/mujoco/           # MuJoCo model + PPO + ONNX toolchain (the module)
│   ├── gait_env.py       # QuadGaitEnv / VecQuadGaitEnv (PPO env, DR)
│   ├── quad_model.py     # builds the quad MuJoCo model from XML fragments
│   ├── leg_utils.py      # five-bar FK/IK + workspace utilities
│   ├── ppo.py            # pure-numpy PPO agent
│   ├── run_onnx.py       # ONNX-vs-numpy bit-exact verifier
│   ├── run_quad.py       # single/quad gait drivers
│   ├── run_single_leg.py # one-leg workspace / bounce diagnostics
│   └── export_onnx.py    # policy -> ONNX exporter
├── tests/                # pytest suite (see "Testing")
├── scripts/              # repo-level utility scripts
├── .github/workflows/    # CI (tests + lint)
└── LICENSE, README.md, pyproject.toml, ...
```

## Installation

Requires **Python 3.11+** on Windows / Linux / macOS.

```bash
git clone https://github.com/ReoJames15951/TenseQuad-Tensegrity-Spine-Quadruped-with-RL-Gait.git
cd seriesquad

python -m venv .venv
# Windows: .venv\Scripts\activate    Linux/macOS: source .venv/bin/activate

pip install -e ".[dev]"
```

That installs: `mujoco`, `numpy`, `onnxruntime`, `onnx`, `ppo` (local),
plus dev tools (`ruff`, `pytest`). Full details in `docs/installation.md`.

## Quickstart

```bash
# 1. Verify the ONNX policy is bit-exact against numpy
python sim/mujoco/run_onnx.py

# 2. Train a policy from scratch (about 10-20 min on a laptop CPU)
python sim/mujoco/train_quad.py --iters 80

# 3. Export the trained policy to ONNX
python sim/mujoco/export_onnx.py quad_ppo.npz quad_ppo.onnx

# 4. Render the learned gait
python sim/mujoco/run_quad.py --mode walk --render
```

## Testing

```bash
pytest                                  # run the whole suite
pytest -k onnx                          # just the bit-exact check
ruff check .                            # lint
ruff format --check .                   # formatting
```

The suite includes a **live MuJoCo compile + 10 s sim roll** as an
integration test — it catches model breakage, not just math-drift.

## License

MIT — see [LICENSE](LICENSE). Copyright (c) 2026 SeriesQuad contributors.

## Status

Sim module is functionally complete and tested. The **W15 hardware bring-up
gate** (measuring the real SEA `(k_s, d_s)` on the single-leg bench and
dropping it into `sea_center.json`) is the next milestone and is **not yet
run** — it requires physical hardware. See `docs/W15-bringup-card.md`.
