# Contributing

## Project shape

- `sim/mujoco/` — MuJoCo sim of the W15 quadruped SEA section:
  `gait_env.py` (task + training envs, SEA drive-center resolution),
  `quad_model.py` (xml builder returning `build_quad(k_s=40.0, d_s=0.3)`),
  `leg_utils.py` (FK/IK + workspace scan), `ppo.py` (policy core),
  `run_onnx.py` (ONNX vs numpy rollout harness), `quad_ppo.npz`/`.onnx`
  (committed canonical policy pair; drift-* variants are gitignored).
- `docs/` — `sizing-sheet.md` (SEA sizing, CAD 40.0/0.3) and
  `W15-bringup-card.md` (the bench gate).

## The honesty contract (please keep it)

1. **No invented bench numbers.**  Until a measured `sea_center.json` is
   produced by the W15 bench, the sim must keep resolving the CAD pair
   `(40.0, 0.3)`.  That is what `tests/test_sea_center.py` pinsched.
2. **Bit-exactness is the deployment gate.**  The ONNX graph is what ships
   to the embedded target; if a change makes `run_onnx.py` report a
   nonzero `max |dx_diff|`, it is a regression, not a tolerance.
3. **State both sides of a bisection.**  When you change a sim knob, say
   what it was and what it is — the sizing sheet keeps the CAD column and
   the measured column as separate honest cells.

## Checks before push

```
ruff check .                 # style gate (pyproject)
python -m pytest tests       # bit-exact + SEA-center gates
```

Add a regression test with any change that moves policy or SEA numbers —
that is what makes the "honest numbers" durable, not just claimed.
