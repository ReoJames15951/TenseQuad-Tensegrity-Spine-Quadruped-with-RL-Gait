# Changelog

All notables are the honest, measurable kind.  This repo is pre-1.0;
SN-null means "not yet at the bench", never "measured but not written up".

## [Unreleased]

### Added
- Public open-sourcing surface for the W15 quadruped SEA section:
  - `LICENSE` (MIT), `README.md`, `pyproject.toml` (ruff + pytest gates),
    `.gitignore` (policy artifacts policy: commit the canonical
    `quad_ppo.npz/.onnx`, ignore every other *.npz/*.onnx),
    `CONTRIBUTING.md`, issue/PR templates.
  - `tests/` suite with two honesty gates:
    - `test_onnx_bit_exact.py` — the exported ONNX graph must reproduce
      the numpy-policy trunk trajectory **bit-exact** on the same seeded
      episode (`max |dx_diff| == 0.00e+00`).
    - `test_sea_center.py` — without `sea_center.json` the SEA drive center
      resolves to the *CAD* fallback `(40.0 N/m, 0.3 N-m-s/rad)` and never
      invents a "measured" pair.
  - `docs/` sizing sheet (SEA section, W15 layout) and the W15 bring-up card.

### Changed
- `run_onnx.py` path handling is now module-anchored (`_HERE`), so the
  ONNX-verification harness imports cleanly from the pytest suite at the
  repository root, not only when run with `sim/mujoco` as the working
  directory.

### Fixed
- `test_onnx_bit_exact.py` bit-exactness rerolls both policies from the
  same `seed=99` episode, so the assertion is a real fp32-equal check, not
  a tolerance guess.

### Honesty notes (bench contract)
- SEA drive center is `(40.0, 0.3)` — the CAD sizing-sheet pair — until the
  measured `sea_center.json` lands from the W15 bring-up.  Nothing here has
  been to the bench yet; the W15 bring-up card is the sheet for that gate,
  and it's still honest about it.
