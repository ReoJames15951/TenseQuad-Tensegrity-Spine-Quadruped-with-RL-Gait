## What / why
One line on the change; link the W15 card step or sizing-sheet row it serves.

## Honesty gates touched
- [ ] `python -m pytest tests` passes (bit-exact 0.00e+00 + SEA CAD fallback)
- [ ] `ruff check .` is clean
- [ ] If I touched the SEA center: fallback is still `(40.0, 0.3)` without `sea_center.json`
- [ ] If I touched the ONNX path: `run_onnx.py` still prints `max |dx_diff| = 0.00e+00`

## Numbers cited (only measured-or-doc ones)
- [ ] Every number in this PR is from the sizing sheet, the W15 card, or the bench.
