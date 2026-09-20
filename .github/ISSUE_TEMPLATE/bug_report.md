---
name: Bug report
about: Report behavior that contradicts the sizing sheet or the ONNX contract.
title: "[bug] "
labels: bug
assignees: ''
---

**What contradicts which doc/spec?**
e.g. "ONNX bit-exactness fails: `python run_onnx.py` prints a nonzero max |dx_diff|" or "SEA center resolved to something other than CAD (40.0, 0.3) with no sea_center.json".

**Environment**
- OS / Python / onnxruntime versions
- Working directory used to run `run_onnx.py` (path-relative bug class)

**Repro**
Exact commands + the printed `run_onnx.py` summary lines.

**Expected vs actual**
Quote the honest number you saw and the number the doc promises.
