---
name: Feature / enhancement
about: Propose a change. Anchored to the W15 bring-up card or sizing sheet.
title: "[feat] "
labels: enhancement
assignees: ''
---

**Why** (the W15 card step or sizing-sheet gap this serves)

**Behavior change, if any**
- Which sim symbol(s) / artifacts change.

**Honesty check**
- Does this change how the SEA center resolves? If yes, the fallback must
  stay `(40.0, 0.3)` until a measured `sea_center.json` exists.
- Does this change the ONNX policy path? If yes, bit-exactness must hold.
