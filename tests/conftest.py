# Tests need the mujoco scripts on path.  pyproject.toml already sets
# pytest pythonpath=["sim/mujoco"]; this file is a belt-and-braces explicit
# alias so the suite also works if someone invokes pytest with a custom
# rootdir override.

from __future__ import annotations

import pathlib
import sys

_ROOT = pathlib.Path(__file__).resolve().parents[1]
SIM = _ROOT / "sim" / "mujoco"
if str(SIM) not in sys.path:
    sys.path.insert(0, str(SIM))
