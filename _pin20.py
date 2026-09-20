"""Cold pin dump: ruff's JSON parsed by PYTHON (never PowerShell fields).

For every remaining finding print:  file:row:col  code  >>>EXACT BYTES<<<
after a warm ruff --fix --unsafe-fixes (its own tool), then the cold gate exit.
"""

from __future__ import annotations

import json
import pathlib
import subprocess
import sys

ROOT = pathlib.Path(r"C:\Users\reoja\seriesquad")
PY = sys.executable


def run(*a: str) -> subprocess.CompletedProcess:
    return subprocess.run(list(a), cwd=ROOT, capture_output=True, text=True, encoding="utf-8")


def main() -> int:
    print("[1] ruff's own mechanical close (deterministic tool pass)", flush=True)
    run(PY, "-m", "ruff", "check", ".", "--fix", "--unsafe-fixes")

    print("[2] COLD gate exactly as CI:", flush=True)
    g = run(PY, "-m", "ruff", "check", ".")
    print(f"    ruff check . exit={g.returncode}", flush=True)

    print("[3] python-parsed JSON pin (field names are ruff's own schema):", flush=True)
    j = run(PY, "-m", "ruff", "check", ".", "--output-format", "json")
    findings = json.loads(j.stdout)
    for f in findings:
        p = pathlib.Path(f["location"]["path"])
        row = f["location"]["row"]
        col = f["location"]["column"]
        code = f["code"]
        try:
            lines = p.read_text(encoding="utf-8").splitlines()
            src = lines[row - 1]
        except FileNotFoundError:
            src = "(file not found)"
        print(f"    {p.name}:{row}:{col}  {code}  >>>{src}<<<", flush=True)
    print(f"    total={len(findings)}", flush=True)

    print("[4] the OTHER gate cold (must stay 2 passed below):", flush=True)
    t = run(PY, "-m", "pytest", "tests", "-q", "--no-header")
    tail = [l for l in t.stdout.splitlines() if "passed" in l or "error" in l]
    print(f"    pytest exit={t.returncode}  {tail[-1] if tail else ''}", flush=True)
    return 0 if g.returncode == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
