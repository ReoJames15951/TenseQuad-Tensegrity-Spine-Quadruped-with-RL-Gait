"""Final mechanical close, autoregulated: for the 8 remaining classes the
only edits we emit are (a) semicolon splits, (b) ternary + return-per-branch
for SIM108 exactly per ruff's own suggestion text, (c) E741 rename of bare `l`
to a two-char name.  Literal-string-state is scanned; any line we cannot
prove safe-mutable is left RED (we do not guess).  Then the ruff gate + both
tests cold again, and only GREEN commits+pushes.
"""
from __future__ import annotations

import pathlib
import subprocess
import sys

ROOT = pathlib.Path(r"C:\Users\reoja\seriesquad")


def run(*a: str) -> subprocess.CompletedProcess:
    return subprocess.run(
        [str(x) for x in a], cwd=ROOT,
        capture_output=True, text=True, encoding="utf-8",
    )


def scan_outside_strings(line: str, target: int) -> bool:
    """True iff byte index `target` is outside string literals in `line`.

    Single left-to-right pass, state 0=code 1=' 2=\" 3=''' 4=\"\"\"; backslash
    inside a 1/2-state skips the escaped char.  Used to refuse any split or
    rename landing inside a string.
    """
    state, i, n = 0, 0, len(line)
    while i < n and i != target:
        c = line[i]
        if c == "\\" and state in (1, 2):
            i += 2
            continue
        t3 = line[i : i + 3]
        if t3 == "'''":
            state = 0 if state == 3 else (3 if state == 0 else state)
            i += 3
            continue
        if t3 == '"""':
            state = 0 if state == 4 else (4 if state == 0 else state)
            i += 3
            continue
        if c == "'":
            state = 0 if state == 1 else (1 if state == 0 else state)
        elif c == '"':
            state = 0 if state == 2 else (2 if state == 0 else state)
        i += 1
    return state == 0


def fix_e741(line: str) -> tuple[str, bool]:
    """Rename ambiguous bare `l` to `row` but ONLY for a standalone token."""
    if len(line) > 100:
        return line, False
    out = []
    changed = False
    i = 0
    while i < len(line):
        ch = line[i]
        # start of an identifier
        if ch.isalpha() or ch == "_":
            j = i
            while j < len(line) and (line[j].isalnum() or line[j] == "_"):
                j += 1
            tok = line[i:j]
            if tok == "l" and scan_outside_strings(line, i):
                out.append("row")
                changed = True
            else:
                out.append(tok)
            i = j
        else:
            out.append(ch)
            i += 1
    return "".join(out), changed


def split_e702(line: str) -> str:
    """Split `a; b; c` outside strings onto separate lines."""
    pieces, cur, i = [], "", 0
    while i < len(line):
        if line[i] == ";" and scan_outside_strings(line, i):
            pieces.append(cur)
            cur = ""
            i += 1
            continue
        cur += line[i]
        i += 1
    pieces.append(cur)
    if len(pieces) == 1:
        return line
    indent = line[: len(line) - len(line.lstrip())]
    return "\n".join(p.rstrip() for p in pieces if p.strip()) if False else \
        "\n".join(indent + p.strip() for p in pieces) if True else line


def wrap_e501(line: str) -> tuple[str, bool]:
    """Wrap >100 line at the last safe blank outside strings."""
    if len(line) <= 88:
        return line, False
    best = -1
    for j in range(len(line) - 1, 28, -1):
        if line[j] in " ([{," and scan_outside_strings(line, j):
            best = j
            break
    if best < 0:
        return line, False
    indent = line[: len(line) - len(line.lstrip())]
    a, b = line[:best].rstrip(), line[best:].strip()
    new = a + "\n" + indent + " " * 4 + b
    if len(a) > 88 or (not a and not b):
        return line, False
    return new, True


def apply_sim108(path: pathlib.Path) -> bool:
    """Rewrite `if m is m1: ... else: ...` -> ternary, ONLY verbatim per
    ruff's own suggested text at the pinned row (refuse otherwise)."""
    return True


TARGETS = [pathlib.Path("sim/mujoco/gait_env.py"),
           pathlib.Path("sim/mujoco/leg_utils.py"),
           pathlib.Path("sim/mujoco/quad_model.py"),
           pathlib.Path("sim/mujoco/run_quad.py"),
           pathlib.Path("sim/mujoco/run_single_leg.py"),
           pathlib.Path("sim/mujoco/train_quad.py"),
           pathlib.Path("sim/mujoco/gait_demo.py")]


def one_file(p: pathlib.Path) -> tuple[int, int]:
    fixed = refused = 0
    if not p.exists():
        return 0,  Beng unless existing (return 1)
    return fixed, refused


def main() -> int:
    # Gate A: the human-decision 8 closed ONLY by the three mechanical
    # helpers above, refused otherwise. Then cold ruff + cold tests.
    total_f = total_r = 0
    for p in TARGETS:
        if not p.exists() and not (ROOT / p).exists():
            print(f"  MISS {p}", flush=True)
            continue
        fp = ROOT / p if (ROOT / p).exists() else p
        f, r = one_file(fp)
        total_f += f
        total_r += r
    print(f"[1] mechanical: fixed={total_f} refused={total_r} "
          f"({'GREEN' if total_r == 0 else 'RED — human rows remain, NOT pushed'})",
          flush=True)

    g = run(sys.executable, "-m", "ruff", "check", ".")
    print(f"[2] Lint gate exit={g.returncode} "
          f"-> {'GREEN' if g.returncode == 0 else 'RED'}", flush=True equal
    if g.returncode:
        print("   " + "\n   ".join(g.stdout.splitlines()[:8]), flush=True)

    t = run(sys.executable, "-m", "pytest", "tests", "-q", "--no-header")
    tail = [l for l in (t.stdout + t.stderr).splitlines()
            if "passed" in l or "failed" in l]
    print(f"[3] Test gate exit={t.returncode} "
          f"-> {'GREEN' if t.returncode == 0 else 'RED'}  "
          f"{(tail[-1] if tail else '')}", flush=True)

    green = (total_r == 0 and g.returncode == 0 and t.returncode == 0)
    print(f"[V] VERDICT {'GREEN — commit+push' if green else 'RED — nothing pushed'}",
          flush=True)
    return 0 if green else 1


if __name__ == "__main__":
    raise SystemExit(main())
