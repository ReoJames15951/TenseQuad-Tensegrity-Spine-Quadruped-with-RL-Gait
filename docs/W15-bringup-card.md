# W15 single-leg SEA bring-up card (ONE page, run top-to-bottom)

Pairs 1:1 with sizing-sheet §10 (gates) + §11 (SEA worksheet). Every gate below
has a boxed PASS value taken from the frozen sim envelope (§9). A failed gate =
STOP, record, fix, re-run. Do NOT touch the onboard policy until 4/4 legs pass.

**Input that makes this card valid: `sea_center.json` is empty until row C3
measures it. Everything before that is sim-CAD numbers acting as the DR center
fallback — that is *intentional and gated*, not a shortcut.**

---

## Bench setup (5 min, once)

- Leg-mass share (trunk 2.2 kg + leg 0.55 kg) hangs on the SEA leg via a
  horizontal gantry; leg free to move in x/z only, yaw locked by the gantry.
- Two sensors: trunk z (bounce) + rotor/link angle pair (SEA phase), both at
  ≥ 1 kHz. IMU optional — trunk quat from the MX stand-in.
- Drive source: a BLDC bench motor with its own rotor encoder.

Gate 0 **[buy]**: motor continuous torque ≥ 3 N-m (sizing §3), rotor encoder ≥
14-bit (AS5047P class). If you don't have this, STOP — you cannot measure
phase lag you cannot resolve.

---

## A. Bounce test → k_s (trunk z decay)

1. Lift trunk-mass share ~2 cm above its **static sag height**, release, log z
   for ≥ 2 s. Use cycles 3–8 (skip implant transients).
2. `f_meas = 1 / mean(T_cycle)` for cycles 3–8. Must land **8–15 Hz**.
3. `k_s_env = M_share * (2π f_meas)^2`. Record on row A1.

PASS A: `f_meas ∈ [8, 15] Hz` (sim envelope), z oscillations decayed to < 10%
of the 1st in ≤ 8 cycles (damping sanity). sag < 0.5 cm.

| row | M_share | f_meas | k_s_env | vs sim 40 N/m |
|---|---|---|---|---|
| A1  |  2.75 kg|        |         |  within ±20% |

## B. Sinusoidal drive → d_s (SEA lag/phase)

1. Drive rotor sinusoidally at `f_meas`, ask the **full** ±1.2 rad range, slow
   enough that stance doesn't wind up (50 Hz target → 5 s/sweep).
2. Cross-correlate rotor vs link angle over ≥ 10 s → phase lag `φ` (deg) and
   amplitude ratio `R = A_link / A_rotor`.

PASS B: `φ` monotone increasing with f (drive), SAR curve has no sign flips,
`R < 0.5` at f_meas (foot is a quarter-wave behind — recoil damping real).

| row | f_drive | φ (deg) | R | d_s from φ,R |
|---|---|---|---|---|
| B1  | f_meas  |         |   |  [sim center 0.3] |

## C. Threshold scans (mechanical envelope, not sim tuning)

1. **Backlash gate (§7):** hold rotor fixed, push link ±, measure free play at
   hip OUTPUT (not motor). PASS: total < 1 deg.
2. **Spring travel:** max foot z change under full rotor sweep, static. PASS:
   no clank (joint travel discontinuity) across ±1.2 rad; foot clears floor.
3. **Preload vs gravity:** hang leg static, trunk z. Compare to no-spring sag
   (leg share gravity only). PASS: sag < 0.5 cm under leg mass.

## D. Write the measured SEA pair (the ONLY sim-handoff line in this card)

```json
// sea_center.json — measured center, replaces CAD fallback in gait_env.py
{ "k_s": <A1 k_s_env>, "d_s": <B1 d_s> }
```

Dropping this JSON on the mujoco dir (with `gait_env.QuadGaitEnv._sea_center()`
already wired) re-seeds the DR center with the measured pair. Rerun
`run_onnx.py`: straights stay straight (march gate §9) *with measured SEA*.
Mismatch > 20% = DR center wrong, not "tune it away".

---

## Day-one sign-off (check all)

- [ ] A1 k_s within ±20% of sim 40, sag < 0.5 cm
- [ ] B1 d_s within ±20% of sim 0.3, φ monotone
- [ ] Output backlash < 1 deg
- [ ] `sea_center.json` written, `run_onnx.py` re-run, march gate still straight
- [ ] All four legs individual (mark days: FL/FR/RL/RR) → then rotate for W15+
