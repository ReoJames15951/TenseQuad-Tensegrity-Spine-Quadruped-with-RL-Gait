# Sizing Sheet — Compliant Series-Elastic Quadruped

DRAFT — every number below is a *target to be re-validated* by the single-leg MuJoCo
model (`sim/mujoco/`) and the physical single-leg test rig. Treat DR (domain
randomization) ranges as the source of truth for the sim, this doc as the source of
truth for hardware intent.

Status legend: **[sim]** to confirm in sim, **[rig]** to confirm on the test rig,
**[buy]** outside-of-simulation procurement decision.

---

## 0. Mission targets

| Item | Target | Notes |
|---|---|---|
| Robot mass (incl. battery) | ~4 kg | 3.5–5 kg design band |
| Speed | 0.8–1.5 m/s trot | steady-state metric |
| Runtime | ~20 min | battery-driven |
| DOF | 4 legs x 2 = 8 actuated | hip-mount only, no knee motor |
| Compliance | SEA per input link | spring in drivetrain, not foot |
| Controller | PPO, proprioceptive only, ONNX on onboard | 50 Hz policy / 500 Hz PD |

## 1. Mass budget

| Subsystem | Mass (kg) | Split |
|---|---|---|
| Torso structure | 1.1 | frame, mounts, fasteners |
| Battery (4S LiPo ~2.2 Ah) | 0.22–0.25 | [buy] |
| IMU + MCU + Jetson Orin Nano | 0.15 | tolerance wide |
| Wiring, connectors, harness | 0.10 | |
| 4x leg assemblies (links, SEA, actuators) | 4 x 0.55 = 2.2 | sec. 4 |
| **Total** | **~4.0** | |

Per-leg budget: proximal links 2x0.045, couplers 2x0.035, foot/tip 0.025,
hip frame + motor stators 0.35 → **~0.55 kg/leg**.

Leg **inertia** target (drives controller bandwidth & impact losses):
kept low because actuators are at the hip — foot + couplers should weigh
<= 0.10 kg combined. **[rig] weigh each part.**

## 2. Leg geometry — crossed five-bar (parallel)

Parameters (values provisional, see `sim/mujoco/leg_utils.py` for the FK):

| Sym | Value | Meaning |
|---|---|---|
| a | 0.060 m | hip mount separation (ground link) |
| L1 | 0.090 m | proximal link (input crank) |
| L2 | 0.110 m | coupler link |
| max reach | ~0.196 m | straight-down pose (L1 + sqrt(L2^2-(a/2)^2)) |
| joint limits | +/-1.2 rad (r, j) | rotor & spring-loaded link |
| coupler limits | +/-1.6 rad | keeps elbow out of fold singularity |

Constraints to verify in sim `workspace` mode:
- Foot reachable region fully x-symmetric; depth range and accessible x-extent
  must cover one trot stride (fwd ~0.10 m, back ~0.05 m at ~60% of max depth).
- Singularity margins: keep stance configs away from `|E1-E2| -> 2*L2` (leg fully
  extended = along-line singularity) and `|E1-E2| -> 0` (elbows collapsed).
  The policy must never be allowed to command into those zones — enforce in
  action limits + safety clipping on hardware. **[sim] map margins, add to obs.**

Why crossed five-bar, not two serial links: motors at the hip → low leg inertia →
higher SEA bandwidth and less energy dumped at touchdown. This is the dynamics
requirement, not styling.

## 3. Torque ladder

| Load case | Per-leg force | Worst-case input torque | Basis |
|---|---|---|---|
| Static stance | ~10 N | -- | 4 kg / 4 legs |
| Trot dynamic | 20–35 N | 4–7 N-m | 2–3x load factor |
| SEA peak (impact + drive) | >40 N | 7–9 N-m | momentum + spring preload |
| **Design target** | -- | **8–10 N-m peak, ~40% duty continuous** | margin over worst |

Motor class **[buy]**: 4x small 50–100 W BLDC (e.g. T-Motor GL / MyActuator RMD or
custom outrunners), belt/planetary reduction 6–12:1 into each input. Selection
triggers:
- continuous torque at ~40% duty >= 3 N-m at output
- rotor reflected inertia (via reduction) such that leg SEA natural frequency lands
  in 8–15 Hz (sec. 4)
- encoder on **output** shaft (AS5047P-class, 14-bit) — not only rotor, else spring
  deflection is unobservable
- 2x encoders per leg (rotor + link) required for SEA torque estimate

Reflected inertia rule of thumb: `J_rotor_eff = J_rotor * N^2`; keep
`J_rotor_eff` comparable to link inertia (~1–4e-3 kg-m^2 at each input) so the
spring mode is well-damped and observable. **[sim] verify with armature value in
XML; [rig] measure J_rotor via spin-down test.**

## 4. Series-elastic element

Transmission: motor (rotor) -> torsional spring-damper -> input link. Modeled in
MuJoCo as a `fixed` tendon coupling rotor joint and link joint (equal-and-opposite
torque = true SEA).

| Qty | Candidate | Foot-stiffness equiv | Natural freq (est.) | Use |
|---|---|---|---|---|
| k_s | 40 N-m/rad | ~2.7–4 kN/m | ~14 Hz | baseline sprint/trot |
| k_s | 20 N-m/rad | ~1.3–2 kN/m | ~10 Hz | soft / rough ground |
| k_s | 70 N-m/rad | ~4.5–6 kN/m | ~18 Hz | stiff, high-power |
| d_s | 0.3–0.5 N-m-s/rad | -- | zeta ~0.1–0.2 | underdamped (bounce) |

Selection rule: springed natural frequency = 1.5–2x max stride fundamental
(3–5 Hz for trot -> target 8–15 Hz). Build the rig with **swappable springs** at
least in {20, 40, 70} set.

Rig procedure **[rig]**:
1. Bounce test: drop leg-share mass on the springed leg, measure rebound
   frequency -> spring foot stiffness.
2. Drive the rotor sinusoidally, record link lag -> identify k_s, d_s, and any
   static sag (spring preload vs. gravity).
3. Feed measured values back into XML + DR center (`spring_stiffness`,
   `spring_damping`) — never sim with CAD numbers.

Watch: too-soft spring => gravity sag + winding creep at stance; too stiff =>
impacts inject into frame. Both are tuning gates in the roadmap.

## 5. Control / sensing bill

| Item | Choice | Why |
|---|---|---|
| Rotor encoder | AS5047P-class, 14-bit | SEA torque est needs rotor+link angle |
| Link/output encoder | AS5047P-class | spring deflection = rotor - link |
| IMU | ICM-42688-P / BMI088 | low drift gyro, accelerometer for gravity vector |
| MCU | Teensy 4.x / STM32F4+ | runs 500 Hz PD + safety clipping onboard |
| Compute | Jetson Orin Nano (Nano EOL; laptop offboard = fallback) | PPO MLP ONNX <5 ms @ 50 Hz |
| Comm | UART/SPI MCU<->Jetson + radio to laptop | telemetry, kill switch |
| Power | 4S LiPo | 8 x BLDC peak ~40 A; keep ESC rating >= 50 A |

## 6. Domain randomization -> hardware mapping

| Sim (uniform) | Range | Hardware it protects |
|---|---|---|
| foot friction | 0.3–1.4, restitution 0–0.4 | floor material variance |
| motor gain | x(0.9–1.1) | thermals, production spread |
| motor damping | x(0.7–1.3) | bearing/belt friction |
| control delay | 0–2 steps (0–4 ms) | comm + dropout |
| torque limit | x(1.1) of nominal | ESC thermal sag |
| k_s, d_s | x(0.8–1.2) | spring tolerance |
| link mass, CoM | +/-20%, +/-5% len | printing/foam variance |
| IMU/joint noise | std + bias drift | real sensors |
| payload | 0–0.3 kg | accessory later |
| pushes | random impulses | disturbance rejection |

## 7. Open decisions — gate to confirm in sim, then rig

- [ ] Leg detect location & exact SEA anchor that keeps equal-and-opposite span loading symmetric.
- [ ] Singularity margins (sec. 2) — set action limits.
- [ ] PD frequency/value split vs. learned gains (start explicit PD, randomize in sim).
- [ ] Spring preload effect on stance torque bias — compensate in control or accept in policy.
- [ ] Battery placement for CoM within each leg-plane sagittal footprint.
- [ ] Backlash budget: belt tensioning vs. planetary — keep total angular backlash < 1 deg measured at output. **[rig]**

## 8. Six-month phase map (condensed)

- W1–2: freeze this sheet (all green = go)
- W3–9: single-leg rig + sim model validated against it (THIS module is W1–4)
- W10–14: PPO + DR, flat -> rough -> pushes
- W15–22: full robot, PD bring-up, closed-loop laptop -> onboard
- W23–24: Jetson port + robustness + buffer

## 9. W10-14 sim validation log (throwaway learnings, not targets)

**DECISION (a) — LOCKED 2026-09-17: the deployment envelope is frozen as
march-in-place/rotary crawl + commandable rotary hip-yaw turn. Forward-cruise
is closed at the topology level (NO-GO below); no further sim pursuit.**

- **Anti-yaw/speed litmus (popper-grade, closes the "reward hack vs real
  mechanism" question):** punish instantaneous trunk yaw-rate greedily ->
  policy collapses to a straight march-in-place (yaw 0.0°, z 0.226, DR-robust)
  and the `--turn` knob goes DEAD. Remove the yaw penalty -> same policy turns
  direction-commandably via phase-bias crawl (~±4 deg/s). The rotary crawl IS a
  genuine commandable turn mechanism (not mere curve-fitting); straight-drive
  and turn draw on the SAME body-yaw channel (SEA five-bar) and cannot coexist
  without a physical turning actuator.
- **Forward drive: mechanically precluded (NO-GO).** Across 5 seeds + 4 configs
  (stride 0.03-0.06, heading ±0.18 rad, DR on/off) the spring-reciprocated
  crossed-five-bar has NO +x attractor that outruns its own recoil. Do not
  pursue forward-cruise in this leg topology; march-in-place/walking + rotary
  hip yaw is the honest envelope.
- **Sim-to-real bridge: BIT-EXACT.** `quad_ppo.onnx` (opset 13) free-runs in
  onnxruntime and matches the numpy policy to max |dx|/|yaw| = 0.0 over 10 s
  live episodes (`run_onnx.py`). The ONNX blob is the embedded artifact; no
  numpy/pytorch needed on target.

- Ship artifacts in `sim/mujoco/`: `quad_ppo.npz` (checkpoint),
  `quad_ppo.onnx` (bit-exact bridge), `export_onnx.py`, `run_onnx.py`.
- **[rig] feedback:** single-leg SEA identify (kappa_s, d_s), feed back into
  XML + DR center (never ship sim with CAD spring numbers).

## 10. W15+ rig bring-up card (no sim until each row passes)

Numbers in brackets are the frozen sim gates (sec. 5/9). Run top-to-bottom;
a failed gate = STOP, record, fix, re-run. Do not touch the quad policy
(onboard/walking bring-up = W16+ only after 4/4 legs tick).

1. Weigh parts -> fill elastic-mass cells (sec. 1). PASS: trunk 0.22–0.26 kg,
   per-leg 0.50–0.60 kg, foot+coupler <= 0.10 kg.
2. Hip-yaw encoder plane check: leg fires SEA march (no turn command) =>
   trunk yaw drift <= +-8 deg / 10 s at rig. **[from sim: expect ~0 deg]**
3. Drive one leg sinusoidally 3–5 Hz -> measure link lag + sag (sec. 3).
   PASS: spring nat-freq lands 8–15 Hz; sag < 0.5 cm under leg mass; no
   joint-travel clank across full rotor sweep.
4. Backlash gate (sec. 7): total angular backlash measured at hip output <
   1 deg. Belt tensioning > planetary if over. **[rig] measure at output, not
   motor.**
5. SEA identify (sec. 4): bounce + sinusoidal-drive -> kappa_s, d_s. Feed the
   measured pair into `gait_env.py` DR center — never sim with CAD springs.
6. Only now run `train_quad.py --eval-only`: offline-displacement and yaw
   should match the frozen gate (0.0 m march, yaw near 0) within ~20%.
   Mismatch > 20% = DR center wrong, not "tune it away".

## 11. W15 SEA-identification worksheet (one page, fill on the bench)

Goal: turn measured bounce/phase numbers into `(kappa_s, d_s)` that re-seed the
XML + DR center. Run in this order; every row = one measurement, one number.

**A. Bounce test (foot stiffness).** Drop the leg-share mass M_bounce on the
springed leg from ~2 cm, log trunk z. Count 8 rebounds, use cycles 3-8 only
(skip the first two as implant transients). f_meas = 1 / mean(period).

Row | M_bounce (kg) | f_meas (Hz) | kappa_s = M*(2*pi*f)^2 (N/m) | g-for-gate
----|----|----|----|----
A1  |               |             |                             | [sim] 8-15 Hz

**B. Sinusoidal drive (SEA lag + damping).** Drive the rotor sinusoidally at
f_meas/2. Record rotor->link phase lag phi (deg) and amplitude ratio R =
A_link/A_rotor.

Row | f_drive (Hz) | phi (deg) | R | d_s = f(kappa_s, phi, R) (N-m-s/rad)
----|----|----|----|----
B1  |              |           |   |

Use the DR-center relation (single-leg SEA LOS): at f_drive the measured
phi/R pairs map to (kappa_s, d_s) through the published DR-center formulas;
f = kappa_s/M_link^2, zeta = d_s/(2*sqrt(kappa_s*M_link)), and the closed-form
for R = 1/sqrt((1-(f/f_n)^2)^2 + (2*zeta*f/f_n)^2).

**C. Static sag.** Hang the leg static, measure trunk z drop from preload.
z_sag (cm) => spring preload vs gravity check (gate 3, < 0.5 cm). Works with
A/B to confirm kappa_s.

**D. Round-trip.** Feed kappa_s, d_s into `gait_env.py` DR center constants;
re-run `run_onnx.py` — expect |dx|/|yaw| still ~0.0 (march gate) with the
measured springs. If the march breaks > 20%, the DR center was wrong, not the
gait.

Watch: **[rig] always measure with real link mass (M_link from §3), never CAD.**
Errors here are DR-center errors and will show up as yaw/march drift on the
full robot — this sheet is the single source of truth for the spring numbers.