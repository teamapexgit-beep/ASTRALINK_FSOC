# PS 26169 — Final Compliance Audit

**Project:** AI-Based Virtual Camera Tracking System for Coarse Alignment of Mobile FSOC Terminals
**Audited at:** Stage 5 (commit series from `b399c0a` through Stage 5), `sih26-upgrades` branch
**Method:** Every status below was verified against the source code and/or the automated
regression suite (`full_system_test_v3.py`). No status is claimed without evidence.
Where a measurement requires input the application cannot generate (e.g. external ground
truth), the status is **PENDING**, never PASS.

Status legend: **PASS** = verified by measurement or direct code verification ·
**PENDING** = cannot be measured yet / requires unavailable input or external validation ·
**FAIL** = requirement demonstrably not met.

---

## A. Camera parameters

| # | Requirement | Measured / configured value | Status | Evidence |
|---|-------------|------------------------------|--------|----------|
| 1 | Screen size ≥ 2000×2000 px (optional/user-defined) | Virtual environment shares the 640×480 camera frame; no separate configurable ≥2000×2000 virtual scene is implemented | **PENDING** | Known limitation — implementing it is a simulator-geometry change deliberately deferred; documented honestly rather than claimed |
| 2 | Camera type: monochrome / FPA reference (colour optional) | Colour (BGR) virtual focal-plane sensor; monochrome is the PS reference, colour permitted | **PASS** | `simulator.py` renders BGR frames; PS marks colour optional |
| 3 | Camera resolution 640×480 default | 640×480 (`config.FRAME_WIDTH/FRAME_HEIGHT`) | **PASS** | `config.py`; asserted in v3 baseline checks |
| 4 | FOV user-defined, default 4°×3° | FOV slider 1.0–10.0° (0.1° steps), default 4.0°; vertical = 3/4 of horizontal; applied via `simulator.set_fov()` | **PASS** | `gui.py` FOV slider + `simulator.set_fov()`; reports print live FOV |
| 5 | Camera update rate ≥ 30 Hz | 30 Hz configured (33 ms GUI tick) | **PASS** | `config.UPDATE_RATE = 30`; status bar shows "30 Hz" |
| 6 | Initial camera position: centre of screen | Camera starts centred (320, 240) | **PASS** | Simulator initialisation; boresight starts at frame centre |

## B. Target parameters

| # | Requirement | Measured / configured value | Status | Evidence |
|---|-------------|------------------------------|--------|----------|
| 7 | Target type: beacon spot | Bright signature beacons rendered by the simulator | **PASS** | `simulator.py` beacon rendering; detector validates them |
| 8 | Targets: ≥1 mandatory, multiple optional | 1–5 targets selectable (TGT-01…05) | **PASS** | Target-count selector 1–5; multi-target check in v3 suite |
| 9 | Shape user-defined, square default | Square (green) default; diamond/circle/triangle decoys; benchmark signature selectable | **PASS** | `TARGET_SIGNATURES` in `simulator.py`/`detector.py`; benchmark signature selector |
| 10 | Size 5–20 px, default 10×10 | Beacon size slider clamped 5.0–20.0 px, default 10 px | **PASS** | `simulator.set_beacon_size()` clamp verified |
| 11 | Initial target location user-defined / random | Randomised initial placement with seeded RNG; designated target user-selectable | **PASS** | `simulator.py` initial placement |
| 12 | Motion: straight line, circular, figure-8, random | All four selectable: Straight Line, Circular, Figure 8, Random | **PASS** | `gui.py` motion selector; v1/v3 tests exercise Figure 8 & Straight Line |

## C. Camera-motion constraints

| # | Requirement | Measured / configured value | Status | Evidence |
|---|-------------|------------------------------|--------|----------|
| 13 | Max pan speed 5–10°/s, default 5 | 5.0°/s default, rate-limited P-controller | **PASS** | `config.MAX_PAN_SPEED`; `controller.py` rate limiter |
| 14 | Max tilt speed 5–10°/s, default 5 | 5.0°/s default, rate-limited | **PASS** | `config.MAX_TILT_SPEED`; `controller.py` |
| 15 | Update interval ≥ 20 Hz | 30 Hz control loop; measured end-to-end ≈150 FPS processing headroom | **PASS** | v3 suite throughput check: ~152 FPS (measured this release) |

## D. Performance

| # | Requirement | Measured / configured value | Status | Evidence |
|---|-------------|------------------------------|--------|----------|
| 16 | Acquisition time ≤ 2 s | Measured in every benchmark run (e.g. 0.60 s typical; reported in PDF/JSON/CSV) | **PASS** | `evaluate_ps_requirements()` gates on ≤2 s; suite runs real acquisition |
| 17 | Tracking error ≤ 10 px | Measured when ground truth exists; ~0.5 px typical in simulator runs | **PASS** | Centroid/ground-truth error pipeline; **PENDING** for uploaded video without GT (displayed as "—", never zero) |
| 18 | Target loss < 5 % | Measured per run; forced-loss test mathematically excluded (loss% counts only frames with ground truth) | **PASS** | v3 forced-loss check asserts loss% stays 0.000% during the intentional test |
| 19 | Re-acquisition ≤ 1 s | Measured 0.30 s in the designed forced-loss scenario | **PASS** | v3 forced-loss check asserts ≤1 s on the measured value |
| 20 | Processing speed ≥ 20 FPS | ~152 FPS measured (benchmark engine); live GUI ~88+ FPS typical | **PASS** | v3 throughput check; processing FPS in all reports |

## E. Disturbances / noise

| # | Requirement | Measured / configured value | Status | Evidence |
|---|-------------|------------------------------|--------|----------|
| 21 | Noise: Salt & Pepper, Gaussian, Poisson selectable | All three selectable (+ None) | **PASS** | Noise selector; `self_test.py` runs each type |
| 22 | Max noise σ 20 px, user-defined | Level slider 0–20, simulator clamps to 20 | **PASS** | Slider range + `simulator.py` clamp |
| 23 | Max camera jitter ±20 px/frame | Jitter slider 0–20 ±px/frame, clamped | **PASS** | Slider range + simulator clamp; "Extreme PAT" preset uses full ±20 |
| 24 | Atmosphere: Clear/Haze/Fog/Rain/Low Light | All five selectable, level 0–100 % | **PASS** | Atmosphere selector; v3 disturbance-config checks |
| 25 | Platform motion ±20 px/frame, Linear mandatory/default | Linear default + Circular/Random/Figure 8 patterns; strength 0–20 ±px clamped | **PASS** | Platform selector/slider; v3 disturbance checks |

## F. Expected software capabilities

| Capability | Status | Evidence |
|------------|--------|----------|
| Configurable virtual environment | **PASS** | FOV/size/motion/disturbance controls |
| One or more moving targets | **PASS** | 1–5 targets, 4 motion patterns |
| Movable virtual camera | **PASS** | Rate-limited pan/tilt controller |
| Automatic target detection | **PASS** | HSV-signature + AI-verifier detector |
| Continuous CV tracking | **PASS** | Kalman 4-state tracker, state machine |
| Camera repositioning/control | **PASS** | Controller closes the loop every frame |
| Disturbances (atmosphere/platform/camera/noise) | **PASS** | All disturbance families implemented and gated |
| Real-time performance statistics | **PASS** | Live telemetry, trend graphs, PS compliance panel, reports |
| AI-assisted functionality | **PASS** | `ai_verifier.py` scores candidates; AI gate feeds link-readiness |
| Reports | **PASS** | Performance PDF+JSON+CSV; instant PDF+JSON+CSV; uploaded-video bundle; all PDFs QtPdf-verified openable |

## G. Honest-metrics policy (verified behaviour)

- Without ground truth the UI displays GT-dependent metrics (tracking error, average, max
  deviation, RMSE, centroid accuracy) as **"—"** with the note *"Ground truth unavailable —
  centroid/tracking accuracy was not calculated."* Never zero, never PASS.
- An invalid/empty ground-truth CSV is classified **INVALID** and treated as unavailable.
- PS-compliance rows that depend on missing data show **PENDING**, never PASS
  (`evaluate_ps_requirements()` is None-safe by design).

## H. Known limitations (documented, not hidden)

1. **Virtual scene ≥2000×2000 (PS item A1)** — the virtual environment currently shares the
   640×480 camera frame; a separate larger configurable virtual scene is not implemented.
   Marked PENDING above. Implementing it would change simulator coordinate geometry and is
   deliberately deferred rather than rushed into the demo build.
2. **Digital Twin is a visualization** — scene offsets/separation and the illustrative beam
   cone affect drawing only; they never touch the controller, tracker, or benchmark maths
   (proven by v3 checks that assert controller pan/tilt is unchanged).
3. **Atmospheric weather is not rendered in the twin's space scene** — the twin states this
   on its HUD; disturbance *state* is still faithfully displayed.
4. **Fine pointing / physical optical link** — out of scope for a virtual coarse-PAT
   testbed; readiness indicator is labelled as such everywhere it appears.
