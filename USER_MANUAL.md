# ASTRALINK — USER MANUAL

**AI-Based Virtual Camera Tracking System for Coarse Alignment of Mobile FSOC Terminals**

**Project:** SIH PS 26169 — Free-Space Optical Communication (FSOC) Precision Acquisition & Tracking (PAT) Testbed  
**Application:** ASTRALINK  
**Version:** Pre-deployment / Demonstration Build (Stage 5)  
**Installation:** To be completed after final packaging  

---

## Table of Contents

1. [About ASTRALINK](#1-about-astralink)
2. [System Architecture & Workflow](#2-system-architecture--workflow)
3. [System Requirements & Current Launch Procedure](#3-system-requirements--current-launch-procedure)
4. [Authentication Gate](#4-authentication-gate)
5. [Main Application Interface](#5-main-application-interface)
6. [Configuring a Mission](#6-configuring-a-mission)
7. [Beacon Detection Pipeline & Target Signatures](#7-beacon-detection-pipeline--target-signatures)
8. [AI-Assisted Candidate Verification](#8-ai-assisted-candidate-verification)
9. [Target Acquisition & Link Readiness](#9-target-acquisition--link-readiness)
10. [Kalman Filter Tracking & State Machine](#10-kalman-filter-tracking--state-machine)
11. [Pan/Tilt Coarse Alignment Controller](#11-pantilt-coarse-alignment-controller)
12. [Understanding Tracking Error & Boresight Metrics](#12-understanding-tracking-error--boresight-metrics)
13. [Target Loss & Automatic Re-acquisition](#13-target-loss--automatic-re-acquisition)
14. [Disturbance System & Presets](#14-disturbance-system--presets)
15. [Running a Disturbance Test](#15-running-a-disturbance-test)
16. [Performance Monitoring & PS 26169 Requirements](#16-performance-monitoring--ps-26169-requirements)
17. [3D Digital Twin Visualization](#17-3d-digital-twin-visualization)
18. [Benchmark Video Evaluation Path](#18-benchmark-video-evaluation-path)
19. [Reports & Telemetry Analysis](#19-reports--telemetry-analysis)
20. [Step-by-Step Recommended Demonstration Sequence](#20-step-by-step-recommended-demonstration-sequence)
21. [Troubleshooting Guide](#21-troubleshooting-guide)
22. [Important Technical Parameters Reference](#22-important-technical-parameters-reference)
23. [Application Code Modules Map](#23-application-code-modules-map)
24. [User Safety, Operational Scope & Known Limitations](#24-user-safety-operational-scope--known-limitations)
25. [Installation Section — To Be Completed Post-Packaging](#25-installation-section--to-be-completed-post-packaging)
26. [Quick Reference Summary](#26-quick-reference-summary)

---

## 1. About ASTRALINK

ASTRALINK is a high-performance software testbed for virtual camera tracking and coarse alignment of **mobile Free-Space Optical Communication (FSOC) terminals**. 

Mobile optical communication links require extreme pointing accuracy to maintain narrow laser beam alignment between moving terminals (ground stations, maritime vessels, UAVs, or aerospace platforms). ASTRALINK simulates an electro-optical acquisition camera on an FSOC terminal, detects incoming laser beacon signals, filters spatial turbulence and noise using computer vision and AI, tracks target kinematics via a 4-state Kalman filter, and drives closed-loop pan/tilt camera control to maintain optical boresight lock.

### Key Capabilities
* **Dual Operating Modes:** Real-time closed-loop FSOC simulation and independent MP4 benchmark video processing.
* **Closed-Loop PAT Control:** Rate-limited proportional pan/tilt controller with dead-zone compensation.
* **AI-Assisted Candidate Verification:** Light-weight 5-feature logistic classifier (`ai_verifier.py`) for false-positive rejection.
* **State-Driven Tracking Engine:** 6-state Kalman estimator with feed-forward velocity smoothing.
* **Environmental & Platform Stress Testing:** 6 disturbance presets including sensor noise, camera jitter, atmospheric turbulence, fog/haze/rain, and platform motion.
* **3D Digital Twin:** Real-time 3D spatial visualizer for mission geometry and beam cone pointing.
* **Comprehensive Telemetry & Reports:** Automated PDF performance reports, JSON summaries, and 24-column per-frame CSV telemetry logs.

---

## 2. System Architecture & Workflow

ASTRALINK operates on two distinct, independent functional paths sharing the core computer vision and tracking engine:

```
                                  ASTRALINK PLATFORM
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    │                                             │
                    ▼                                             ▼
         LIVE FSOC SIMULATION LOOP                       BENCHMARK VIDEO PATH
                    │                                             │
                    ▼                                             ▼
          Controls & Disturbances                         MP4 Input & Ground Truth
                    │                                             │
                    ▼                                             ▼
        [640x480 Virtual Camera]                      [Video Frame Extraction]
                    │                                             │
                    └──────────────────────┬──────────────────────┘
                                           │
                                           ▼
                            ┌─────────────────────────────┐
                            │    DETECTOR + AI VERIFIER   │
                            │  (CV + 5-Feature Logistic)  │
                            └──────────────┬──────────────┘
                                           │
                                           ▼
                            ┌─────────────────────────────┐
                            │    KALMAN STATE TRACKER     │
                            │   (4-State Pos + Velocity)  │
                            └──────────────┬──────────────┘
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    │                                             │
                    ▼                                             ▼
        PROPORTIONAL CONTROLLER                           BENCHMARK METRICS
        (Pan/Tilt Slew Update)                       (Error, RMSE, FPS, PDF/CSV)
                    │
                    ▼
          [Closed-Loop Feedback]
```

### Complete Processing Cycle (Live Simulation)
1. **Mission Configuration:** User selects target count (1–5), designated signature, motion profile, and disturbance levels.
2. **Target Kinematics & Scene Generation:** Simulator updates true target position and optical focal plane.
3. **Camera Frame Rendering:** Virtual sensor renders 640×480 frame with starfield background and atmospheric effects.
4. **Disturbance Injection:** Sensor noise, jitter, atmospheric attenuation, and platform sway are applied to the raw frame.
5. **Multi-Candidate Detection:** OpenCV pipeline applies Gaussian filtering, thresholding, and contour extraction.
6. **Signature & AI Verification:** Candidates are scored against HSV hue signature ranges and the 5-feature AI verifier.
7. **Kalman State Estimation:** Position $(x, y)$ and velocity $(v_x, v_y)$ are updated; covariance uncertainty is calculated.
8. **Pointing Error Calculation:** Pixel displacement from optical center $(320, 240)$ is determined.
9. **Pan/Tilt Command Execution:** Proportional controller applies gain $K_p = 2.0$ with 0.25 px dead-zone and 5.0°/s slew limits.
10. **State Machine Update:** System evaluates state (`SEARCHING`, `ACQUIRING`, `TRACKING`, `PREDICTING`, `LOST`, `RE-ACQUIRING`).
11. **Telemetry & Visuals:** Radar widget, trend graphs, 3D twin, camera feed overlay, and telemetry cards update at 30 Hz.

---

## 3. System Requirements & Current Launch Procedure

> [!IMPORTANT]
> **Packaging Status:** The final standalone installer package (.exe / MSI) has not yet been finalized. Therefore, installation instructions in this section document the pre-deployment development launch method.

### Recommended System Specifications
* **Operating System:** Windows 10/11 (64-bit)
* **Python Runtime:** Python 3.13 (or Python 3.10+)
* **Dependencies:** `PySide6` (Qt6 binding), `opencv-python` (`cv2`), `numpy`, `matplotlib`
* **Display Resolution:** 1920×1080 or higher recommended

### Current Development / Pre-Deployment Launch Procedure

1. **Open Console / Terminal:** Open PowerShell or Command Prompt.
2. **Navigate to FSOC Directory:**
   ```powershell
   cd "D:\FSOC FOLDER ___\FSOC"
   ```
3. **Launch Application:**
   Run using the configured Python environment:
   ```powershell
   python main.py
   ```
   *Alternatively, if using the local virtual environment:*
   ```powershell
   .\.venv\Scripts\python main.py
   ```

---

## 4. Authentication Gate

To prevent unauthorized configuration changes and support point-in-time test validation, ASTRALINK enforces an **Authentication Gate** on application launch.

```
┌────────────────────────────────────────────────────────┐
│               ASTRALINK SECURITY GATE                  │
│                                                        │
│   Enter Access Key to unlock FSOC Control System       │
│   Password: [ * * * * * * * * ]                        │
│                                                        │
│   Default Access Key: isro@1969                        │
│   [ UNLOCK SYSTEM ]                                    │
└────────────────────────────────────────────────────────┘
```

* **Default Access Password:** `isro@1969` (configured in `auth_config.py` → `APP_PASSWORD`)
* **Behavior:** The simulation timer and controls remain locked until a valid password is entered.
* **Customization:** Modify `APP_PASSWORD` in `auth_config.py` to change authorization credentials.

---

## 5. Main Application Interface

The main interface (`FSOCWindow`) is divided into three primary functional areas:

```
┌───────────────────────────────────────────────────────────────────────────────────┐
│ HEADER BAR: ASTRALINK FSOC | Status: ACTIVE | Target: TGT-01 | Rate: 30 Hz       │
├───────────────────┬─────────────────────────────────┬─────────────────────────────┤
│ LEFT COLUMN       │ CENTER COLUMN                   │ RIGHT COLUMN                │
│ (Control Panel)   │ (Radar & State Visuals)         │ (Camera & Telemetry)        │
│                   │                                 │                             │
│ • Simulation      │ • Virtual FSOC Radar            │ • Virtual Camera Feed       │
│   Controls        │   (Polar Tracking Plot)         │   (640x480 + Overlays)      │
│ • Target & Motion │ • Real-Time Error & Motion      │ • Zoom Controls (+ / -)     │
│   Selectors       │   Trend Graphs                  │ • Live Telemetry Grid       │
│ • Pointing &      │ • Kalman Filter Panel           │   (16 Metric Cards)         │
│   Optics          │   (State, Pos, Vel, Covariance) │ • Kalman Prediction Engine  │
│ • Disturbances &  │ • Link Readiness Panel          │ • Link Readiness Status     │
│   Presets         │ • 3D Digital Twin Pane          │ • Report Center             │
│ • Force Loss /    │ • Navigation: [Page 1: Live]    │   [Instant Snapshot]        │
│   Recovery        │               [Page 2: Reports] │   [Benchmark Video]         │
└───────────────────┴─────────────────────────────────┴─────────────────────────────┘
```

### Key UI Navigation Tabs
* **Page 1 (Live Closed-Loop Tracking):** Primary operational screen containing real-time camera feed, radar widget, trend graphs, disturbance controls, and digital twin pane.
* **Page 2 (Reports & Video Benchmarking):** Dedicated page for loading MP4 benchmark files, reviewing historical test runs, and exporting PDF/JSON/CSV report bundles.
* **Pop-out Dialogs:**
  * **3D Digital Twin Window:** Pop out the 3D space scene for multi-monitor viewing.
  * **Video Analysis Window:** Inspect ground-truth vs. tracked trajectory comparisons.

---

## 6. Configuring a Mission

Follow this sequence to establish a simulation scenario:

### 1. Target & Motion Configuration
* **Active Targets (1–5):** Select the number of targets active in the scene. `TGT-01` is designated for PAT tracking by default; additional targets (`TGT-02` through `TGT-05`) serve as moving decoys.
* **Designated Target Signature:** Choose from 5 distinct visual signatures (`GREEN-SQUARE`, `AMBER-DIAMOND`, `MAGENTA-CIRCLE`, `CYAN-TRIANGLE`, `RED-CIRCLE`).
* **Target Motion Pattern:** Select target path profile:
  * `Straight Line`: Constant linear velocity across focal plane.
  * `Circular`: Orbital trajectory around focal center.
  * `Figure 8`: Lemniscate pattern testing cross-axis acceleration.
  * `Random`: Stochastic Brownian trajectory testing tracker adaptation.
* **Speed Multiplier:** Adjust target speed scaling factor ($0.2\times$ to $3.0\times$).

### 2. Camera & Optics Setup
* **Horizontal FOV:** Default `4.0°` (adjustable from $1.0°$ to $10.0°$). Vertical FOV automatically locks to $3/4$ of horizontal FOV ($3.0°$ default).
* **Beacon Size:** Default `10 px` (adjustable from $5\text{ px}$ to $20\text{ px}$).

---

## 7. Beacon Detection Pipeline & Target Signatures

The detection module (`detector.py`) extracts candidate optical beacons from noisy background frames:

```
[Raw BGR Frame] ──▶ [Grayscale Conversion] ──▶ [Gaussian Blur 3x3, σ=0.65]
                                                          │
[Candidate Features & Bounding Boxes] ◄── [Contours] ◄── [Otsu Binarization]
               │
               ├──▶ [HSV Hue Signature Scoring]
               ├──▶ [5-Feature AI Logistic Verifier]
               └──▶ [Expected Location Proximity Weight]
                               │
                               ▼
                   [Best Candidate (x, y)]
```

### Signature HSV Hue Definitions
| Signature Name | Shape | Hue Range (HSV) | Default Role |
| :--- | :--- | :--- | :--- |
| **GREEN-SQUARE** | Square | $45° \le H \le 85°$ | Primary PAT Target (`TGT-01`) |
| **AMBER-DIAMOND** | Diamond | $8° \le H \le 35°$ | Decoy (`TGT-02`) |
| **MAGENTA-CIRCLE**| Circle | $135° \le H \le 175°$ | Decoy (`TGT-03`) |
| **CYAN-TRIANGLE** | Triangle | $80° \le H \le 110°$ | Decoy (`TGT-04`) |
| **RED-CIRCLE** | Circle | $H \le 8° \lor H \ge 172°$ | Decoy (`TGT-05`) |

---

## 8. AI-Assisted Candidate Verification

To differentiate valid optical beacons from thermal sensor noise, cloud reflections, or solar glint, ASTRALINK integrates a lightweight logistic classifier (`ai_verifier.py`).

### Classifier Specifications
* **Implementation:** Standard 5-feature logistic model with L2 regularization ($l_2 = 0.002$), trained using gradient descent (320 iterations).
* **Dependency Free:** Native NumPy implementation—requires no PyTorch, TensorFlow, or ONNX runtimes.
* **Training Dataset:** 1,800 synthetic candidate samples (900 positive beacon blobs + 900 non-beacon noise artifacts).

### Evaluated Candidate Features ($x_1 \dots x_5$)
1. **Area Ratio ($A / A_{expected}$):** Ratio of candidate blob area to nominal $10\times10\text{ px}$ beacon area.
2. **Aspect Ratio ($W / H$):** Proportional width-to-height ratio (ideal beacon $\approx 1.0$).
3. **Mean Brightness:** Normalized mean intensity of pixels within candidate contour $[0.0, 1.0]$.
4. **Fill Ratio ($A / (W \cdot H)$):** Solitary contour area relative to bounding box rectangle.
5. **Contour Circularity ($4\pi A / P^2$):** Perimeter compactness metric.

> [!NOTE]
> The AI module acts strictly as a **candidate validator** (scoring confidence $[0.0, 1.0]$). Kinematic state estimation and target motion prediction are handled independently by the Kalman tracker.

---

## 9. Target Acquisition & Link Readiness

ASTRALINK prevents false-positive lock declarations by enforcing a strict multi-condition **Link Readiness Evaluator** before declaring the communication link ready:

```
                                [DETECTION EVENT]
                                        │
                                        ▼
                         Is Tracking State = TRACKING?
                                        │ Yes
                                        ▼
                          Is Boresight Error ≤ 10.0 px?
                                        │ Yes
                                        ▼
                            Is Processing FPS ≥ 20.0?
                                        │ Yes
                                        ▼
                            Is AI Score ≥ 0.20 for
                             10 Consecutive Frames?
                                        │ Yes
                                        ▼
                             ╔═════════════════════╗
                             ║  LINK READY (LOCK)  ║
                             ╚═════════════════════╝
```

---

## 10. Kalman Filter Tracking & State Machine

The tracking engine (`tracker.py`) uses a discrete-time **4-state Constant Velocity Kalman Filter**:

$$\mathbf{x}_k = \begin{bmatrix} x \\ y \\ v_x \\ v_y \end{bmatrix}_k$$

### Filter Matrices
* **State Transition ($\mathbf{F}$):**
  $$\mathbf{F} = \begin{bmatrix} 1 & 0 & \Delta t & 0 \\ 0 & 1 & 0 & \Delta t \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix} \quad (\Delta t = 0.0333\text{ s at 30 Hz})$$

* **Measurement Model ($\mathbf{H}$):**
  $$\mathbf{H} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix}, \quad \mathbf{R} = \begin{bmatrix} 1.0 & 0.0 \\ 0.0 & 1.0 \end{bmatrix}$$

* **Process Acceleration Variance ($q$):** Configured to $q = 90000.0\text{ px}^2/\text{s}^4$ to allow quick response during sudden target maneuver.
* **Velocity Feed-Forward Smoothing:** Combines Kalman velocity with a 5-frame median velocity buffer ($60\%\text{ Kalman} + 40\%\text{ median}$) to minimize tracking lag during vibration.

### Tracking State Machine (6 States)

```
                 ┌───────────────────────────────────────────────┐
                 │          TRACKING STATE MACHINE               │
                 └───────────────────────────────────────────────┘

       No detection                                First detection
            │                                             │
            ▼                                             ▼
      ┌──────────┐                               ┌────────────────┐
      │SEARCHING │                               │   ACQUIRING    │
      └──────────┘                               └───────┬────────┘
                                                         │ Confirmed
                                                         │ Measurements ≥ 2
                                                         ▼
                                                 ┌────────────────┐
                                                 │    TRACKING    │◄─────────────┐
                                                 └───────┬────────┘              │
                                                         │ Detection lost        │
                                                         ▼                       │
                                                 ┌────────────────┐              │
                                                 │   PREDICTING   │              │
                                                 └───────┬────────┘              │
                                                         │ Lost frames > 10      │
                                                         ▼                       │
                                                 ┌────────────────┐              │ Re-detected
                                                 │      LOST      │──────────────┘
                                                 └───────┬────────┘  (RE-ACQUIRING)
                                                         │
                                                         ▼
                                                 ┌────────────────┐
                                                 │  RE-ACQUIRING  │
                                                 └────────────────┘
```

---

## 11. Pan/Tilt Coarse Alignment Controller

> [!IMPORTANT]
> **Controller Type:** The pointing controller in `controller.py` is a **proportional control law with rate limiting and dead-zone compensation**. It is **NOT** a PID controller.

### Control Formulas

Given camera center $(x_c, y_c) = (320, 240)$, tracking error $(e_x, e_y)$ is:
$$e_x = x_{predicted} - x_c, \quad e_y = y_{predicted} - y_c$$

1. **Pixel-to-Degree Scaling:**
   $$K_{px\_x} = \frac{\text{FRAME\_WIDTH}}{\text{FOV\_HORIZONTAL}} = \frac{640}{4.0°} = 160\text{ px/°}$$
   $$K_{px\_y} = \frac{\text{FRAME\_HEIGHT}}{\text{FOV\_VERTICAL}} = \frac{480}{3.0°} = 160\text{ px/°}$$

2. **Proportional Error Update (with 0.25 px Dead Zone):**
   $$\theta_{pan\_desired} = \theta_{pan} + \begin{cases} K_p \cdot \frac{e_x}{K_{px\_x}} & |e_x| > 0.25\text{ px} \\ 0 & |e_x| \le 0.25\text{ px} \end{cases}$$
   $$\theta_{tilt\_desired} = \theta_{tilt} + \begin{cases} K_p \cdot \frac{e_y}{K_{px\_y}} & |e_y| > 0.25\text{ px} \\ 0 & |e_y| \le 0.25\text{ px} \end{cases}$$

3. **Slew-Rate Limiting (5.0°/s Max Speed at 30 Hz):**
   $$\Delta \theta_{max} = \frac{\text{MAX\_SPEED}}{\text{UPDATE\_RATE}} = \frac{5.0°/\text{s}}{30\text{ Hz}} = 0.1667°/\text{frame}$$
   $$\theta_{pan\_next} = \theta_{pan} + \text{clamp}\left(\theta_{pan\_desired} - \theta_{pan}, -\Delta \theta_{max}, +\Delta \theta_{max}\right)$$

4. **Angular Range Limits:** Clamped to $\pm 5.0°$ pan and tilt limits.

---

## 12. Understanding Tracking Error & Boresight Metrics

Tracking accuracy measures displacement on the $640\times480$ sensor plane:

* **Horizontal Error ($e_x$):** $x_{target} - x_{boresight}$
* **Vertical Error ($e_y$):** $y_{target} - y_{boresight}$
* **Radial Boresight Error ($e$):** $e = \sqrt{e_x^2 + e_y^2}\text{ (pixels)}$
* **Root Mean Square Error (RMSE):** $\text{RMSE} = \sqrt{\frac{1}{N} \sum_{i=1}^N e_i^2}$

### Metric Interpretation Thresholds
* **Optimal Lock ($e \le 2.0\text{ px}$):** Beacon is centered within fine-pointing alignment envelope.
* **Acceptable Tracking ($2.0\text{ px} < e \le 10.0\text{ px}$):** Within PS 26169 compliance limit.
* **Degraded / High Error ($e > 10.0\text{ px}$):** Caused by extreme vibration, platform sway, or heavy fog.

---

## 13. Target Loss & Automatic Re-acquisition

Target loss occurs when the optical beacon is obscured by cloud cover, atmospheric drop-out, or physical obstruction.

### Target Loss Sequence
1. **Obstruction Event:** Detector fails to find candidate above threshold.
2. **Kalman Prediction Phase (`PREDICTING`):** For frames 1 through 10 ($\le 0.33\text{ s}$), the Kalman filter propagates target position using estimated velocity $(v_x, v_y)$. Position uncertainty grows with each frame.
3. **State Transition to `LOST`:** If candidate remains missing after 10 frames ($>0.33\text{ s}$), system transitions to `LOST`.
4. **Re-acquisition (`RE-ACQUIRING`):** When beacon reappears, detector matches candidate signature and restores state to `TRACKING`.

### Forced-Loss Test Feature
Clicking the **FORCE LOSS** button in the UI temporarily disables beacon rendering for 15 frames ($\approx 0.50\text{ s}$) to demonstrate automatic prediction, loss handling, and sub-second re-acquisition.

---

## 14. Disturbance System & Presets

ASTRALINK implements a multi-tier disturbance engine (`simulator.py`) to stress-test computer vision and tracking robustness.

### Disturbance Presets
1. **Clear:** Zero disturbances (Clean optical baseline).
2. **Sensor Noise:** Gaussian thermal noise ($\sigma = 20\text{ px}$).
3. **Vibration:** Atmospheric turbulence ($10\text{ px}$), camera jitter ($15\text{ px}$), platform vibration ($15\text{ px}$).
4. **Atmosphere:** Atmospheric turbulence ($8\text{ px}$), Dense Fog ($70\%$ opacity).
5. **Full Combined:** Gaussian noise ($\sigma=8$), Jitter ($8\text{ px}$), Platform sway ($8\text{ px}$), Haze ($35\%$).
6. **Extreme PAT:** Gaussian noise ($\sigma=20$), Turbulence ($20\text{ px}$), Jitter ($20\text{ px}$), Platform sway ($20\text{ px}$), Dense Fog ($85\%$).

---

## 15. Running a Disturbance Test

Follow this structured workflow during evaluation:

```
[Start Baseline Simulation] ──▶ [Verify Lock & Note Baseline FPS / Error]
                                              │
                                              ▼
                             [Select Disturbance Preset]
                         (e.g., Vibration or Extreme PAT)
                                              │
                                              ▼
                             [Observe Detector & Tracker Response]
                                              │
                                              ▼
                             [Trigger "Force Loss / Recovery"]
                                              │
                                              ▼
                             [Generate Instant PDF Performance Report]
```

---

## 16. Performance Monitoring & PS 26169 Requirements

The application automatically checks live telemetry against **Point-Statement PS 26169 Evaluation Requirements**:

| Performance Metric | PS 26169 Target | Baseline Performance | Extreme Stress Performance | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Tracking Error** | $\le 10.0\text{ px}$ | $1.89\text{ px}$ avg / $4.47\text{ px}$ max | $11.18\text{ px}$ max (Platform Sway) | **PASS** (Baseline) |
| **Target Loss Rate**| $< 5.0\%$ | $0.00\%$ | $0.00\%$ (Excluding forced loss) | **PASS** |
| **Acquisition Time**| $\le 2.0\text{ s}$ | $0.60\text{ s}$ | $1.15\text{ s}$ | **PASS** |
| **Re-acquisition Time**| $\le 1.0\text{ s}$ | $0.267\text{ s}$ | $0.45\text{ s}$ | **PASS** |
| **Processing Speed**| $\ge 20.0\text{ FPS}$ | $334 - 406\text{ FPS}$ (Engine) | $37 - 52\text{ FPS}$ (Fog / Noise) | **PASS** |

> [!NOTE]
> Under extreme platform motion ($20\text{ px}$ sway) and extreme turbulence ($20\text{ px}$ jitter), maximum transient tracking errors can reach $11.18\text{ px} - 12.17\text{ px}$. This is normal physical behavior under severe disturbance and highlights the operational boundary of coarse alignment.

---

## 17. 3D Digital Twin Visualization

The **3D Digital Twin** (`digital_twin.py`) provides spatial situational awareness of the terminal and target relationship:

* **Spatial Representation:** Renders relative 3D vector positions of tracking terminal and optical target.
* **Optical Beam Cone:** Visualizes the expanding optical FOV beam cone.
* **Disturbance Indicators:** Real-time HUD overlay showing current pitch, roll, jitter, and atmospheric opacity.
* **Pop-Out Support:** Click **Pop Out Twin** to launch an independent high-resolution 3D window (`DigitalTwinWindow`).

---

## 18. Benchmark Video Evaluation Path

The Benchmark Engine (`benchmark_video.py`) allows processing recorded MP4 camera feeds through the identical detection and tracking code:

```
[Load MP4 File] ──▶ [Optional: Select Ground-Truth CSV] ──▶ [Async QThread Processing]
                                                                      │
[Export Tracked MP4 + Telemetry CSV + JSON + PDF] ◄───────────────────┘
```

### Ground-Truth CSV Format
To calculate exact tracking accuracy during video playback, supply a CSV file with columns:
```csv
frame,x,y
1,320.5,240.2
2,321.0,241.0
...
```

> [!IMPORTANT]
> **Honest Metrics Policy:** If no ground-truth CSV is provided (or if CSV is invalid), error metrics are displayed as **"—"** (em-dash), never fabricated as `0.0` or marked as false PASS.

---

## 19. Reports & Telemetry Analysis

ASTRALINK provides automated, exportable performance documentation:

### Available Reports
1. **Instant Telemetry Snapshot:** Click **INSTANT REPORT** in the Report Center to generate an immediate point-in-time PDF summary, JSON state dump, and telemetry CSV.
2. **Benchmark Evaluation Report:** Produced automatically after completing an MP4 video benchmark run.

### Generated Artifact Files (Saved in `reports/` folder)
* `{stem}_benchmark_summary.json`: Detailed machine-readable JSON metrics summary.
* `{stem}_benchmark_summary.csv`: Tabular executive summary.
* `{stem}_tracked_telemetry_{stamp}.csv`: 24-column per-frame telemetry breakdown.
* `{stem}_tracked_{stamp}.mp4`: Rendered MP4 video with target bounding boxes, Kalman trajectory vectors, and status overlay.
* `{stem}_benchmark_report.pdf`: Publication-ready PDF performance report.

---

## 20. Step-by-Step Recommended Demonstration Sequence

Follow this sequence for evaluator demonstrations:

1. **Launch Application:** Execute `python main.py` in terminal.
2. **Authenticate:** Enter access password `isro@1969`.
3. **Verify Clean Baseline:** Confirm target acquisition (`SEARCHING` → `ACQUIRING` → `TRACKING`). Observe baseline tracking error ($< 2.0\text{ px}$) and processing FPS ($> 30\text{ FPS}$).
4. **Demonstrate Closed-Loop Pointing:** Observe virtual camera pan/tilt adjustments centering the beacon.
5. **Apply Disturbance Preset:** Select **Vibration** or **Full Combined**. Observe Kalman velocity filtering stabilizing the position estimate.
6. **Trigger Forced Loss:** Click **FORCE LOSS**. Demonstrate state transition (`TRACKING` → `PREDICTING` → `LOST` → `RE-ACQUIRING` → `TRACKING`) with re-acquisition time $< 0.30\text{ s}$.
7. **Pop-Out 3D Digital Twin:** Click **Pop Out Twin** to present spatial terminal alignment.
8. **Run Video Benchmark:** Switch to Page 2, load a sample MP4 video, and run asynchronous evaluation.
9. **Export PDF Report:** Click **Instant Report** or **Export PDF** to view final compliance metrics.

---

## 21. Troubleshooting Guide

| Issue | Probable Cause | Recommended Solution |
| :--- | :--- | :--- |
| **Application locked / Controls disabled** | Security password not entered | Enter `isro@1969` in the authentication dialog box. |
| **Beacon not detected (`SEARCHING`)** | Incorrect signature or severe fog | Verify designated target signature matches simulated beacon color/shape; reduce atmospheric opacity. |
| **High Tracking Error ($> 10\text{ px}$)** | Extreme platform motion / jitter | Reduce disturbance level or increase controller gain $K_p$. |
| **FPS drops below 20 FPS** | High CPU background load or heavy noise filtering | Close competing background applications; set noise mode to None. |
| **Ground-truth error displays "—"** | No valid CSV loaded during benchmark | Load a valid ground-truth CSV containing `frame,x,y` columns. |
| **Video output missing in benchmark** | OpenCV VideoWriter FourCC mismatch | System automatically falls back to `mp4v`, `avc1`, or `MJPG`. Ensure standard video codecs are installed. |

---

## 22. Important Technical Parameters Reference

### Detection & AI Pipeline
* **Gaussian Kernel:** $3 \times 3$, $\sigma = 0.65$
* **Morphological Kernel:** $3 \times 3$ ellipse
* **AI Model:** 5-feature logistic classifier ($900\text{ pos} + 900\text{ neg}$ samples)
* **Link Readiness Streak:** 10 consecutive stable frames with AI score $\ge 0.20$

### Kalman Filter & Controller
* **State Vector:** $[x, y, v_x, v_y]^T$
* **Measurement Noise Variance ($R$):** $1.0\text{ px}^2$
* **Process Acceleration Variance ($q$):** $90000.0\text{ px}^2/\text{s}^4$
* **Controller Gain ($K_p$):** $2.0$ (configurable $0.1 - 2.0$)
* **Dead Zone:** $0.25\text{ px}$
* **Max Pan/Tilt Rate:** $5.0°/\text{s}$ ($0.1667°/\text{frame}$ at $30\text{ Hz}$)

---

## 23. Application Code Modules Map

| Module Name | Primary Responsibilities |
| :--- | :--- |
| [`main.py`](file:///D:/FSOC%20FOLDER%20___/FSOC/main.py) | Application entry point; creates Qt application and presents main window. |
| [`gui.py`](file:///D:/FSOC%20FOLDER%20___/FSOC/gui.py) | Main graphical user interface (`FSOCWindow`), custom widgets, telemetry grid, and timer loop. |
| [`simulator.py`](file:///D:/FSOC%20FOLDER%20___/FSOC/simulator.py) | Virtual FSOC camera scene, target kinematic generation, and disturbance injection. |
| [`detector.py`](file:///D:/FSOC%20FOLDER%20___/FSOC/detector.py) | Multi-candidate blob detection, HSV color signature matching, and AI verification interface. |
| [`ai_verifier.py`](file:///D:/FSOC%20FOLDER%20___/FSOC/ai_verifier.py) | 5-feature synthetic-trained logistic classifier for false candidate rejection. |
| [`tracker.py`](file:///D:/FSOC%20FOLDER%20___/FSOC/tracker.py) | 4-state constant-velocity Kalman filter and 6-state tracking state machine. |
| [`controller.py`](file:///D:/FSOC%20FOLDER%20___/FSOC/controller.py) | Proportional pan/tilt camera pointing controller with slew-rate limiting. |
| [`digital_twin.py`](file:///D:/FSOC%20FOLDER%20___/FSOC/digital_twin.py) | Real-time 3D spatial visualization widget and pop-out window. |
| [`benchmark_video.py`](file:///D:/FSOC%20FOLDER%20___/FSOC/benchmark_video.py) | Asynchronous MP4 video benchmarking engine (`VideoBenchmarkWorker` QThread). |
| [`performance_report.py`](file:///D:/FSOC%20FOLDER%20___/FSOC/performance_report.py) | PDF, JSON, and CSV performance summary report generation. |
| [`instant_report.py`](file:///D:/FSOC%20FOLDER%20___/FSOC/instant_report.py) | Point-in-time instant telemetry snapshot generator. |
| [`metrics.py`](file:///D:/FSOC%20FOLDER%20___/FSOC/metrics.py) | Pure mathematical helpers for RMSE, tracking error, loss %, and lock retention. |
| [`auth_config.py`](file:///D:/FSOC%20FOLDER%20___/FSOC/auth_config.py) | Local security authentication password configuration. |
| [`config.py`](file:///D:/FSOC%20FOLDER%20___/FSOC/config.py) | System constants, camera defaults, and PS 26169 compliance limits. |

---

## 24. User Safety, Operational Scope & Known Limitations

### Operational Scope
ASTRALINK is a **software-based simulation and coarse-alignment testbed**. It evaluates computer vision algorithms, estimation filters, and control strategies for optical tracking.

### Explicit Exclusions (Not Implemented)
* Real laser propagation physics or atmospheric turbulence power-law phase screens.
* Physical optical link budget, fiber coupling efficiency, or Bit Error Rate (BER) measurement.
* Physical gimbal hardware-in-the-loop (HIL) drivers or optical payload interfaces.

### Documented Limitations
1. **Virtual Resolution:** Shared 640×480 camera coordinate space (configurable $\ge 2000\times2000$ virtual scene is deferred).
2. **Extreme Stress Deviations:** Severe platform sway ($20\text{ px}$) and turbulence ($20\text{ px}$) transiently exceed the $10.0\text{ px}$ error target.

---

## 25. Installation Section — To Be Completed Post-Packaging

*(This section will be updated upon release of the standalone installer package)*

```
System Requirements:
• OS: Windows 10/11 64-bit
• Memory: 8 GB RAM minimum
• Storage: 500 MB free space
• Graphics: OpenGL 2.1+ compatible GPU

Installation Steps:
1. Run ASTRALINK_Setup_v1.0.exe
2. Follow desktop installer wizard instructions
3. Launch ASTRALINK from Start Menu / Desktop shortcut
```

---

## 26. Quick Reference Summary

* **Launch Command:** `python main.py` (in `FSOC` directory)
* **Access Password:** `isro@1969`
* **Control Loop Rate:** 30 Hz (33 ms QTimer tick)
* **Camera Resolution & FOV:** 640×480 px, $4.0° \times 3.0°$ FOV
* **Controller:** Proportional ($K_p = 2.0$, 0.25 px dead-zone, 5.0°/s slew limit)
* **Tracking Filter:** 4-State Constant Velocity Kalman Filter ($x, y, v_x, v_y$)
* **Evaluation Targets:** Error $\le 10\text{ px}$, FPS $\ge 20$, Loss $< 5\%$, Acq $\le 2\text{ s}$, Re-acq $\le 1\text{ s}$
* **Key State Progression:** `SEARCHING` → `ACQUIRING` → `TRACKING` → `PREDICTING` → `LOST` → `RE-ACQUIRING`
