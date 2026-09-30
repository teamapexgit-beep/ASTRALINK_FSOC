# ASTRALINK FSOC — Technical Architecture & Stack

## Executive Summary

ASTRALINK is an AI-based virtual camera tracking system for mobile Free-Space Optical Communication (FSOC). It implements Precision Acquisition and Tracking (PAT) using computer vision, Kalman filtering, and closed-loop camera control in a real-time Qt-based desktop application.

---

## Technology Stack

### Core Framework & Language
| Component | Technology | Version/Details |
|-----------|-----------|-----------------|
| **Language** | Python 3.13 | Primary implementation language |
| **GUI Framework** | PySide6 (Qt for Python) | Qt6 binding, QMainWindow, QDialog, custom widgets |
| **Computer Vision** | OpenCV (cv2) | Video I/O, image processing, frame analysis |
| **Numerical Computing** | NumPy | Array operations, random number generation, matrix ops |
| **Data Serialization** | CSV, JSON | Telemetry logging, performance reports |
| **PDF Generation** | (via performance_report module) | Report export |

### Architecture Pattern
- **MVC-inspired** (Model-View-Controller) with clear separation:
  - **Model**: `Simulator`, `BeaconTracker`, `CameraController`, `detect_beacon()`
  - **View**: Custom Qt widgets (`RadarWidget`, `TrendGraph`, `CameraFeedWidget`, `BenchmarkScreen`)
  - **Controller**: `FSOCWindow` (main window), event handlers, timer-driven update loop

### Real-Time Architecture
- **Update Loop**: QTimer at 33ms interval (~30 Hz) calling `_safe_update_simulation()`
- **Threading**: 
  - Main thread: GUI rendering, simulation update
  - Worker thread: `VideoBenchmarkWorker` (QThread) for benchmark video processing
- **Signal/Slot**: Qt signals for cross-thread communication (progress, completion, errors)

---

## System Architecture — Dual-Path Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                    ASTRALINK FSOC PLATFORM                          │
│            AI-Based Virtual Camera Tracking System                  │
└─────────────────────────────┬───────────────────────────────────────┘
                              │
          ┌───────────────────┴───────────────────┐
          │                                       │
          ▼                                       ▼
 ┌─────────────────────┐               ┌─────────────────────┐
 │  USER / OPERATOR    │               │  USER / OPERATOR    │
 └──────────┬──────────┘               └──────────┬──────────┘
            │                                     │
            ▼                                     ▼
 ╔═══════════════════════════╗         ╔═══════════════════════════╗
 ║   LIVE FSOC SIMULATION    ║         ║   BENCHMARK VIDEO         ║
 ║        LOOP               ║         ║   EVALUATION PATH         ║
 ╚═══════════════════════════╝         ╚═══════════════════════════╝
            │                                     │
            ▼                                     ▼
 ┌─────────────────────────┐           ┌─────────────────────────┐
 │     USER INTERFACE      │           │       MP4 INPUT         │
 │                         │           │                         │
 │ PySide6 / Qt6           │           │ User / Benchmark Video  │
 │ Python 3.13             │           │ (any .mp4 file)         │
 │                         │           └────────────┬────────────┘
 │ • Controls              │                        │
 │ • Radar Visualisation   │                        ▼
 │ • Live Telemetry        │           ┌─────────────────────────┐
 │ • Trend Graphs          │           │    BENCHMARK ENGINE     │
 │ • 3D Digital Twin       │           │                         │
 │ • Reports               │           │ Python + OpenCV         │
 └────────────┬────────────┘           │ Qt QThread Worker       │
              │                        │ (VideoBenchmarkWorker)  │
              ▼                        │                         │
 ┌─────────────────────────┐           │ • Async frame-by-frame  │
 │   FSOC SIMULATION       │           │   processing            │
 │      ENGINE             │           │ • Progress signals      │
 │                         │           │ • Optional ground-truth │
 │ Python + NumPy + OpenCV │           │   CSV ingestion         │
 │ (simulator.py)          │           └────────────┬────────────┘
 │                         │                        │
 │ • Beacon Generation     │         ┌──────────────┤
 │ • Motion Modelling      │         │  ╔═══════════════════════╗
 │ • Virtual Camera        │         │  ║    SHARED DETECTION   ║
 │ • Noise & Turbulence    │         │  ║    & TRACKING         ║
 │ • Platform Disturbance  │         │  ║    PIPELINE           ║
 │ • Camera Jitter         │         │  ╚═══════════════════════╝
 └────────────┬────────────┘         │              │
              │                      │              │
              ▼                      │              ▼
 ┌─────────────────────────┐         │  ┌─────────────────────────┐
 │   SIMULATED CAMERA      │         │  │    TARGET DETECTION     │
 │        FRAME            │         │  │                         │
 │                         │         │  │ Python + OpenCV         │
 │ 640 × 480 / Configurable│         │  │ (detector.py)           │
 │           FOV           │         │  │                         │
 └────────────┬────────────┘         │  │ • Image Processing      │
              │                      │  │ • Beacon Detection      │
              ▼                      │  │ • Signature Matching    │
 ┌─────────────────────────┐         │  │ • Confidence Scoring    │
 │    TARGET DETECTION     │         │  │ • AI-Assisted           │
 │                         │         │  │   Candidate Verif.      │
 │ Python + OpenCV         │         │  │                         │
 │ (detector.py)           │         │  │    KALMAN TRACKER       │
 │                         │         │  │                         │
 │ • Image Processing      │         │  │ Python + NumPy          │
 │ • Beacon Detection      │         │  │ (tracker.py)            │
 │ • Signature Matching    │         │  │ • Position Estimation   │
 │ • Confidence Scoring    │         │  │ • Velocity Estimation   │
 │ • AI-Assisted           │         │  │ • Motion Prediction     │
 │   Candidate Verif.      │         │  │ • Uncertainty Estim.    │
 └────────────┬────────────┘         │  │ • Tracking State        │
              │                      │  └────────────┬────────────┘
              ▼                      │               │
 ┌─────────────────────────┐         │               ▼
 │     KALMAN TRACKER      │         │  ┌─────────────────────────┐
 │                         │         │  │   BENCHMARK METRICS     │
 │ Python + NumPy          │         │  │                         │
 │ (tracker.py)            │         │  │ • Centroid Error        │
 │                         │         │  │ • Tracking Error        │
 │ • Position Estimation   │         │  │ • RMSE                  │
 │ • Velocity Estimation   │         │  │ • FPS                   │
 │ • Motion Prediction     │         │  │ • Acquisition Time      │
 │ • Uncertainty Estim.    │         │  │ • Re-acquisition Time   │
 │ • Tracking State Machine│         │  │ • Target Loss           │
 └────────────┬────────────┘         │  │ • Lock Retention        │
              │                      │  └────────────┬────────────┘
              ▼                      │               │
 ┌─────────────────────────┐         │               ▼
 │  COARSE ALIGNMENT       │         │  ┌─────────────────────────┐
 │     CONTROLLER          │         │  │   BENCHMARK OUTPUTS     │
 │                         │         │  │                         │
 │ Python (controller.py)  │         │  │ • CSV (telemetry,       │
 │                         │         │  │   summary)              │
 │ • Pan / Tilt Control    │         │  │ • JSON summary          │
 │ • Proportional Control  │         │  │ • Tracked MP4           │
 │ • Dead Zone (0.25 px)   │         │  │ • PDF Performance       │
 │ • Slew-Rate Limiting    │         │  │   Report                │
 └────────────┬────────────┘         └──└─────────────────────────┘
              │
              ▼
 ┌─────────────────────────┐
 │   CAMERA / POINTING     │
 │      ADJUSTMENT         │
 │ simulator.update_camera │
 │      (pan, tilt)        │
 └────────────┬────────────┘
              │
              │  CLOSED-LOOP FEEDBACK
              └──────────────────────────────────────┐
                                                     │
                                                     ▼
                                        (returns to FSOC SIMULATION ENGINE)

              │
              ▼
 ┌─────────────────────────┐
 │  PERFORMANCE ANALYSIS   │
 │                         │
 │ Python + NumPy          │
 │ (metrics.py)            │
 │                         │
 │ • Tracking Error        │
 │ • Centroid Error        │
 │ • RMSE                  │
 │ • FPS                   │
 │ • Acquisition Time      │
 │ • Re-acquisition Time   │
 │ • Target Loss           │
 │ • Lock Retention        │
 │ • Confidence            │
 └────────────┬────────────┘
              │
              ▼
 ┌─────────────────────────┐
 │        OUTPUTS          │
 │                         │
 │ CSV • JSON • PDF Report │
 └─────────────────────────┘
```

┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              BUSINESS LOGIC LAYER (MODEL)                            │
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐ │
│  │                        SIMULATION ENGINE                                          │ │
│  │  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────────────────┐  │ │
│  │  │   Simulator     │───▶│  CameraController│───▶│  BeaconTracker (Kalman)    │  │ │
│  │  │  - Creates frames│   │  - Pan/Tilt     │   │  - State estimation         │  │ │
│  │  │  - Moving targets│   │  - Angle limits  │   │  - Velocity prediction      │  │ │
│  │  │  - Disturbances  │   │  - Slew rates    │   │  - Tracking state machine   │  │ │
│  │  │  - Noise models   │   │  - Gain control  │   │  - Confidence scoring       │  │ │
│  │  └─────────────────┘    └─────────────────┘    └─────────────────────────────┘  │ │
│  └─────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐ │
│  │                        DETECTION & TRACKING PIPELINE                              │ │
│  │                                                                                 │ │
│  │  Raw Frame ──▶ detect_beacon() ──▶ Detection Result ──▶ BeaconTracker.update()│ │
│  │                    │      │                                                      │ │
│  │                    │      ├──▶ (x, y) position                                 │ │
│  │                    │      ├──▶ confidence score                                │ │
│  │                    │      ├──▶ ai_score (AI verification)                     │ │
│  │                    │      ├──▶ signature_score                                 │ │
│  │                    │      ├──▶ candidates[] (all detections)                  │ │
│  │                    │      └──▶ bbox[] (bounding boxes)                        │ │
│  └─────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐ │
│  │                        BENCHMARK ENGINE                                          │ │
│  │  ┌─────────────────────────────────────────────────────────────────────────────┐│ │
│  │  │  run_video_benchmark()                                                       ││ │
│  │  │  - Input: MP4 video file                                                     ││ │
│  │  │  - Output: tracked MP4 + telemetry CSV + JSON summary + PDF report          ││ │
│  │  │  - Uses SAME detector/tracker pipeline as live simulation                    ││ │
│  │  │  - VideoBenchmarkWorker (QThread) for async execution                        ││ │
│  │  └─────────────────────────────────────────────────────────────────────────────┘│ │
│  └─────────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              DATA FLOW ARCHITECTURE                                  │
│                                                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  SIMULATOR   │───▶│   DETECTOR   │───▶│    TRACKER   │───▶│  CONTROLLER  │      │
│  │  (Frame Gen) │    │ (detect_     │    │ (Kalman      │    │ (Pan/Tilt    │      │
│  │              │    │  beacon)     │    │  Filter)     │    │  Update)     │      │
│  └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘      │
│         │                  │                  │                  │                   │
│         ▼                  ▼                  ▼                  ▼                   │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  Frame with  │    │ Position +   │    │ Tracked Pos  │    │  Updated     │      │
│  │  Disturbances│    │ Confidence   │    │ + Velocity   │    │  Pan/Tilt    │      │
│  └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘      │
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐ │
│  │                           TELEMETRY & REPORTING                                  │ │
│  │  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    ┌──────────────┐  │ │
│  │  │  LIVE TELEMETRY│  │ BENCHMARK    │    │  PERFORMANCE │    │  INSTANT     │  │ │
│  │  │  (Real-time   │  │ TELEMETRY    │    │  REPORT      │    │  REPORT      │  │ │
│  │  │   GUI update) │  │ (CSV export) │    │  (JSON/PDF)  │    │  (Snapshot)  │  │ │
│  │  └──────────────┘    └──────────────┘    └──────────────┘    └──────────────┘  │ │
│  └─────────────────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              COMPONENT DEPENDENCY GRAPH                               │
│                                                                                      │
│  gui.py (FSOCWindow)                                                                │
│  ├── benchmark_video.py (run_video_benchmark, BenchmarkScreen)                     │
│  │   └── benchmark.py (legacy run_video_benchmark)                                 │
│  ├── video_analysis.py (VideoAnalysisWindow)                                       │
│  ├── digital_twin.py (DigitalTwinWidget, DigitalTwinWindow, TwinPane)              │
│  ├── controller.py (CameraController)                                              │
│  ├── detector.py (detect_beacon)                                                   │
│  ├── tracker.py (BeaconTracker)                                                    │
│  ├── simulator.py (Simulator)                                                      │
│  ├── config.py (Constants: FRAME_WIDTH, FOV, UPDATE_RATE, etc.)                   │
│  ├── metrics.py (average, maximum, rmse, lock_retention, target_loss, etc.)       │
│  ├── performance_report.py (PerformanceSummary, save_performance_report)          │
│  ├── auth_config.py (APP_PASSWORD)                                                 │
│  ├── instant_report.py (parse_report_time, save_instant_report)                   │
│  └── main.py (Entry point: creates QApplication, shows FSOCWindow)                │
└─────────────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              CONFIGURATION VALUES                                    │
│                                                                                      │
│  ┌─────────────────┬──────────────────┬──────────────────────────────────────────┐ │
│  │  PARAMETER      │  DEFAULT VALUE   │  DESCRIPTION                            │ │
│  ├─────────────────┼──────────────────┼──────────────────────────────────────────┤ │
│  │  FRAME_WIDTH    │  640 px          │  Camera frame width                     │ │
│  │  FRAME_HEIGHT   │  480 px          │  Camera frame height                    │ │
│  │  FOV_HORIZONTAL │  4.0°           │  Camera horizontal field of view        │ │
│  │  FOV_VERTICAL   │  3.0°           │  Camera vertical field of view         │ │
│  │  UPDATE_RATE    │  30 Hz          │  Simulation update rate (timer = 33ms) │ │
│  │  MAX_PAN_SPEED  │  5.0°/s         │  Maximum camera pan speed               │ │
│  │  MAX_TILT_SPEED │  5.0°/s         │  Maximum camera tilt speed              │ │
│  │  BEACON_SIZE    │  10 px          │  Default beacon size in pixels          │ │
│  │  LINK_READY_AI_THRESHOLD │ 0.20     │  AI score threshold for link ready     │ │
│  │  LINK_READY_STREAK_FRAMES │ 10      │  Consecutive frames for link ready     │ │
│  │  LOCK_REQUIRED_FRAMES    │ 5       │  Frames needed for acquisition lock    │ │
│  │  LOCK_ERROR_THRESHOLD_PX │ 10.0 px  │  Max error for lock status             │ │
│  │  PS_ACQUISITION_MAX_S    │ 2.0 s   │  PS requirement: max acquisition time  │ │
│  │  PS_TRACKING_ERROR_MAX_PX│ 10.0 px │  PS requirement: max tracking error    │ │
│  │  PS_TARGET_LOSS_MAX_PERCENT│ 5.0%  │  PS requirement: max target loss      │ │
│  │  PS_REACQUISITION_MAX_S  │ 1.0 s   │  PS requirement: max re-acquisition   │ │
│  │  PS_PROCESSING_MIN_FPS   │ 20.0 FPS│  PS requirement: min processing speed │ │
│  └─────────────────┴──────────────────┴──────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              DISTURBANCE SYSTEM                                      │
│                                                                                      │
│  ┌─────────────────────────────────────────────────────────────────────────────────┐ │
│  │  Disturbance Presets (applied via Simulator.set_disturbance_preset())          │ │
│  │                                                                                 │ │
│  │  Clear:           noise=0, turbulence=0, jitter=0, platform=0, atmosphere=Clear │ │
│  │  Sensor Noise:    noise=Gaussian(20), turbulence=0, jitter=0, platform=0       │ │
│  │  Vibration:       turbulence=10px, jitter=15px, platform=15px, atmosphere=Clear │ │
│  │  Atmosphere:      turbulence=8px, atmosphere=Fog(70%), jitter=0, platform=0    │ │
│  │  Full Combined:   noise=Gaussian(8), jitter=8px, platform=8px, atmosphere=Haze(35%)│
│  │  Extreme PAT:     noise=Gaussian(20), turbulence=20px, jitter=20px,             │ │
│  │                    platform=20px, atmosphere=Fog(85%)                           │ │
│  └─────────────────────────────────────────────────────────────────────────────────┘ │
│                                                                                      │
│  Disturbance Types:                                                                 │
│  ├── Noise: None, Salt & Pepper, Gaussian, Poisson                                │
│  ├── Atmospheric Turbulence:Enabled/disabled + strength (0-20 px)                 │
│  ├── Camera Jitter:Enabled/disabled + strength (0-20 px)                         │
│  ├── Platform Motion:Enabled/disabled + strength (0-20 px) + pattern             │
│  │   Patterns: Linear, Circular, Random, Figure 8                                │
│  └── Atmosphere: Clear, Haze, Fog, Rain, Low Light + intensity (0-100%)          │
└─────────────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              VIDEO BENCHMARK WORKFLOW                                │
│                                                                                      │
│  User clicks BENCHMARK VIDEO                                                        │
│       │                                                                             │
│       ▼                                                                             │
│  QFileDialog: Select MP4 video                                                      │
│       │                                                                             │
│       ▼                                                                             │
│  QFileDialog: Select ground-truth CSV (optional, Cancel if none)                   │
│       │                                                                             │
│       ▼                                                                             │
│  Create BenchmarkScreen (QDialog)                                                   │
│       │                                                                             │
│       ▼                                                                             │
│  Set video preview (first frame of input)                                          │
│       │                                                                             │
│       ▼                                                                             │
│  Create VideoBenchmarkWorker (QThread)                                              │
│       │                                                                             │
│       ▼                                                                             │
│  Worker.run(): run_video_benchmark()                                                │
│       │                                                                             │
│       ├──▶ For each frame:                                                         │
│       │     ├──▶ detect_beacon(frame)                                              │
│       │     ├──▶ tracker.update(detection)                                        │
│       │     ├──▶ controller.update(prediction)                                    │
│       │     ├──▶ Render overlay on frame                                          │
│       │     ├──▶ Write frame to output video                                      │
│       │     └──▶ Write telemetry row to CSV                                       │
│       │                                                                             │
│       ▼                                                                             │
│  Worker emits: finished(summary, paths, video_path, status)                        │
│       │                                                                             │
│       ▼                                                                             │
│  FSOCWindow._on_benchmark_screen_finished():                                       │
│       ├──▶ Update progress bar to 100%                                             │
│       ├──▶ Display results in FINAL PERFORMANCE RESULTS panel                      │
│       ├──▶ Show tracked video in TRACKED OUTPUT panel                              │
│       ├──▶ Enable video playback controls                                         │
│       └──▶ Store report paths for later access                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              FILE OUTPUTS                                            │
│                                                                                      │
│  Benchmark Run Produces:                                                            │
│  ├── {stem}_benchmark_summary.json          → PerformanceSummary serialization      │
│  ├── {stem}_benchmark_summary.csv          → Tabular metrics summary               │
│  ├── {stem}_tracked_telemetry_{stamp}.csv  → Per-frame telemetry (24 fields)      │
│  ├── {stem}_tracked_{stamp}.mp4            → Tracked video with overlay           │
│  └── {stem}_benchmark_report.pdf           → PDF report (if generation succeeds)   │
│                                                                                      │
│  Telemetry CSV Fields (24 columns):                                                │
│  frame, timestamp_s, source_fps, detected, target_x, target_y,                    │
│  predicted_x, predicted_y, center_error_px, centroid_error_px,                    │
│  pan_deg, tilt_deg, fps_running, state, filter_confidence,                        │
│  detector_confidence, ai_confidence, signature_score, acquisition_s,              │
│  loss_event, reacquisition_event, reacquisition_time_s, link_ready                │
└─────────────────────────────────────────────────────────────────────────────────────┘


┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              KEY DESIGN DECISIONS                                    │
│                                                                                      │
│  1. DUAL OPERATING PATHS — NOT A SINGLE SEQUENTIAL LOOP               │
│     The architecture has two independent entry points sharing the      │
│     same detector/tracker code:                                        │
│     • Live FSOC Loop  → closed-loop PAT, disturbances, live GUI       │
│     • Benchmark Path  → async MP4 analysis, no simulator involvement  │
│                                                                        │
│  2. SINGLE SOURCE OF TRUTH: Simulator is authoritative mission state. │
│     All views (radar, camera, twin) render from the same state.       │
│                                                                        │
│  3. SAME PIPELINE FOR LIVE & BENCHMARK: benchmark uses identical      │
│     detector/tracker/controller code, ensuring results reflect        │
│     real system performance.                                           │
│                                                                        │
│  4. DISPLAY-ONLY ZOOM: Camera zoom only scales the displayed pixmap,  │
│     never the 640×480 tracking frame. Zero effect on accuracy.        │
│                                                                        │
│  5. HONEST METRICS: GT-dependent metrics shown as "—" when ground    │
│     truth is unavailable. Never fabricated as 0.0.                    │
│                                                                        │
│  6. THREADING MODEL: Benchmark runs on QThread worker; Qt signals     │
│     carry progress/completion/error back to the main thread.          │
│                                                                        │
│  7. AUTHENTICATION GATE: APP_PASSWORD must be satisfied before        │
│     simulation can start.                                              │
│                                                                        │
│  8. CLOSED-LOOP POINTING: Controller slews toward Kalman-predicted    │
│     position every frame. Dead zone (0.25 px), slew-rate limit,      │
│     and P-gain (default 2.0) prevent oscillation.                     │
│                                                                        │
│  9. AI VERIFIER — NO EXTERNAL ML DEPENDENCY: BeaconAIVerifier is     │
│     a 5-feature logistic model trained once at startup on 1800        │
│     synthetic samples. No PyTorch / TensorFlow required.              │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Running the Application

### Prerequisites
```
Python 3.13
pip install opencv-python-headless numpy PySide6
```

### Launch
```bash
cd "FSOC FOLDER ___/FSOC"
python main.py
```

### Authentication
- Password configured in `auth_config.py` → `APP_PASSWORD`
- Required before simulation starts

### Benchmark Video
1. Click **BENCHMARK VIDEO** in the main window or Report Center
2. Select an `.mp4` video file
3. Optionally select a ground-truth CSV (`frame,x,y` columns) — cancel for no GT
4. Benchmark runs asynchronously; progress shown in separate window
5. Results: tracked video · telemetry CSV · JSON summary · PDF report

### Disturbance Testing
- Select a preset from the **Disturbance Preset** dropdown
  *(Clear · Sensor Noise · Vibration · Atmosphere · Full Combined · Extreme PAT)*
- Or configure individual disturbances manually via sliders / checkboxes

---

## Performance Metrics (PS Requirements)

The application tracks and reports compliance with these Pointing System requirements:

| Metric | Requirement | Where Displayed |
|--------|-------------|-----------------|
| Tracking Error | ≤ 10 px | LIVE TELEMETRY, FINAL RESULTS |
| Target Loss | < 5% | LIVE TELEMETRY, FINAL RESULTS |
| Acquisition Time | ≤ 2 s | LIVE TELEMETRY, FINAL RESULTS |
| Re-acquisition Time | ≤ 1 s | LIVE TELEMETRY, FINAL RESULTS |
| Processing FPS | ≥ 20 FPS | LIVE TELEMETRY, FINAL RESULTS |
| Link Ready | AI score ≥ 0.20 for 10 consecutive frames | LINK READINESS panel |

---

## Documentation Files

| File | Purpose |
|------|---------|
| `ARCHITECTURE.md` | This document — full technical architecture |
| `ARCHITECTURE_FLOWCHART.png` | Visual dual-path flowchart (high resolution) |
| `TECHNICAL_SUMMARY.md` | Quick-reference card |
| `PS26169_COMPLIANCE_AUDIT.md` | PS 26169 requirement-by-requirement audit |
| `ASTRALINK_Mathematical_Formulas.docx` | Kalman / control law derivations |
