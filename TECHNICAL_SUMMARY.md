# ASTRALINK FSOC - Technical Summary

## Technology Stack

| Layer | Technology | Purpose |
|-------|------------|---------|
| **Language** | Python 3.13 | Core implementation |
| **GUI Framework** | PySide6 (Qt6 for Python) | Desktop application UI |
| **Computer Vision** | OpenCV (cv2) | Video I/O, image processing, frame analysis |
| **Numerical Computing** | NumPy | Array operations, random number generation |
| **Data Serialization** | CSV, JSON | Telemetry logging, performance reports |
| **Diagram Generation** | Matplotlib | Architecture diagrams |

## Architecture Pattern

**MVC-inspired (Model-View-Controller):**

- **View (Presentation Layer):** PySide6 widgets - QMainWindow, custom widgets (RadarWidget, TrendGraph, CameraFeedWidget, BenchmarkScreen)
- **Model (Business Logic Layer):** Simulator, BeaconTracker (Kalman filter), CameraController, detect_beacon()
- **Controller:** FSOCWindow - handles user input, coordinates model and view, manages timer-driven update loop

## Core Components

### 1. Simulator (simulator.py)
- Generates 640x480 frames with starfield background
- Moves multiple optical targets (1-5) with configurable motion patterns
- Applies disturbances: noise, turbulence, jitter, platform motion, atmospheric effects
- Provides ground truth positions for metrics calculation

### 2. Detection Pipeline (detector.py)
- `detect_beacon(frame, expected_position, preferred_signature)` - main entry point
- Returns: position (x,y), confidence, ai_score, signature_score, candidates[]
- Uses signature-based detection for designated target selection

### 3. Kalman Tracker (tracker.py)
- `BeaconTracker` - state estimation filter
- Tracks: x, y, velocity_x, velocity_y, covariance matrix
- States: SEARCHING, ACQUIRING, TRACKING, PREDICTING, RE-ACQUIRING, LOST
- Provides confidence scoring and velocity prediction

### 4. Camera Controller (controller.py)
- `CameraController` - pan/tilt control
- Configurable angle limits, slew rates, gain
- Closed-loop pointing toward predicted beacon position

### 5. GUI (gui.py)
- `FSOCWindow` - main application window (QMainWindow)
- Real-time update loop via QTimer at 33ms (~30 Hz)
- Custom widgets for radar, graphs, camera feed, telemetry
- Benchmark video support with async QThread worker
- Authentication gate, instant reporting, 3D digital twin

### 6. Benchmark Engine (benchmark_video.py)
- `run_video_benchmark()` - processes video files
- Uses SAME detector/tracker/controller pipeline as live simulation
- Outputs: tracked MP4 + telemetry CSV + JSON summary + PDF report
- Async execution via VideoBenchmarkWorker (QThread)

## Configuration (config.py)

| Parameter | Default | Description |
|-----------|---------|-------------|
| FRAME_WIDTH | 640 | Camera frame width (pixels) |
| FRAME_HEIGHT | 480 | Camera frame height (pixels) |
| FOV_HORIZONTAL | 4.0° | Horizontal field of view |
| FOV_VERTICAL | 3.0° | Vertical field of view |
| UPDATE_RATE | 30 Hz | Simulation/timer update rate |
| MAX_PAN_SPEED | 5.0°/s | Maximum pan speed |
| MAX_TILT_SPEED | 5.0°/s | Maximum tilt speed |
| BEACON_SIZE | 10 px | Default beacon size |
| LOCK_ERROR_THRESHOLD_PX | 10.0 | Max error for lock status |
| LINK_READY_AI_THRESHOLD | 0.20 | AI score threshold for link ready |
| PS_* | Various | Pointing System requirements |

## PS Requirements (Pointing System)

- **Tracking Error:** ≤10 px
- **Target Loss:** <5%
- **Acquisition Time:** ≤2 s
- **Re-acquisition Time:** ≤1 s
- **Processing FPS:** ≥20 FPS

## File Outputs

### Benchmark Run Produces:
1. `{stem}_benchmark_summary.json` - PerformanceSummary serialization
2. `{stem}_benchmark_summary.csv` - Tabular metrics
3. `{stem}_tracked_telemetry_{stamp}.csv` - Per-frame telemetry (24 fields)
4. `{stem}_tracked_{stamp}.mp4` - Tracked video with overlay
5. `{stem}_benchmark_report.pdf` - PDF report (if generation succeeds)

## Architecture Flow

```
User Input → GUI (View) → Business Logic (Model) → Data Flow → Outputs
     │                                                    │
     └── Authentication ───────────────────────────────────┘
```

1. User interacts with GUI (controls, buttons, authentication)
2. GUI displays real-time visualization (radar, camera, telemetry, graphs)
3. Business logic runs simulation: Simulator → Detector → Tracker → Controller
4. Data flows between components every 30ms (timer-driven)
5. Outputs: live telemetry (GUI), benchmark results (files), reports (JSON/PDF)

## Key Design Decisions

1. **Single Source of Truth:** Simulator is authoritative mission state
2. **Same Pipeline:** Benchmark uses identical code as live simulation
3. **Display-Only Zoom:** Camera zoom doesn't affect tracking
4. **Honest Metrics:** Missing data shown as "—", never fabricated
5. **Async Benchmark:** QThread worker keeps GUI responsive
6. **Authentication Gate:** Password required before simulation
7. **Closed-Loop Pointing:** Controller slews toward predicted position

## Documentation Files

- `ARCHITECTURE.md` - Full technical architecture document
- `ARCHITECTURE_FLOWCHART.png` - Visual flowchart (high resolution)
- `TECHNICAL_SUMMARY.md` - This quick reference
