"""Default configuration for the FSOC virtual-camera testbed."""

FRAME_WIDTH = 640
FRAME_HEIGHT = 480

# Virtual simulation / environment screen (the rendered world surface the
# virtual camera samples). This is NOT the browser/application viewport: the
# GUI stays fully responsive and never forces a 2000x2000 window.
SCREEN_WIDTH = 2000
SCREEN_HEIGHT = 2000

# Virtual camera type presented by the simulation/testbed.
CAMERA_TYPE = "Monochrome — Focal Plane Array"

# Problem-statement defaults
FOV_HORIZONTAL = 4.0
FOV_VERTICAL = 3.0
UPDATE_RATE = 30
MAX_PAN_SPEED = 5.0
MAX_TILT_SPEED = 5.0
BEACON_SIZE = 10

# Shared internal thresholds
LINK_READY_AI_THRESHOLD = 0.20
LINK_READY_STREAK_FRAMES = 10
LOCK_REQUIRED_FRAMES = 5
LOCK_ERROR_THRESHOLD_PX = 10.0

# PS performance limits (used for automatic report checks)
PS_ACQUISITION_MAX_S = 2.0
PS_TRACKING_ERROR_MAX_PX = 10.0
PS_TARGET_LOSS_MAX_PERCENT = 5.0  # strict: loss must be < 5%
PS_REACQUISITION_MAX_S = 1.0
PS_PROCESSING_MIN_FPS = 20.0
