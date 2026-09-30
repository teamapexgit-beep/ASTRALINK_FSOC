import cv2
import time

from simulator import Simulator
from detector import detect_beacon
from controller import CameraController
import config


simulator = Simulator()
controller = CameraController()

# FPS variables
previous_time = time.perf_counter()
fps = 0.0


while True:

    # Get camera frame
    frame = simulator.get_frame()

    # Detect beacon
    beacon_position = detect_beacon(frame)

    # Camera center
    center_x = config.FRAME_WIDTH // 2
    center_y = config.FRAME_HEIGHT // 2

    # Default status
    status = "TARGET: LOST"

    if beacon_position is not None:

        # Beacon detected
        # A single measurement is a detection, not a validated alignment lock.
        status = "TARGET: DETECTED"

        x, y = beacon_position

        # Calculate pixel error
        error_x = x - center_x
        error_y = y - center_y

        # Calculate camera correction
        pan, tilt = controller.update(
            error_x,
            error_y
        )

        # Apply correction to virtual camera
        simulator.update_camera(
            pan,
            tilt
        )

        # Draw detected beacon center
        cv2.circle(
            frame,
            (x, y),
            5,
            (0, 0, 255),
            -1
        )

        # Display beacon coordinates
        cv2.putText(
            frame,
            f"Beacon: ({x}, {y})",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # Display tracking error
        cv2.putText(
            frame,
            f"Error: ({error_x}, {error_y})",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # Display camera orientation
        cv2.putText(
            frame,
            f"Pan: {pan:.2f} deg  Tilt: {tilt:.2f} deg",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

    # Calculate FPS
    current_time = time.perf_counter()
    elapsed_time = current_time - previous_time

    if elapsed_time > 0:
        fps = 1 / elapsed_time

    previous_time = current_time

    # Display FPS
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 120),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    # Display tracking status
    cv2.putText(
        frame,
        status,
        (10, 155),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2
    )

    # Draw camera center
    cv2.circle(
        frame,
        (center_x, center_y),
        5,
        (255, 0, 0),
        -1
    )

    # Show frame
    cv2.imshow(
        "FSOC Virtual Camera",
        frame
    )

    # Press Q to quit
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break


cv2.destroyAllWindows()
