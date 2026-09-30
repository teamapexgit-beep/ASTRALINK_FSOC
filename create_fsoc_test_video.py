import cv2
import csv
import math
import numpy as np

W, H = 640, 480
FPS = 30
SECONDS = 15
N = FPS * SECONDS

VIDEO_OUT = "synthetic_fsoc_beacon.mp4"
GT_OUT = "synthetic_fsoc_beacon_groundtruth.csv"

fourcc = cv2.VideoWriter_fourcc(*"mp4v")
writer = cv2.VideoWriter(VIDEO_OUT, fourcc, FPS, (W, H))

if not writer.isOpened():
    raise RuntimeError("Could not create MP4 video.")

with open(GT_OUT, "w", newline="") as f:
    csv_writer = csv.writer(f)
    csv_writer.writerow(["frame", "x", "y"])

    for frame in range(N):
        t = frame / FPS

        if t < 5:
            u = t / 5.0
            x = 90 + 460 * u
            y = 240 + 45 * math.sin(u * math.pi)

        elif t < 10:
            u = (t - 5) / 5.0
            x = 550 - 250 * u
            y = 240 - 140 * math.sin(u * math.pi / 2)

        else:
            u = (t - 10) / 5.0
            x = 300 + 70 * math.sin(u * math.pi)
            y = 100 + 140 * u

        x = int(round(x))
        y = int(round(y))

        img = np.full((H, W, 3), 25, dtype=np.uint8)

        cv2.rectangle(
            img,
            (8, 8),
            (W - 9, H - 9),
            (45, 45, 45),
            1
        )

        for radius, intensity in [
            (18, 18),
            (12, 30),
            (8, 55)
        ]:
            cv2.circle(
                img,
                (x, y),
                radius,
                (intensity, intensity, intensity),
                -1
            )

        cv2.rectangle(
            img,
            (x - 5, y - 5),
            (x + 5, y + 5),
            (255, 255, 255),
            -1
        )

        writer.write(img)
        csv_writer.writerow([frame, x, y])

writer.release()

print("DONE!")
print("Created:", VIDEO_OUT)
print("Created:", GT_OUT)
print("Frames:", N)
print("Resolution:", W, "x", H)
print("FPS:", FPS)
