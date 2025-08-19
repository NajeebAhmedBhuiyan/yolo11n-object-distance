#!/usr/bin/env python3
"""
follow_person_usb.py

Detect a person with ultralytics YOLO on RPi5 camera and send single-byte
commands to an Arduino Nano over USB serial to keep the person centered & at a
target distance.

Commands:
  'L' - turn left
  'R' - turn right
  'F' - step forward
  'B' - step back
  'X' - stop / stand

Usage:
  python3 follow_person_usb.py --model path/to/best.pt --source 0 --resolution 640x480
  Optional: --serial /dev/ttyACM0 to override auto-detection
"""

import argparse
import math
import time
import collections
import os
import sys
import glob

import cv2
import numpy as np
from ultralytics import YOLO

# Serial (pyserial)
try:
    import serial
except Exception:
    print("pyserial missing. Install: pip3 install pyserial")
    serial = None

# ---------- Defaults you can tune ----------
SERIAL_BAUD = 9600
FRAME_BUFFER = 5
CONFIRM_FRAMES = 3
CMD_INTERVAL = 0.18
H_CENTER_DEADZONE = 0.08
HYSTERESIS = 0.02
TARGET_DIAG = 220.0
DIAG_TOLERANCE = 0.12
VERBOSE = True

# ---------- helpers ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='Path to YOLO model file (e.g. runs/detect/train/weights/best.pt)')
    p.add_argument('--source', required=True, help='camera index (0) or video file')
    p.add_argument('--resolution', default='640x480')
    p.add_argument('--thresh', type=float, default=0.5)
    p.add_argument('--serial', default=None, help='Serial port (e.g. /dev/ttyACM0). If omitted script tries to auto-find.')
    return p.parse_args()

def find_person_class_indices(labels):
    person_idxs = set()
    if isinstance(labels, dict):
        for k, v in labels.items():
            if isinstance(v, str) and v.lower() == 'person':
                try:
                    person_idxs.add(int(k))
                except:
                    pass
    else:
        for i, name in enumerate(labels):
            if isinstance(name, str) and name.lower() == 'person':
                person_idxs.add(i)
    return person_idxs

def auto_find_serial():
    # try common serial device names
    candidates = glob.glob('/dev/ttyACM*') + glob.glob('/dev/ttyUSB*') + glob.glob('/dev/serial/by-id/*')
    if not candidates:
        return None
    # prefer /dev/ttyACM0 if present
    for c in candidates:
        if os.path.basename(c).startswith('ttyACM'):
            return c
    return candidates[0]

def open_serial(port, baud):
    if serial is None:
        print("pyserial not installed.")
        return None
    try:
        s = serial.Serial(port, baud, timeout=0.1)
        time.sleep(0.2)
        if VERBOSE:
            print("[SERIAL] Opened", port)
        # flush any startup bytes
        s.reset_input_buffer()
        s.reset_output_buffer()
        return s
    except Exception as e:
        print("[SERIAL] Failed to open", port, e)
        return None

def send_cmd(ser, ch):
    if ser is None:
        return False
    try:
        ser.write(ch.encode('ascii'))
        return True
    except Exception as e:
        print("Serial write failed:", e)
        return False

# ---------- main ----------
def main():
    args = parse_args()

    # resolution
    try:
        w,h = args.resolution.lower().split('x')
        resW, resH = int(w), int(h)
    except:
        print("Invalid resolution format (use WIDTHxHEIGHT).")
        return

    # model exists?
    if not os.path.exists(args.model):
        print("Model file not found:", args.model)
        return

    model = YOLO(args.model, task='detect')
    labels = model.names
    person_class_idxs = find_person_class_indices(labels)
    if not person_class_idxs:
        person_class_idxs.add(0)
        print("Warning: person class not found in model names; assuming index 0")

    # open serial port
    serial_port = args.serial
    if serial_port is None:
        serial_port = auto_find_serial()
        if serial_port:
            print("Auto-detected serial device:", serial_port)
        else:
            print("No serial device auto-detected. Use --serial to set path (e.g. /dev/ttyACM0).")
    ser = None
    if serial_port:
        ser = open_serial(serial_port, SERIAL_BAUD)

    # open camera
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Failed to open camera/source:", source)
        if ser: ser.close()
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)

    # buffers & state
    errx_buf = collections.deque(maxlen=FRAME_BUFFER)
    diag_buf = collections.deque(maxlen=FRAME_BUFFER)
    stable_h_count = 0
    last_cmd_time = 0
    last_sent_cmd = None
    no_person_frames = 0
    MAX_NO_PERSON_BEFORE_STOP = 10

    print("Starting loop. Press 'q' in the window to stop.")

    while True:
        start = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Camera read failed.")
            break
        img_h, img_w = frame.shape[:2]

        results = model(frame, verbose=False)
        res = results[0]
        boxes = res.boxes
        chosen = None

        if boxes is not None and len(boxes) > 0:
            try:
                xyxy_all = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy().astype(int)
            except Exception:
                xyxy_all = np.array(boxes.xyxy)
                confs = np.array(boxes.conf)
                clss = np.array(boxes.cls).astype(int)

            candidates = []
            for (xyxy, conf, classidx) in zip(xyxy_all, confs, clss):
                if int(classidx) not in person_class_idxs:
                    continue
                if float(conf) <= args.thresh:
                    continue
                xmin, ymin, xmax, ymax = [int(round(x)) for x in xyxy]
                xmin = max(0, min(xmin, img_w - 1))
                xmax = max(0, min(xmax, img_w - 1))
                ymin = max(0, min(ymin, img_h - 1))
                ymax = max(0, min(ymax, img_h - 1))
                cx = int((xmin + xmax) / 2)
                cy = int((ymin + ymax) / 2)
                box_w = float(xmax - xmin)
                box_h = float(ymax - ymin)
                diag_px = math.sqrt(box_w**2 + box_h**2)
                candidates.append((cx, cy, diag_px, float(conf), xmin, ymin, xmax, ymax))
            if candidates:
                chosen = max(candidates, key=lambda x: x[2])

        if chosen is None:
            no_person_frames += 1
            if no_person_frames >= MAX_NO_PERSON_BEFORE_STOP:
                # send stop (X)
                now = time.time()
                if ser and (now - last_cmd_time) >= CMD_INTERVAL:
                    if last_sent_cmd != 'X':
                        if send_cmd(ser, 'X'):
                            last_sent_cmd = 'X'
                            last_cmd_time = now
                            if VERBOSE:
                                print("[CMD] No person - Sent X")
            cv2.putText(frame, "No person", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.imshow('Follow', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            continue
        else:
            no_person_frames = 0

        cx, cy, diag_px, conf, xmin, ymin, xmax, ymax = chosen
        color = (0,200,0)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(frame, f'person {int(conf*100)}% diag:{diag_px:.1f}', (xmin, max(0, ymin-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        half_w = img_w / 2.0
        errx = (cx - half_w) / half_w   # -1..1

        # smoothing
        errx_buf.append(errx)
        diag_buf.append(diag_px)
        errx_med = float(np.median(list(errx_buf)))
        diag_med = float(np.median(list(diag_buf)))

        # thresholds
        dead = H_CENTER_DEADZONE
        enter_left = -dead - HYSTERESIS
        enter_right = dead + HYSTERESIS

        diag_low = TARGET_DIAG * (1.0 - DIAG_TOLERANCE)
        diag_high = TARGET_DIAG * (1.0 + DIAG_TOLERANCE)

        # horizontal decision
        horizontal_cmd = None
        if errx_med <= enter_left:
            horizontal_cmd = 'L'
        elif errx_med >= enter_right:
            horizontal_cmd = 'R'
        else:
            horizontal_cmd = None

        if horizontal_cmd is None:
            stable_h_count = 0
        else:
            stable_h_count += 1

        # distance decision
        distance_cmd = None
        if diag_med < diag_low:
            distance_cmd = 'F'
        elif diag_med > diag_high:
            distance_cmd = 'B'
        else:
            distance_cmd = None

        # choose final command
        chosen_cmd = None
        if horizontal_cmd is not None and stable_h_count >= CONFIRM_FRAMES:
            chosen_cmd = horizontal_cmd
        elif horizontal_cmd is None:
            chosen_cmd = distance_cmd if distance_cmd is not None else 'X'
        else:
            chosen_cmd = None  # waiting for confirm frames

        now = time.time()
        if chosen_cmd is not None and ser is not None:
            if (now - last_cmd_time) >= CMD_INTERVAL:
                if chosen_cmd != last_sent_cmd:
                    ok = send_cmd(ser, chosen_cmd)
                    if ok:
                        last_sent_cmd = chosen_cmd
                        last_cmd_time = now
                        if VERBOSE:
                            print(f"[CMD] Sent {chosen_cmd} errx_med={errx_med:.3f} diag_med={diag_med:.1f}")

        status = f'err_med={errx_med:+.3f} diag_med={diag_med:.1f} cmd={chosen_cmd}'
        cv2.putText(frame, status, (10, img_h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.imshow('Follow', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            cv2.imwrite('capture_follow.png', frame)

    cap.release()
    cv2.destroyAllWindows()
    if ser: ser.close()
    print("Exiting.")

if __name__ == '__main__':
    main()
