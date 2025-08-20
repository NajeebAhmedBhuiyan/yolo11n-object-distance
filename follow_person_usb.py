#!/usr/bin/env python3
"""
follow_person_usb_handshake.py

Use this on the RPi5 with a webcam. Detect person (YOLO) and send single-byte motion
commands to an Arduino over USB, using a handshake: Pi sends one command, Arduino
executes and then replies "DONE". Pi waits for DONE before sending the next command.

Usage:
  I'll give it later
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

# -------- CONFIG (tune these) --------
SERIAL_BAUD = 9600
FRAME_BUFFER = 5
CONFIRM_FRAMES = 3
ACK_TIMEOUT = 3.0           # seconds to wait for Arduino "DONE"
H_CENTER_DEADZONE = 0.08
HYSTERESIS = 0.02
TARGET_DIAG = 220.0
DIAG_TOLERANCE = 0.12
VERBOSE = True

# -------- helpers ----------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--source', required=True)
    p.add_argument('--resolution', default='640x480')
    p.add_argument('--thresh', type=float, default=0.5)
    p.add_argument('--serial', default=None, help='Serial path (e.g. /dev/ttyACM0). If omitted auto-detect.')
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
    candidates = glob.glob('/dev/ttyACM*') + glob.glob('/dev/ttyUSB*') + glob.glob('/dev/serial/by-id/*')
    if not candidates:
        return None
    # prefer ACM
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
            print(f"[SERIAL] Opened {port} @ {baud}")
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

# -------- main ----------
def main():
    args = parse_args()

    # resolution parsing
    try:
        w,h = args.resolution.lower().split('x')
        resW, resH = int(w), int(h)
    except:
        print("Invalid resolution. Use WIDTHxHEIGHT")
        return

    # model
    if not os.path.exists(args.model):
        print("Model file not found:", args.model)
        return
    model = YOLO(args.model, task='detect')
    labels = model.names
    person_class_idxs = find_person_class_indices(labels)
    if not person_class_idxs:
        print("Warning: person class not found; assuming 0")
        person_class_idxs.add(0)

    # serial
    serial_port = args.serial or auto_find_serial()
    if serial_port:
        print("Using serial port:", serial_port)
    else:
        print("No serial port auto-detected. Use --serial to set device.")
    ser = open_serial(serial_port, SERIAL_BAUD) if serial_port else None

    # camera
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("Failed to open camera/source", source)
        if ser: ser.close()
        return
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)

    # buffers and state
    errx_buf = collections.deque(maxlen=FRAME_BUFFER)
    diag_buf = collections.deque(maxlen=FRAME_BUFFER)
    stable_h_count = 0
    no_person_frames = 0
    MAX_NO_PERSON_BEFORE_STOP = 10

    awaiting_ack = False
    ack_deadline = 0.0
    last_cmd_time = 0.0

    print("Starting loop. Press 'q' to quit window. Waiting for target...")

    while True:
        t0 = time.time()
        ret, frame = cap.read()
        if not ret:
            print("Camera read error.")
            break
        img_h, img_w = frame.shape[:2]

        # YOLO inference
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
                if int(classidx) not in person_class_idxs: continue
                if float(conf) <= args.thresh: continue
                xmin, ymin, xmax, ymax = [int(round(x)) for x in xyxy]
                xmin = max(0, min(xmin, img_w-1))
                xmax = max(0, min(xmax, img_w-1))
                ymin = max(0, min(ymin, img_h-1))
                ymax = max(0, min(ymax, img_h-1))
                cx = int((xmin + xmax)/2)
                cy = int((ymin + ymax)/2)
                box_w = float(xmax - xmin)
                box_h = float(ymax - ymin)
                diag_px = math.sqrt(box_w**2 + box_h**2)
                candidates.append((cx, cy, diag_px, float(conf), xmin, ymin, xmax, ymax))

            if candidates:
                chosen = max(candidates, key=lambda x: x[2])  # closest (largest diag)

        if chosen is None:
            no_person_frames += 1
            if no_person_frames >= MAX_NO_PERSON_BEFORE_STOP:
                # try to stop robot if not already doing so
                if ser and not awaiting_ack:
                    send_cmd(ser, 'X')
                    awaiting_ack = True
                    ack_deadline = time.time() + ACK_TIMEOUT
                    if VERBOSE: print("[CMD] No person -> Sent X, awaiting DONE")
            cv2.putText(frame, "No person", (10,40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,255), 2)
            cv2.imshow('Follow', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            # poll serial for DONE while no person
            if ser and awaiting_ack:
                try:
                    line = ser.readline().decode(errors='ignore').strip()
                    if line:
                        if VERBOSE: print("[SER RX]", repr(line))
                        if line == 'DONE':
                            awaiting_ack = False
                except Exception:
                    pass
                if awaiting_ack and time.time() > ack_deadline:
                    print("[SERIAL] Ack timeout while no-person; clearing awaiting_ack")
                    awaiting_ack = False
            continue
        else:
            no_person_frames = 0

        cx, cy, diag_px, conf, xmin, ymin, xmax, ymax = chosen

        # overlays
        color = (0,200,0)
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
        cv2.putText(frame, f'person {int(conf*100)}% diag:{diag_px:.1f}', (xmin, max(0,ymin-10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # error metrics
        half_w = img_w / 2.0
        errx = (cx - half_w) / half_w    # normalized -1..1
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

        # final choice: prioritize centering
        chosen_cmd = None
        if horizontal_cmd is not None and stable_h_count >= CONFIRM_FRAMES:
            chosen_cmd = horizontal_cmd
        elif horizontal_cmd is None:
            chosen_cmd = distance_cmd if distance_cmd is not None else 'X'
        else:
            chosen_cmd = None  # waiting for confirm frames

        # --- HANDSHAKE logic: send one command at a time and wait for DONE ---
        now = time.time()

        # If not awaiting ack and we have a command to send, send it
        if not awaiting_ack and chosen_cmd is not None and ser is not None:
            # send command
            try:
                # clear old input to reduce stale lines
                ser.reset_input_buffer()
                ser.write(chosen_cmd.encode('ascii'))
                awaiting_ack = True
                ack_deadline = now + ACK_TIMEOUT
                if VERBOSE:
                    print(f"[SERIAL TX] Sent '{chosen_cmd}', awaiting DONE")
            except Exception as e:
                print("Serial write error:", e)
                awaiting_ack = False

        # If awaiting ack, poll serial for responses (non-blocking)
        if ser is not None and awaiting_ack:
            try:
                line = ser.readline().decode(errors='ignore').strip()
                if line:
                    if VERBOSE:
                        print("[SERIAL RX]", repr(line))
                    # Arduino sends "DONE" after finishing motion
                    if line == 'DONE':
                        awaiting_ack = False
                        # allow immediate next send on next loop iteration
            except Exception:
                pass

            # ack timeout guard
            if awaiting_ack and time.time() > ack_deadline:
                print("[SERIAL] ACK timeout, clearing awaiting_ack")
                awaiting_ack = False

        # overlays and debug info
        status = f'err_med={errx_med:+.3f} diag_med={diag_med:.1f} cmd={chosen_cmd} awaiting_ack={awaiting_ack}'
        cv2.putText(frame, status, (10, img_h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
        cv2.imshow('Follow', frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('p'):
            cv2.imwrite('capture_follow.png', frame)

    # cleanup
    cap.release()
    cv2.destroyAllWindows()
    if ser:
        ser.close()
    print("Exiting.")

if __name__ == '__main__':
    main()
