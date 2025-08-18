#!/usr/bin/env python3
import os
import sys
import argparse
import glob
import time
import math

import cv2
import numpy as np
from ultralytics import YOLO

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True,
                   help='Path to YOLO model file (e.g. runs/detect/train/weights/best.pt)')
    p.add_argument('--source', required=True,
                   help='Image, folder, video file, camera index (0) or "usb0", or "picamera"')
    p.add_argument('--thresh', type=float, default=0.5,
                   help='Display confidence threshold (default 0.5)')
    p.add_argument('--resolution', default=None,
                   help='Display/record resolution as WIDTHxHEIGHT (e.g. 640x480)')
    p.add_argument('--record', action='store_true',
                   help='Record output to demo1.avi (requires --resolution)')
    return p.parse_args()

def safe_int(v, default=0):
    try:
        return int(v)
    except:
        return default

def is_image_ext(ext):
    return ext.lower() in ('.jpg','.jpeg','.png','.bmp')

def is_video_ext(ext):
    return ext.lower() in ('.avi','.mov','.mp4','.mkv','.wmv')

def main():
    args = parse_args()
    model_path = args.model
    img_source = args.source
    min_thresh = args.thresh
    user_res = args.resolution
    record = args.record

    # model file exists?
    if not os.path.exists(model_path):
        print('ERROR: Model path not found:', model_path)
        sys.exit(1)

    # load model
    model = YOLO(model_path, task='detect')
    labels = model.names

    # decide source type
    source_type = None
    usb_idx = None
    picam_idx = None

    if os.path.isdir(img_source):
        source_type = 'folder'
    elif os.path.isfile(img_source):
        _, ext = os.path.splitext(img_source)
        if is_image_ext(ext):
            source_type = 'image'
        elif is_video_ext(ext):
            source_type = 'video'
        else:
            print(f'Unsupported file extension: {ext}')
            sys.exit(1)
    elif img_source.lower().startswith('usb'):
        source_type = 'usb'
        usb_idx = safe_int(img_source[3:], 0)
    elif img_source.isdigit():
        source_type = 'usb'
        usb_idx = int(img_source)
    elif 'picamera' in img_source.lower():
        source_type = 'picamera'
        # optional index after 'picamera'
        try:
            picam_idx = int(img_source.lower().replace('picamera','') or 0)
        except:
            picam_idx = 0
    else:
        print('Invalid --source value. Use image, folder, video file, camera index, usb0, or picamera.')
        sys.exit(1)

    # parse resolution
    resize = False
    resW = resH = None
    if user_res:
        try:
            parts = user_res.lower().split('x')
            if len(parts) != 2:
                raise ValueError
            resW, resH = int(parts[0]), int(parts[1])
            resize = True
        except Exception:
            print('Invalid --resolution. Use WIDTHxHEIGHT like 640x480.')
            sys.exit(1)

    # recording check
    recorder = None
    if record:
        if source_type not in ('video','usb'):
            print('Recording only valid for video or camera sources.')
            sys.exit(1)
        if not resize:
            print('Please specify --resolution to record.')
            sys.exit(1)
        record_name = 'demo1.avi'
        record_fps = 30
        # MJPG sometimes works most compatibly
        recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW, resH))

    # prepare source
    imgs_list = []
    cap = None
    if source_type == 'image':
        imgs_list = [img_source]
    elif source_type == 'folder':
        # deterministic order
        all_files = sorted(glob.glob(os.path.join(img_source, '*')))
        for f in all_files:
            _, ext = os.path.splitext(f)
            if is_image_ext(ext):
                imgs_list.append(f)
        if len(imgs_list) == 0:
            print('No images found in folder.')
            sys.exit(1)
    elif source_type in ('video','usb'):
        cap_arg = img_source if source_type == 'video' else (usb_idx if usb_idx is not None else 0)
        cap = cv2.VideoCapture(cap_arg)
        if not cap.isOpened():
            print('ERROR: Unable to open video/camera:', cap_arg)
            sys.exit(1)
        if resize:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, resW)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, resH)
    elif source_type == 'picamera':
        # Picamera2 usage can vary; only configure if resolution provided, else rely on defaults.
        try:
            from picamera2 import Picamera2
            cap = Picamera2()
            if resize:
                cap.configure(cap.create_video_configuration(main={"format": 'XRGB8888', "size": (resW, resH)}))
            else:
                # choose a default moderate size if none specified
                cap.configure(cap.create_video_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
            cap.start()
        except Exception as e:
            print('Picamera2 not available or failed to start:', e)
            sys.exit(1)

    # colors
    bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106),
                   (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

    fps_buffer = []
    fps_avg_len = 200
    img_count = 0

    # main loop
    while True:
        t_start = time.perf_counter()

        if source_type in ('image','folder'):
            if img_count >= len(imgs_list):
                print('All images processed. Exiting.')
                break
            frame = cv2.imread(imgs_list[img_count])
            if frame is None:
                print('Failed to read image:', imgs_list[img_count])
                img_count += 1
                continue
            img_count += 1

        elif source_type == 'video':
            ret, frame = cap.read()
            if not ret:
                print('End of video reached.')
                break

        elif source_type == 'usb':
            ret, frame = cap.read()
            if not ret or frame is None:
                print('Camera read failed; exiting.')
                break

        elif source_type == 'picamera':
            frame_bgra = cap.capture_array()
            frame = cv2.cvtColor(np.copy(frame_bgra), cv2.COLOR_BGRA2BGR)
            if frame is None:
                print('Picamera read failed; exiting.')
                break

        # resize if requested
        if resize and frame is not None:
            frame = cv2.resize(frame, (resW, resH))

        # model inference
        # NOTE: ultralytics returns a list of Results; use the first element
        results = model(frame, verbose=False)  # returns a sequence
        res = results[0]

        # vectorized extraction (faster than per-box .cpu() calls)
        boxes = res.boxes
        object_count = 0

        if boxes is not None and len(boxes) > 0:
            # xyxy -> (N,4), conf -> (N,), cls -> (N,)
            try:
                xyxy_all = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy().astype(int)
            except Exception:
                # fallback if attributes not present as tensors
                xyxy_all = np.array(boxes.xyxy)
                confs = np.array(boxes.conf)
                clss = np.array(boxes.cls).astype(int)

            img_h, img_w = frame.shape[:2]

            for (xyxy, conf, classidx) in zip(xyxy_all, confs, clss):
                if float(conf) <= min_thresh:
                    continue

                xmin, ymin, xmax, ymax = [int(round(x)) for x in xyxy]
                xmin, xmax = max(0, min(xmin, img_w-1)), max(0, min(xmax, img_w-1))
                ymin, ymax = max(0, min(ymin, img_h-1)), max(0, min(ymax, img_h-1))

                classname = labels.get(classidx, str(classidx)) if isinstance(labels, dict) else labels[classidx]
                color = bbox_colors[classidx % len(bbox_colors)]

                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

                # label
                label = f'{classname}: {int(conf*100)}%'
                labelSize, baseLine = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                label_w, label_h = labelSize
                label_x1 = xmin
                label_y1 = max(ymin, label_h + 10)
                cv2.rectangle(frame, (label_x1, label_y1 - label_h - 8), (label_x1 + label_w, label_y1 + baseLine - 8), color, cv2.FILLED)
                cv2.putText(frame, label, (label_x1, label_y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

                # diagonal in pixels
                box_w = float(xmax - xmin)
                box_h = float(ymax - ymin)
                diag_px = math.sqrt(box_w**2 + box_h**2)
                diag_label = f'{diag_px:.1f}px'
                diag_size, diag_base = cv2.getTextSize(diag_label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
                diag_w, diag_h = diag_size
                padding = 6

                tx1 = xmax - diag_w - padding
                ty1 = ymax - diag_h - padding
                tx2 = xmax
                ty2 = ymax

                # keep inside image
                if tx1 < 0:
                    tx1 = max(0, xmin)
                    tx2 = tx1 + diag_w + padding
                if ty1 < 0:
                    ty1 = max(0, ymin)
                    ty2 = ty1 + diag_h + padding
                if tx2 > img_w:
                    tx2 = img_w
                    tx1 = max(0, tx2 - diag_w - padding)
                if ty2 > img_h:
                    ty2 = img_h
                    ty1 = max(0, ty2 - diag_h - padding)

                # ensure ints
                tx1i, ty1i, tx2i, ty2i = map(int, (tx1, ty1, tx2, ty2))
                cv2.rectangle(frame, (tx1i, ty1i), (tx2i, ty2i), color, cv2.FILLED)
                text_x = tx1i + 2
                text_y = ty2i - 2
                cv2.putText(frame, diag_label, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,0,0), 1)

                object_count += 1

        # FPS calculation
        t_stop = time.perf_counter()
        frame_time = max(1e-6, t_stop - t_start)
        fps = 1.0 / frame_time
        if len(fps_buffer) >= fps_avg_len:
            fps_buffer.pop(0)
        fps_buffer.append(fps)
        avg_fps = float(np.mean(fps_buffer)) if fps_buffer else 0.0

        # overlays
        if source_type in ('video','usb','picamera'):
            cv2.putText(frame, f'FPS: {avg_fps:0.2f}', (10,20), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)
        cv2.putText(frame, f'Objects: {object_count}', (10,40), cv2.FONT_HERSHEY_SIMPLEX, .7, (0,255,255), 2)

        cv2.imshow('YOLO detection results', frame)
        if recorder is not None:
            recorder.write(frame)

        # key handling
        if source_type in ('image','folder'):
            key = cv2.waitKey(0) & 0xFF
        else:
            key = cv2.waitKey(5) & 0xFF

        if key == ord('q'):
            break
        elif key == ord('s'):
            cv2.waitKey(0)  # pause
        elif key == ord('p'):
            cv2.imwrite('capture.png', frame)
            print('Saved capture.png')

    # cleanup
    print(f'Average pipeline FPS: {avg_fps:.2f}')
    if cap is not None:
        if source_type == 'picamera':
            try:
                cap.stop()
            except:
                pass
        else:
            cap.release()
    if recorder is not None:
        recorder.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
