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

# -----------------------
# arg parsing & helpers
# -----------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True, help='Path to YOLO model file (e.g. runs/detect/train/weights/best.pt)')
    p.add_argument('--source', required=True, help='Image, folder, video file, camera index (0) or "usb0", or "picamera"')
    p.add_argument('--thresh', type=float, default=0.5, help='Display confidence threshold (default 0.5)')
    p.add_argument('--resolution', default=None, help='Display/record resolution as WIDTHxHEIGHT (e.g. 640x480)')
    p.add_argument('--record', action='store_true', help='Record output to demo1.avi (requires --resolution)')
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

def find_person_class_indices(labels):
    """Return a set of class indices corresponding to 'person' label (case-insensitive)."""
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

# -----------------------
# main
# -----------------------
def main():
    args = parse_args()
    model_path = args.model
    img_source = args.source
    min_thresh = args.thresh
    user_res = args.resolution
    record = args.record

    if not os.path.exists(model_path):
        print('ERROR: Model path not found:', model_path)
        sys.exit(1)

    model = YOLO(model_path, task='detect')
    labels = model.names
    person_class_idxs = find_person_class_indices(labels)
    if len(person_class_idxs) == 0:
        print('Warning: "person" label not found in model names. Falling back to class index 0 (COCO typical).')
        person_class_idxs.add(0)

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
        recorder = cv2.VideoWriter(record_name, cv2.VideoWriter_fourcc(*'MJPG'), record_fps, (resW, resH))

    # prepare source
    imgs_list = []
    cap = None
    if source_type == 'image':
        imgs_list = [img_source]
    elif source_type == 'folder':
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
        try:
            from picamera2 import Picamera2
            cap = Picamera2()
            if resize:
                cap.configure(cap.create_video_configuration(main={"format": 'XRGB8888', "size": (resW, resH)}))
            else:
                cap.configure(cap.create_video_configuration(main={"format": 'XRGB8888', "size": (640, 480)}))
            cap.start()
        except Exception as e:
            print('Picamera2 not available or failed to start:', e)
            sys.exit(1)

    bbox_colors = [(164,120,87), (68,148,228), (93,97,209), (178,182,133), (88,159,106),
                   (96,202,231), (159,124,168), (169,162,241), (98,118,150), (172,176,184)]

    fps_buffer = []
    fps_avg_len = 200
    img_count = 0

    # fonts & scales
    PERSON_LABEL_FONT_SCALE = 0.5
    POS_LABEL_FONT_SCALE = 0.45
    DIAG_LABEL_FONT_SCALE = 0.5
    FONT = cv2.FONT_HERSHEY_SIMPLEX

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

        if resize and frame is not None:
            frame = cv2.resize(frame, (resW, resH))

        # inference
        results = model(frame, verbose=False)
        res = results[0]
        boxes = res.boxes
        object_count = 0
        img_h, img_w = frame.shape[:2]

        if boxes is not None and len(boxes) > 0:
            try:
                xyxy_all = boxes.xyxy.cpu().numpy()
                confs = boxes.conf.cpu().numpy()
                clss = boxes.cls.cpu().numpy().astype(int)
            except Exception:
                xyxy_all = np.array(boxes.xyxy)
                confs = np.array(boxes.conf)
                clss = np.array(boxes.cls).astype(int)

            for (xyxy, conf, classidx) in zip(xyxy_all, confs, clss):
                if int(classidx) not in person_class_idxs:
                    continue
                if float(conf) <= min_thresh:
                    continue

                xmin, ymin, xmax, ymax = [int(round(x)) for x in xyxy]
                xmin, xmax = max(0, min(xmin, img_w-1)), max(0, min(xmax, img_w-1))
                ymin, ymax = max(0, min(ymin, img_h-1)), max(0, min(ymax, img_h-1))

                color = bbox_colors[classidx % len(bbox_colors)]

                # draw bbox
                cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

                # 1) top-left label (class + conf)
                person_label = f'person: {int(conf*100)}%'
                pl_size, pl_base = cv2.getTextSize(person_label, FONT, PERSON_LABEL_FONT_SCALE, 1)
                pl_w, pl_h = pl_size
                pl_x1 = xmin
                pl_y1 = max(ymin, pl_h + 8)
                cv2.rectangle(frame, (pl_x1, pl_y1 - pl_h - 6), (pl_x1 + pl_w, pl_y1 + pl_base - 6), color, cv2.FILLED)
                cv2.putText(frame, person_label, (pl_x1, pl_y1 - 2), FONT, PERSON_LABEL_FONT_SCALE, (0,0,0), 1)

                # compute center & normalized pos
                cx = int((xmin + xmax) / 2)
                cy = int((ymin + ymax) / 2)
                nx = cx / float(img_w)
                ny = cy / float(img_h)
                pos_label = f'({cx},{cy}) {nx:.0%},{ny:.0%}'

                # pos label (top-right of bbox)
                pos_size, pos_base = cv2.getTextSize(pos_label, FONT, POS_LABEL_FONT_SCALE, 1)
                pos_w, pos_h = pos_size
                padding = 6

                pos_px2 = xmax  # right edge align
                pos_py1 = ymin  # top align
                pos_px1 = pos_px2 - pos_w - padding
                pos_py2 = pos_py1 + pos_h + padding

                # clamp inside image
                if pos_px1 < 0:
                    pos_px1 = max(0, xmin)
                    pos_px2 = pos_px1 + pos_w + padding
                if pos_py1 < 0:
                    pos_py1 = 0
                    pos_py2 = pos_py1 + pos_h + padding
                if pos_px2 > img_w:
                    pos_px2 = img_w
                    pos_px1 = max(0, pos_px2 - pos_w - padding)
                if pos_py2 > img_h:
                    pos_py2 = img_h
                    pos_py1 = max(0, pos_py2 - pos_h - padding)

                pos_rect = (int(pos_px1), int(pos_py1), int(pos_px2), int(pos_py2))

                # 2) diag label (bottom-right of bbox) - keep one decimal place
                box_w = float(xmax - xmin)
                box_h = float(ymax - ymin)
                diag_px = math.sqrt(box_w**2 + box_h**2)
                diag_label = f'{diag_px:.1f}px'
                diag_size, diag_base = cv2.getTextSize(diag_label, FONT, DIAG_LABEL_FONT_SCALE, 1)
                diag_w, diag_h = diag_size

                diag_px2 = xmax
                diag_py2 = ymax
                diag_px1 = diag_px2 - diag_w - padding
                diag_py1 = diag_py2 - diag_h - padding

                # clamp inside image
                if diag_px1 < 0:
                    diag_px1 = max(0, xmin)
                    diag_px2 = diag_px1 + diag_w + padding
                if diag_py1 < 0:
                    diag_py1 = max(0, ymin)
                    diag_py2 = diag_py1 + diag_h + padding
                if diag_px2 > img_w:
                    diag_px2 = img_w
                    diag_px1 = max(0, diag_px2 - diag_w - padding)
                if diag_py2 > img_h:
                    diag_py2 = img_h
                    diag_py1 = max(0, diag_py2 - diag_h - padding)

                diag_rect = (int(diag_px1), int(diag_py1), int(diag_px2), int(diag_py2))

                # If pos_rect and diag_rect intersect inside the bbox, attempt to move one:
                def rects_intersect(r1, r2):
                    x11,y11,x12,y12 = r1
                    x21,y21,x22,y22 = r2
                    return not (x12 <= x21 or x22 <= x11 or y12 <= y21 or y22 <= y11)

                # Try resolving overlap: prefer keeping pos at top-right; move diag above bbox if possible.
                if rects_intersect(pos_rect, diag_rect):
                    # try move diag above bbox (y_top = ymin - diag_h - padding*2)
                    new_diag_py2 = ymin - 2  # just above bbox
                    new_diag_py1 = new_diag_py2 - diag_h - padding
                    if new_diag_py1 >= 0:
                        diag_py1 = int(new_diag_py1)
                        diag_py2 = int(new_diag_py2)
                        diag_px1 = int(max(0, diag_px2 - diag_w - padding))
                        diag_rect = (diag_px1, diag_py1, diag_px2, diag_py2)
                    else:
                        # else try moving pos label above bbox
                        new_pos_py2 = ymin - 2
                        new_pos_py1 = new_pos_py2 - pos_h - padding
                        if new_pos_py1 >= 0:
                            pos_py1 = int(new_pos_py1)
                            pos_py2 = int(new_pos_py2)
                            pos_px1 = int(max(0, pos_px2 - pos_w - padding))
                            pos_rect = (pos_px1, pos_py1, pos_px2, pos_py2)
                        else:
                            # as last resort, nudge diag left inside bbox
                            diag_px1 = int(max(xmin, diag_px1 - (pos_w + padding)))
                            diag_px2 = int(diag_px1 + diag_w + padding)
                            diag_rect = (diag_px1, diag_py1, diag_px2, diag_py2)

                # draw pos rect and text
                rect_tl = (int(pos_rect[0]), int(pos_rect[1]))
                rect_br = (int(pos_rect[2]), int(pos_rect[3]))
                cv2.rectangle(frame, rect_tl, rect_br, color, cv2.FILLED)
                text_x = rect_tl[0] + 2
                text_y = rect_br[1] - 4
                cv2.putText(frame, pos_label, (text_x, text_y), FONT, POS_LABEL_FONT_SCALE, (0,0,0), 1)

                # draw diag rect and text
                d_tl = (int(diag_rect[0]), int(diag_rect[1]))
                d_br = (int(diag_rect[2]), int(diag_rect[3]))
                cv2.rectangle(frame, d_tl, d_br, color, cv2.FILLED)
                d_text_x = d_tl[0] + 2
                d_text_y = d_br[1] - 4
                cv2.putText(frame, diag_label, (d_text_x, d_text_y), FONT, DIAG_LABEL_FONT_SCALE, (0,0,0), 1)

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
            cv2.putText(frame, f'FPS: {avg_fps:0.2f}', (10,20), FONT, .7, (0,255,255), 2)
        cv2.putText(frame, f'Persons: {object_count}', (10,40), FONT, .7, (0,255,255), 2)

        cv2.imshow('YOLO person-only', frame)
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
