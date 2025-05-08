import os
import cv2
import json
import argparse
import numpy as np
from ultralytics import YOLO
from typing import Tuple, List

Point = Tuple[float, float]
LineCoeffs = Tuple[float, float, float]
INITIAL_DOWNSTREAM_OFFSET = 5


def load_lines(path: str) -> Tuple[Point, Point, Point, Point]:
    with open(path) as f:
        d = json.load(f)
    return (*map(tuple, d['upstream']), *map(tuple, d['downstream']))


def line_eq(p1: Point, p2: Point) -> LineCoeffs:
    """
    Compute line coefficients (a, b, c) such that a*x + b*y + c = 0
    passes through p1 and p2.
    """
    a = p1[1] - p2[1]
    b = p2[0] - p1[0]
    c = p1[0] * p2[1] - p2[0] * p1[1]
    return a, b, c


def init_detector(model: str = 'yolov8m.pt'):
    return YOLO(model)


def setup_io(video: str):
    cap = cv2.VideoCapture(video)
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    os.makedirs('output', exist_ok=True)
    base = os.path.splitext(os.path.basename(video))[0]
    out_vid = f'output/{base}_vis.mp4'
    out_json = f'output/{base}_counts.json'
    writer = cv2.VideoWriter(out_vid, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
    return cap, writer, out_vid, out_json, fps


def is_between(pt: Point, l1: LineCoeffs, l2: LineCoeffs) -> bool:
    x, y = pt
    return (l1[0]*x + l1[1]*y + l1[2]) * (l2[0]*x + l2[1]*y + l2[2]) < 0


def draw(frame, p1, p2, p3, p4, boxes: List, up_cum: int, down_cum: int):
    # draw main segments
    cv2.line(frame, tuple(map(int,p1)), tuple(map(int,p2)), (255,0,0), 2)
    cv2.line(frame, tuple(map(int,p3)), tuple(map(int,p4)), (0,0,255), 2)
    # draw side boundaries
    cv2.line(frame, tuple(map(int,p1)), tuple(map(int,p3)), (0,255,0), 2)
    cv2.line(frame, tuple(map(int,p2)), tuple(map(int,p4)), (0,255,0), 2)
    # draw boxes
    for vid, (x1,y1,x2,y2), _ in boxes:
        cv2.rectangle(frame, (int(x1),int(y1)), (int(x2),int(y2)), (0,255,0), 2)
        cv2.putText(frame, str(vid), (int((x1+x2)/2), int((y1+y2)/2)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
    # large counters top-right
    margin = 200
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 10
    thick = 8
    text_up = f'Up: {up_cum}'
    (w_up, h_up), _ = cv2.getTextSize(text_up, font, scale, thick)
    x_up = frame.shape[1] - margin - w_up
    y_up = margin + h_up
    text_down = f'Down: {down_cum}'
    (w_dn, h_dn), _ = cv2.getTextSize(text_down, font, scale, thick)
    x_dn = frame.shape[1] - margin - w_dn
    y_dn = y_up + margin + h_dn
    cv2.putText(frame, text_up, (x_up, y_up), font, scale, (255,255,255), thick)
    cv2.putText(frame, text_down, (x_dn, y_dn), font, scale, (255,255,255), thick)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('video')
    p.add_argument('-b','--boundary', default=None)
    p.add_argument('-t','--time', type=float, default=60.0,
                   help='total time window in seconds')
    args = p.parse_args()

    COUNT_START_SEC = 10.4                      # ---------- new ----------
    cap, writer, out_vid, out_json, fps = setup_io(args.video)
    start_frame     = int(fps * COUNT_START_SEC)  # first frame to count
    max_frames      = int(fps * args.time)

    prev_up   = {}
    prev_down = {}
    count_up  = set()
    count_down = set()
    time_counts = {}
    frame_no = 0

    bfile = args.boundary or f'boundary/{os.path.splitext(os.path.basename(args.video))[0]}.json'
    p1, p2, p3, p4 = load_lines(bfile)
    # compute line equations
    l_up = line_eq(p1, p2)
    l_down = line_eq(p3, p4)
    l_left = line_eq(p1, p3)
    l_right = line_eq(p2, p4)

    model = init_detector()
    cap, writer, out_vid, out_json, fps = setup_io(args.video)
    max_frames = int(fps * args.time)

    prev_up = {}
    prev_down = {}
    count_up = set()
    count_down = set()
    time_counts = {}
    frame_no = 0

    for res in model.track(source=args.video, classes=[2,5,7], stream=True):
        frame_no += 1
        if frame_no > max_frames:
            break
        frame = res.orig_img
        boxes = []
        for box in res.boxes:
            vid = int(box.id)
            x1, y1, x2, y2 = box.xyxy[0]
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            pt = (cx, cy)
            if not is_between(pt, l_left, l_right):
                continue

            v_up   = l_up[0]   * cx + l_up[1]   * cy + l_up[2]
            v_down = l_down[0] * cx + l_down[1] * cy + l_down[2]

            # --------------------------------------------------------
            # ➋  Only *record* a count after the warm‑up period
            # --------------------------------------------------------
            if frame_no > start_frame:
                if vid in prev_up   and prev_up[vid]   * v_up   < 0:
                    count_up.add(vid)
                if vid in prev_down and prev_down[vid] * v_down < 0:
                    count_down.add(vid)

            prev_up[vid]   = v_up
            prev_down[vid] = v_down

            if is_between(pt, l_up, l_down):
                boxes.append((vid, (x1, y1, x2, y2), pt))
        sec       = (frame_no - 1) // int(fps) + 1
        if sec <= COUNT_START_SEC:
            raw_up, adj_down = 0, 0
        else:
            raw_up     = len(count_up)
            raw_down   = len(count_down)
            adj_down   = max(raw_down - INITIAL_DOWNSTREAM_OFFSET, 0)

        time_counts[sec] = (raw_up, adj_down)
        draw(frame, p1, p2, p3, p4, boxes, raw_up, adj_down)
        writer.write(frame)

    writer.release()
    with open(out_json, 'w') as f:
        json.dump({str(k): v for k, v in time_counts.items()}, f, indent=2)
    print(f'Saved video → {out_vid}')
    print(f'Saved counts → {out_json}')

if __name__=='__main__':
    main()