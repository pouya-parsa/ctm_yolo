import cv2
import os
import json
import argparse
from typing import List, Tuple

Point = Tuple[int, int]


def load_first_frame(cap: cv2.VideoCapture):
    ret, frame = cap.read()
    if not ret:
        raise RuntimeError("Cannot read video")
    return frame


def draw_feedback(img, pts: List[Point]):
    vis = img.copy()
    # color points
    colors = [(255, 0, 0)] * 2 + [(0, 0, 255)] * 2  # BGR: blue for upstream, red for downstream
    for i, p in enumerate(pts):
        cv2.circle(vis, p, 5, colors[i], -1)
    # draw upstream line
    if len(pts) >= 2:
        cv2.line(vis, pts[0], pts[1], (255, 0, 0), 2)
    # draw downstream line
    if len(pts) == 4:
        cv2.line(vis, pts[2], pts[3], (0, 0, 255), 2)
        # draw connecting lines in green
        cv2.line(vis, pts[0], pts[2], (0, 255, 0), 2)
        cv2.line(vis, pts[1], pts[3], (0, 255, 0), 2)
    return vis


def collect_line_points(frame) -> Tuple[Point, Point, Point, Point]:
    pts: List[Point] = []
    window = "Select 4 pts (Upstream then Downstream)"

    def on_click(evt, x, y, *_):
        if evt == cv2.EVENT_LBUTTONDOWN and len(pts) < 4:
            pts.append((x, y))

    cv2.namedWindow(window)
    cv2.setMouseCallback(window, on_click)

    while True:
        cv2.imshow(window, draw_feedback(frame, pts))
        key = cv2.waitKey(10) & 0xFF
        if key == 27:
            cv2.destroyAllWindows()
            raise RuntimeError("Selection aborted by user")
        elif key == ord('r'):
            pts.clear()
        elif (key == 13 or key == 10) and len(pts) == 4:
            break
    cv2.destroyAllWindows()

    if len(pts) != 4:
        raise RuntimeError("Need exactly 4 clicks (2 per line)")
    return pts[0], pts[1], pts[2], pts[3]


def save_boundary(video_path: str, p1: Point, p2: Point, p3: Point, p4: Point):
    os.makedirs("boundary", exist_ok=True)
    name = os.path.splitext(os.path.basename(video_path))[0]
    out = os.path.join("boundary", f"{name}.json")
    with open(out, "w", encoding="utf-8") as f:
        json.dump({
            "upstream": [list(p1), list(p2)],
            "downstream": [list(p3), list(p4)]
        }, f, indent=2)
    print(f"Saved → {out}")


def main():
    ap = argparse.ArgumentParser(description="Mark road cell boundaries")
    ap.add_argument("video", help="Input video path")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.video)
    frame = load_first_frame(cap)
    p1, p2, p3, p4 = collect_line_points(frame)
    save_boundary(args.video, p1, p2, p3, p4)


if __name__ == "__main__":
    main()
