"""Run persistent Ultralytics tracking on a football video and write MOT-style CSV."""
from __future__ import annotations
import argparse
import csv
from pathlib import Path
from ultralytics import YOLO

FIELDS = ["frame", "track_id", "x", "y", "width", "height", "confidence", "class_id"]

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("video", type=Path)
    parser.add_argument("--weights", type=Path, required=True, help="Detector .pt weights")
    parser.add_argument("--tracker", type=Path, default=Path("configs/bytetrack_optimized.yaml"))
    parser.add_argument("--output", type=Path, default=Path("results/tracks.csv"))
    parser.add_argument("--imgsz", type=int, default=1536)
    parser.add_argument("--conf", type=float, default=0.35)
    args = parser.parse_args()
    if not args.video.is_file(): parser.error(f"video not found: {args.video}")
    if not args.weights.is_file(): parser.error(f"weights not found: {args.weights}")
    if not args.tracker.is_file(): parser.error(f"tracker config not found: {args.tracker}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(args.weights))
    with args.output.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        for frame_no, result in enumerate(model.track(
            source=str(args.video), tracker=str(args.tracker), persist=True,
            stream=True, imgsz=args.imgsz, conf=args.conf, verbose=False,
        ), start=1):
            boxes = result.boxes
            if boxes is None or boxes.id is None:
                continue
            xyxy = boxes.xyxy.cpu().tolist()
            ids = boxes.id.int().cpu().tolist()
            confs = boxes.conf.cpu().tolist()
            classes = boxes.cls.int().cpu().tolist()
            for (x1, y1, x2, y2), track_id, conf, cls in zip(xyxy, ids, confs, classes):
                writer.writerow({"frame": frame_no, "track_id": track_id, "x": x1, "y": y1,
                                 "width": x2-x1, "height": y2-y1, "confidence": conf, "class_id": cls})
    print(f"Wrote tracks to {args.output}")

if __name__ == "__main__": main()
