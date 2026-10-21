"""Compute MOT metrics from SoccerTrack ground truth and src/track_video.py output."""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd
import motmetrics as mm

# motmetrics versions that still call np.asfarray need this compatibility alias.
if not hasattr(np, "asfarray"):
    np.asfarray = lambda value, dtype=float: np.asarray(value, dtype=dtype)

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("ground_truth", help="SoccerTrack MOT text file (comma separated, no header)")
    parser.add_argument("predictions", help="CSV produced by track_video.py")
    parser.add_argument("--frames", type=int, help="Evaluate only the first N frames")
    args = parser.parse_args()
    gt = pd.read_csv(args.ground_truth, header=None)
    if gt.shape[1] < 6: parser.error("ground truth must contain frame, id, x, y, width, height columns")
    gt = gt.iloc[:, :6].copy()
    gt.columns = ["frame", "id", "x", "y", "width", "height"]
    pred = pd.read_csv(args.predictions)
    pred = pred[["frame", "track_id", "x", "y", "width", "height"]].rename(columns={"track_id":"id"})
    if args.frames:
        gt, pred = gt[gt.frame <= args.frames], pred[pred.frame <= args.frames]
    accumulator = mm.MOTAccumulator(auto_id=False)
    frames = sorted(set(gt.frame.astype(int)) | set(pred.frame.astype(int)))
    for frame in frames:
        g, p = gt[gt.frame == frame], pred[pred.frame == frame]
        g_boxes = g[["x", "y", "width", "height"]].to_numpy()
        p_boxes = p[["x", "y", "width", "height"]].to_numpy()
        distances = mm.distances.iou_matrix(g_boxes, p_boxes, max_iou=0.5)
        accumulator.update(g.id.tolist(), p.id.tolist(), distances, frameid=int(frame))
    summary = mm.metrics.create().compute(accumulator, metrics=["mota", "idf1", "num_switches", "precision", "recall", "num_false_positives", "num_misses"], name="tracking")
    print(summary.to_string(float_format=lambda v: f"{v:.4f}"))

if __name__ == "__main__": main()
