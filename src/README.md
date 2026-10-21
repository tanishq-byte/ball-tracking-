# Source scripts

- `track_video.py` runs a supplied Ultralytics detector with persistent ByteTrack IDs and writes frame-indexed bounding boxes to CSV.
- `evaluate_tracking.py` compares that CSV with a SoccerTrack MOT annotation file using `motmetrics` and IoU matching.

The notebook contains the full Colab workflow for downloading the two source matches, preparing detector training labels, fine-tuning, ROI experiments, and evaluations. The helper scripts are a small command-line entry point for inference/evaluation; they do not reproduce every notebook experiment. Ground-truth format assumptions follow the notebook's SoccerTrack MOT file: frame, identity, x, y, width, height in the first six comma-separated columns.
