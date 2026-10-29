# Football CV — player and ball tracking

## Latest progress: 29 October 2026

The complete dated backup is in [`progress/2026-10-29`](progress/2026-10-29/README.md), including the ball-tracking notebook, player checkpoint, tracking outputs, 40 benchmark frames and annotations, evaluation CSVs, and motion-tracklet experiments. The original ZIP contents are preserved; `snapshot_integrity.json` records their SHA-256 hashes.

- **Player tracking:** fine-tuned YOLO26n V1 checkpoint and ByteTrack configuration, predictions for segments A/B/C, and development metrics.
- **Ball benchmark:** 40 annotated frames, including 36 valid visible-ball annotations and 4 uncertain/excluded frames.
- **Initial tracking evaluation:** predictions on 11 of 36 valid frames (**30.56% position coverage**); mean center error **2.16 pixels on those 11 predictions**. This is not 97% accuracy and does not measure performance on the 25 missing frames.
- **Diagnostic experiments:** local oracle detection and V8 association use ground truth as guidance. Their recovery results are diagnostic and must not be presented as autonomous test accuracy.
- **V9:** automatic motion-tracklet discovery does not use ground truth for association; subsequent GT comparisons are evaluation. It remains an experiment rather than a validated full-match tracker.

The notebook prompts for the Roboflow API key using `getpass`; supply your own key when executing. It retains Colab/Google Drive paths, so local execution requires adapting those paths. This is an archived progress snapshot, not a standalone installed application. Full source videos and training datasets are not included. The backup includes extracted benchmark images, annotations and model weights.

These experiments use development footage; independent match-level generalization remains unverified. The original Phase 1 documentation below is preserved as historical context. Its statement that weights are not included refers to the original delivery; today's snapshot now includes the player checkpoint.

---

# Football player tracking (Phase 1)

A reproducible starting point for detecting and tracking football players in broadcast video. The Colab workflow samples SoccerTrack v2 match footage, fine-tunes a YOLO26n detector, explores a pitch region of interest (ROI), and compares multi-object tracking configurations. The selected tracking configuration is Ultralytics ByteTrack.

## Repository contents

- `notebooks/player_tracking.ipynb` — source Colab notebook with outputs and execution counters removed. Run it in Google Colab; GPU is recommended for detector training.
- `configs/bytetrack_optimized.yaml` — ByteTrack parameters selected in the notebook's development experiments.
- `results/tracking_metrics.csv` — the three displayed 300-frame segment results from the notebook.
- `src/track_video.py` — run tracking with a trained detector and export track boxes.
- `src/evaluate_tracking.py` — compute MOT metrics against SoccerTrack annotations.

## Dataset

The notebook downloads clips and MOT labels from the Hugging Face dataset [`atomscott/soccertrack-v2`](https://huggingface.co/datasets/atomscott/soccertrack-v2). It uses match `117093` for development/validation and `117092` for detector training. Review the dataset card and its license/access terms before use. Video and annotations are fetched by the notebook rather than checked into this repository.

The Colab source is [Player Tracking notebook](https://colab.research.google.com/drive/1PuBsVHdbO5jcQRLB2AVl_JVKPaiE7JYy?usp=sharing).

## Setup and run

Use Python 3.10 or newer. Install the notebook dependencies in a Colab GPU runtime, or locally in a virtual environment:

```bash
python -m pip install ultralytics opencv-python huggingface-hub pandas motmetrics scipy pyyaml
```

Download or train weights separately. The notebook saves its fine-tuned checkpoint in Google Drive as `Football_CV/models/yolo26n_football_v1_best.pt`; model weights are not included here.

Run inference from the project root:

```bash
python src/track_video.py /path/to/video.mp4 \
  --weights /path/to/yolo26n_football_v1_best.pt \
  --tracker configs/bytetrack_optimized.yaml \
  --output results/tracks.csv
```

Evaluate against a SoccerTrack MOT annotation file:

```bash
python src/evaluate_tracking.py /path/to/117093.txt results/tracks.csv --frames 300
```

The prediction CSV uses 1-based frame numbers and `frame,track_id,x,y,width,height,confidence,class_id` columns. Evaluation uses the first six ground-truth columns as `frame,id,x,y,width,height` and IoU matching at 0.5.

## Development results

The notebook reports these results for three 300-frame segments from match `117093`:

| Segment | MOTA | IDF1 | ID switches |
| --- | ---: | ---: | ---: |
| A | 77.6% | 72.3% | 58 |
| B | 68.3% | 68.6% | 106 |
| C | 81.4% | 74.7% | 48 |

The CSV also records displayed precision, recall, false positives, and false negatives. Values are transcribed from the notebook's rounded output, not recalculated from raw predictions. Segment B was used while tuning tracker parameters, and all three segments come from the development/validation match; treat these as development results rather than independent test performance. The notebook's evaluation applies its manually selected ROI mask before scoring. Re-run evaluation for exact values and document matching settings when comparing future experiments.

## Current scope and next steps

Phase 1 focuses on player detection and persistent IDs. It does not yet provide calibrated pitch coordinates, team/role identification, tactical event interpretation, or a RAG analyst. A useful next step is to add held-out matches, track quality checks, pitch calibration, and structured event extraction before grounding tactical answers in retrieved match reports or verified statistics.
