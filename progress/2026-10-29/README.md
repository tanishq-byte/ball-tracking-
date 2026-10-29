# Football CV - Progress Backup (29 October 2026)

## Project
Panoramic football video analysis using YOLO, player tracking,
team classification, ball tracking and temporal motion analysis.

## Completed
- Player YOLO fine-tuning and ByteTrack experiments
- Team jersey classification experiments
- Trained football-specific YOLO ball detector
- Ball detector evaluation on Roboflow test dataset
- Segment C tracking consolidation (frames 4000-4299)
- Ground-truth benchmark: 40 frames
- 36 visible-ball annotations and 4 uncertain annotations
- Tracking evaluation: 11/36 positions, mean error 2.16 px
- Oracle local detection: 11/25 additional missed frames recovered
- Background-subtraction airborne-ball experiments
- V8 ground-truth-guided motion association
- V9 automatic motion tracklets
- V10 flight-segment diagnostic, if saved in session

## Key distinction
The V9 trajectory-discovery experiment does not use GT for association.
V8 and local oracle inference are GT-guided diagnostic experiments.

## Data
Original SoccerTrack videos, large datasets, and model training data
are not included in this GitHub ZIP.

## Caution
Review notebooks and configuration files for credentials before
pushing to a public GitHub repository.
