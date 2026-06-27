# Student 4 — Video Analyst Module

Processes CCTV surveillance footage from the CAVIAR dataset to detect motion events
and produce a structured incident report CSV.

## Setup

```bash
conda activate incident-analyzer
pip install -r requirements.txt
```

## Dataset

5 clips from the [CAVIAR CCTV Dataset](http://homepages.inf.ed.ac.uk/rbf/CAVIARDATA1/)
stored in `data/raw/`:

| File | Scenario |
|---|---|
| Browse1.mpg | Person browsing / walking |
| Meet_Crowd.mpg | Multiple persons / crowd |
| Rest_FallOnFloor.mpg | Person collapsing |
| Fight_RunAway1.mpg | Fighting / running away |
| Fight_OneManDown.mpg | Fighting / person down |

## How to Run

```bash
python video_pipeline.py
```

Output is saved to `output/video_events.csv`.

## Pipeline Steps

1. **Frame extraction** — samples 1 frame every 10 frames (~1 fps)
2. **Motion detection** — frame difference + contour area threshold (500 px²)
3. **Object detection** — YOLOv8n on motion frames (confidence > 0.4)
4. **Event classification** — maps detected objects + clip name to event label
5. **CSV generation** — one row per motion event with VID- prefixed Incident_ID

## Output Schema

| Column | Description |
|---|---|
| Incident_ID | VID-001, VID-002, ... |
| Video_File | Source .mpg filename |
| Timestamp | HH:MM:SS of the frame |
| Frame_ID | FRM_XXXX (frame number) |
| Event_Detected | Event label |
| Objects | YOLO detections with confidence |
| Confidence | Max YOLO confidence (0.0–1.0) |
| Severity | Low / Medium / High |

## Severity Thresholds

| Confidence | Severity |
|---|---|
| 0.7 – 1.0 | High |
| 0.3 – 0.7 | Medium |
| 0.0 – 0.3 | Low |

## Integration (for Student 6)

The CSV uses `VID-` prefixed Incident_IDs. Columns that map to the unified schema:

| Unified column | This module's column |
|---|---|
| Source | "Video" |
| Event | Event_Detected |
| Location | N/A (CAVIAR has no location metadata) |
| Time | Timestamp |
| Severity | Severity |
