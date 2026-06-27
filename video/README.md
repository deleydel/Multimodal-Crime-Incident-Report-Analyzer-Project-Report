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
5. **CSV generation** — one row per motion event

## Output Schema

| Column | Description |
|---|---|
| Timestamp | HH:MM:SS of the frame |
| Frame_ID | FRM_XXXX (frame number) |
| Event_Detected | Event label (e.g. Person Walking, Fighting / Altercation) |
| Objects | Count-based YOLO detections (e.g. `2 person, 1 car`) |
| Confidence | Max YOLO confidence (0.0–1.0); 0.0 if no detection |
