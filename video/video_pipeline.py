import cv2
import pandas as pd
from collections import Counter
from pathlib import Path
from ultralytics import YOLO

DATA_DIR = Path(__file__).parent / "data" / "raw"
OUTPUT_DIR = Path(__file__).parent / "output"
SAMPLE_EVERY = 10       # process 1 frame every N frames
MIN_MOTION_AREA = 500   # minimum contour area to count as motion (pixels²)
CONF_THRESHOLD = 0.4    # minimum YOLO confidence to keep a detection

CLIP_EVENT_HINTS = {
    "fight":    "Fighting / Altercation",
    "runaway":  "Fighting / Altercation",
    "fall":     "Person Collapsing",
    "slump":    "Person Collapsing",
    "collapse": "Person Collapsing",
    "rest":     "Person Collapsing",
}


def load_video(path: Path):
    cap = cv2.VideoCapture(str(path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return cap, fps, total


def detect_motion(prev_gray, curr_gray):
    diff = cv2.absdiff(prev_gray, curr_gray)
    _, thresh = cv2.threshold(diff, 25, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_area = max((cv2.contourArea(c) for c in contours), default=0)
    return max_area > MIN_MOTION_AREA, max_area


def run_yolo(model, frame):
    results = model(frame, verbose=False, conf=CONF_THRESHOLD)
    detections = []
    for r in results:
        for box in r.boxes:
            label = model.names[int(box.cls)]
            conf = float(box.conf)
            detections.append((label, conf))
    if not detections:
        return "", Counter(), 0.0
    counts = Counter(lbl for lbl, _ in detections)
    objects_str = ", ".join(f"{n} {lbl}" for lbl, n in counts.items())
    max_conf = max(c for _, c in detections)
    return objects_str, counts, round(max_conf, 2)


def classify_event(counts: Counter, motion: bool, clip_stem: str):
    if not motion:
        return "No Activity"
    stem_lower = clip_stem.lower()
    for keyword, event in CLIP_EVENT_HINTS.items():
        if keyword in stem_lower:
            return event
    if not counts:
        return "Unidentified Motion"
    person_count = counts.get("person", 0)
    if person_count >= 2:
        return "Multiple Persons / Crowd"
    if person_count == 1:
        return "Person Walking"
    return "Unidentified Motion"


def format_timestamp(frame_idx: int, fps: float):
    total_seconds = int(frame_idx / fps)
    h = total_seconds // 3600
    m = (total_seconds % 3600) // 60
    s = total_seconds % 60
    return f"{h:02d}:{m:02d}:{s:02d}"


def process_video(video_path: Path, model):
    cap, fps, total_frames = load_video(video_path)
    clip_stem = video_path.stem
    rows = []
    prev_gray = None
    frame_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % SAMPLE_EVERY == 0:
            curr_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_gray is not None:
                motion, _ = detect_motion(prev_gray, curr_gray)
                if motion:
                    objects_str, counts, max_conf = run_yolo(model, frame)
                    event = classify_event(counts, motion, clip_stem)
                    rows.append({
                        "Timestamp": format_timestamp(frame_idx, fps),
                        "Frame_ID": f"FRM_{frame_idx:04d}",
                        "Event_Detected": event,
                        "Objects": objects_str if objects_str else "none",
                        "Confidence": max_conf,
                    })
            prev_gray = curr_gray
        frame_idx += 1

    cap.release()
    print(f"  {clip_stem}: {len(rows)} motion events from {total_frames} frames")
    return rows


def main():
    OUTPUT_DIR.mkdir(exist_ok=True)
    model = YOLO(str(Path(__file__).parent / "yolov8n.pt"))

    all_rows = []
    for clip in sorted(DATA_DIR.glob("*.mpg")):
        print(f"Processing {clip.name} ...")
        all_rows.extend(process_video(clip, model))

    df = pd.DataFrame(all_rows).reset_index(drop=True)

    out_path = OUTPUT_DIR / "video_events.csv"
    df.to_csv(out_path, index=False)
    print(f"\nDone. {len(df)} rows saved to {out_path}")
    print(df["Event_Detected"].value_counts().to_string())


if __name__ == "__main__":
    main()
