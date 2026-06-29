# Multimodal Crime / Incident Report Analyzer

An end-to-end pipeline that ingests heterogeneous, **unstructured** incident
streams — audio 911 calls, official PDF reports, images, surveillance video, and
social/news text — runs a dedicated AI/heuristic extractor over each, and
programmatically merges all five structured outputs into **one unified incident
dataset**.

Every modality is extracted independently into its own CSV. The integration
layer then performs a vertical **UNION** of those CSVs into a single
schema-aligned master table, keyed by a modality-prefixed `Incident_ID`, with a
derived severity classification per row.

---

## Repository layout

```text
.
├── audio/
│   ├── AudioAnalyst.ipynb              # Whisper speech-to-text + event/urgency extraction
│   └── audio_extracted_report.csv      # → Incident_ID, Call_ID, Transcript, Extracted_Event, Location, Sentiment, Urgency_Score
├── pdf/
│   ├── extract.py                      # PDF→text (OCR), sub-document segmentation, field extraction
│   ├── pages_cache.json                # shipped OCR cache → rebuilds output.csv without the source PDF
│   └── output.csv                      # → Report_ID, Incident_Type, Date, Location, Officer, Summary, Suspect_Description, Outcome
├── images/
│   ├── image_analyzer.py               # YOLOv8 objects + fire/scene heuristic + OCR
│   ├── sample_images/                  # drop input images here to re-run (originals not shipped)
│   └── image_report.csv                # → Image_ID, Scene_Type, Objects_Detected, Text_Extracted, Confidence_Score
├── video/
│   ├── video_pipeline.py               # frame sampling + motion detection + YOLOv8 event classification
│   ├── yolov8n.pt                      # YOLOv8 model weights
│   ├── data/raw/*.mpg                  # shipped CAVIAR input clips
│   └── output/video_events.csv         # → Timestamp, Frame_ID, Event_Detected, Objects, Confidence
├── text/
│   ├── crime_nlp_pipeline.ipynb        # zero-shot topic classification + sentiment + NER
│   ├── CrimeReport.txt                 # shipped input text
│   └── crime_text_report.csv           # → Text_ID, Source, Raw_Text, Sentiment, Entities, Topic
├── integration/
│   ├── build_dataset.py                # ★ merges all five CSVs into the unified dataset
│   ├── dashboard.py                    # ★ Streamlit dashboard for exploring the unified dataset
│   └── master_incident_dataset.csv     # ★ final unified incident dataset
├── README.md                           # this file
└── requirements.txt                    # consolidated dependencies for the whole pipeline
```

The five modality folders are self-contained: each holds its extraction **code**
and its **output CSV**. `integration/` holds only the orchestration code and the
merged result.

---

## The integration layer

`integration/build_dataset.py` is the most important deliverable. It implements
the four required steps:

1. **Prefixed IDs** — every row gets a unique, modality-prefixed `Incident_ID`
   (`AUD-`, `DOC-`, `IMG-`, `VID-`, `TXT-`). Each modality's own source ID
   (`Call_ID`, `Report_ID`, `Image_ID`, `Frame_ID`, `Text_ID`) is preserved in
   the `Source_Record_ID` column.
2. **UNION** — the five normalised DataFrames are concatenated with
   `pandas.concat` into one common schema, **one row per source record**. The
   modalities describe *different* incidents, so this is a vertical union, not a
   key join.
3. **Missing-value defaults** — any field a modality does not provide is filled
   with `N/A` (e.g. images and video have no `Location`; only PDF/video carry a
   `Time`).
4. **Severity classification** — a `Severity_Score` (0–10) and `Severity_Level`
   (Low / Medium / High) are derived from each row's own confidence/urgency
   signal (see below).

### Unified schema

| Column | Meaning |
|---|---|
| `Incident_ID` | Modality-prefixed unique id (`AUD-001`, `DOC-001`, …) |
| `Source` | Modality label (Audio / PDF / Image / Video / Text) |
| `Source_Record_ID` | The modality's own record id |
| `Event` | What happened (mapped from each modality's event/type field) |
| `Location` | Where — best-effort, may be `N/A` |
| `Time` | When — best-effort, may be `N/A` |
| `Severity_Score` | 0–10 numeric score |
| `Severity_Level` | Low / Medium / High |
| `Details` | Extra modality-specific context |

### Field mapping (per modality → unified)

| Unified | Audio | PDF | Image | Video | Text |
|---|---|---|---|---|---|
| `Event` | `Extracted_Event` | `Incident_Type` | `Scene_Type` | `Event_Detected` | `Topic` |
| `Location` | `Location` | `Location` | `N/A` | `N/A` | parsed `(Location)` entities |
| `Time` | `N/A` | `Date` | `N/A` | `Timestamp` | `N/A` |
| confidence signal | `Urgency_Score` | document-type priority | `Confidence_Score` | `Confidence` | derived from `Sentiment` |

### Severity scale (also stated in the report)

```text
Severity_Score = confidence_signal × 10          (range 0 – 10)

    0  ≤ score < 3    →  Low
    3  ≤ score < 7    →  Medium
    7  ≤ score ≤ 10   →  High
```

`confidence_signal` (0.0–1.0) per modality:

- **Audio** → `Urgency_Score` directly.
- **Image** → `Confidence_Score` directly.
- **Video** → `Confidence` directly.
- **PDF** → a document is a record, not an active incident, so it has no
  numeric urgency signal. Severity is ranked by **document-type priority**:
  event-bearing documents (`Memorandum`, `Letter`, `*Report`) → **0.50**
  (Medium); routine administrative paperwork (cover letters, certificates,
  policies, invoices, training/course documents, forms) → **0.25** (Low).
- **Text** → mapped from sentiment (`Negative` 0.70, `Neutral` 0.40,
  `Positive` 0.20), boosted by **+0.15** when the topic is a violent /
  high-priority category (capped at 1.0).

---

## How to run

Each modality is **self-contained** — it has its own extraction code and writes
its own CSV into its folder. The integration step only reads those five CSVs, so
you can run any piece independently. All scripts resolve paths relative to
themselves, so the commands below work from the repository root regardless of
your shell's current directory.

### 0. Install dependencies (once)

```bash
pip install -r requirements.txt          # see the file header for system packages
python -m spacy download en_core_web_sm  # used by pdf / text / audio
python -m nltk.downloader stopwords punkt # used by text
```

### 1. Run a single modality (each is optional and independent)

| Modality | Command | Input it reads | Output it writes |
|---|---|---|---|
| **Video** | `python video/video_pipeline.py` | `video/data/raw/*.mpg` *(shipped)* + `video/yolov8n.pt` *(shipped)* | `video/output/video_events.csv` |
| **PDF** | `python pdf/extract.py` | `pdf/pages_cache.json` *(shipped OCR cache)* | `pdf/output.csv` |
| **Image** | `python images/image_analyzer.py` | images placed in `images/sample_images/` | `images/image_report.csv` |
| **Audio** | open `audio/AudioAnalyst.ipynb`, run all cells | raw 911-call audio *(not shipped)* | `audio/audio_extracted_report.csv` |
| **Text** | open `text/crime_nlp_pipeline.ipynb`, run all cells | `text/CrimeReport.txt` *(shipped)* | `text/crime_text_report.csv` |

Reproducibility notes:
- **Video** and **PDF** reproduce their CSVs out of the box from shipped inputs
  (PDF rebuilds from the cached OCR, however, the original source PDF is also provided).
- **Image** ships its `image_report.csv`, but the original images are not
  redistributed — drop your own images into `images/sample_images/` to re-run.
- **Audio** and **Text** are Colab-style notebooks; their output CSVs are
  shipped, so the integration step runs even without re-executing them.

### 2. Build the unified dataset (always runnable — all five CSVs are shipped)

```bash
python integration/build_dataset.py
```

Reads the five modality CSVs and writes:

```text
integration/master_incident_dataset.csv
```

### 3. Launch the dashboard

```bash
streamlit run integration/dashboard.py
```

Opens at **http://localhost:8501** (or the next available port). Press **Ctrl + C** to stop.

---

## Dashboard

`integration/dashboard.py` is an interactive Streamlit web app that reads
`integration/master_incident_dataset.csv` and lets you explore all 1,031
unified incidents with live filters and charts.

### Prerequisites

`streamlit` and `plotly` are already in `requirements.txt` and installed by
`pip install -r requirements.txt`. To install only these two:

```bash
pip install streamlit>=1.35.0 plotly>=5.18.0
```

`master_incident_dataset.csv` is shipped in the repo, so the dashboard runs
immediately without re-executing any extraction pipeline.

### Optional launch flags

| Flag | Purpose | Example |
|------|---------|---------|
| `--server.port` | Change port (default 8501) | `--server.port 8080` |
| `--server.headless true` | Suppress auto-open browser | `--server.headless true` |
| `--theme.base dark` | Force dark theme | `--theme.base dark` |

```bash
streamlit run integration/dashboard.py --server.port 8080 --server.headless true
```

### Sidebar filters

| Control | What it does |
|---------|-------------|
| **Data Source** multiselect | Show only Audio / PDF / Image / Video / Text incidents |
| **Severity Level** multiselect | Filter by Low / Medium / High |
| **Severity Score** slider | Narrow to a 0–10 score range |
| **Event Type** multiselect | Pick one or more specific event categories |
| **Search box** | Free-text search across Event, Location, and Details columns |

All filters are live — every chart and the data table update instantly.

### Main area sections

| Section | Description |
|---------|-------------|
| **KPI cards** | Total incidents · High · Medium · Low counts · Average Severity Score |
| **Incidents by Source** | Donut chart — share of each modality |
| **Severity by Source** | Stacked bar — Low / Medium / High per source |
| **Top 15 Event Types** | Horizontal bar chart of the most frequent events |
| **Severity Score Distribution** | Stacked histogram with Low→Medium and Medium→High threshold lines |
| **Severity Breakdown** | Grouped bar — Low, Medium, High side-by-side per source |
| **Modality Spotlight tabs** | One tab per source: KPIs, top-events bar, severity-mix donut, expandable high-severity table |
| **All Incident Records table** | Scrollable table with colour-coded Severity column (all 9 unified fields) |
| **Download CSV** | Export the currently filtered rows as a CSV file |

### Refreshing the data

After re-running any modality extractor, regenerate the master dataset and
Streamlit will auto-detect the file change:

```bash
python integration/build_dataset.py   # regenerate master_incident_dataset.csv
streamlit run integration/dashboard.py
```

---

### Current run summary

- **1,031** total incidents — Audio 707, Video 164, Text 115, PDF 42, Image 3
- Severity distribution — Medium 733, High 174, Low 124
