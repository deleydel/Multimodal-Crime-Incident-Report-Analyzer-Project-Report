# Integration & Dashboard — Setup and Run Guide

This folder contains two deliverables for **Stage 4 (Structured Dataset Generation)**
and **Stage 5 (Dashboard / Query System)** of the Multimodal Crime Incident Report Analyzer:

| File | Purpose |
|------|---------|
| `build_dataset.py` | Merges the five modality CSVs into one unified master dataset |
| `master_incident_dataset.csv` | The merged output — 1,031 incidents across all five sources |
| `dashboard.py` | Streamlit web dashboard for interactive exploration of the master dataset |

---

## Prerequisites

`streamlit` and `plotly` are listed in the repo-root `requirements.txt`.
Install everything at once from the **repository root**:

```bash
pip install -r requirements.txt
```

Or install only the dashboard dependencies:

```bash
pip install streamlit>=1.35.0 plotly>=5.18.0
```

No other setup is required. `master_incident_dataset.csv` is already shipped in
this folder, so the dashboard works immediately without re-running any extractor.

---

## Running the dashboard

Always run from the **repository root** (one level above this folder) so that
the internal file paths resolve correctly:

```bash
# from the repo root
streamlit run integration/dashboard.py
```

Streamlit prints the URL when it starts:

```
  You can now view your Streamlit app in your browser.

  Local URL:    http://localhost:8501
  Network URL:  http://<your-ip>:8501
```

Open the **Local URL** in any browser. Press **Ctrl + C** in the terminal to stop.

### Optional flags

| Flag | Purpose | Example |
|------|---------|---------|
| `--server.port` | Change the port (default 8501) | `--server.port 8080` |
| `--server.headless true` | Prevent the browser from opening automatically | `--server.headless true` |
| `--theme.base dark` | Force dark theme regardless of system settings | `--theme.base dark` |

```bash
# Example: custom port, no auto-open
streamlit run integration/dashboard.py --server.port 8080 --server.headless true
```

---

## Dashboard features

### Sidebar (filters — always visible)

| Control | What it does |
|---------|-------------|
| **Data Source** multiselect | Show only Audio / PDF / Image / Video / Text incidents |
| **Severity Level** multiselect | Filter by Low / Medium / High |
| **Severity Score slider** | Narrow results to a 0–10 score range |
| **Event Type** multiselect | Pick one or more specific event categories |
| **Search box** | Free-text search across Event, Location, and Details columns |

All filters are live — charts and the table update instantly.

### Main area — top to bottom

| Section | Description |
|---------|-------------|
| **KPI cards** | Total incidents · High · Medium · Low counts · Average Severity Score |
| **Incidents by Data Source** | Donut chart — share of each modality |
| **Severity Distribution by Source** | Stacked bar — Low / Medium / High per source |
| **Top 15 Event Types** | Horizontal bar chart of the most frequent events |
| **Severity Score Distribution** | Stacked histogram with Low → Medium and Medium → High threshold lines |
| **Severity Breakdown Across Modalities** | Grouped bar chart — three severity levels side-by-side per source |
| **Modality Spotlight tabs** | One tab per source: per-source KPIs, top-events bar, severity-mix donut, and an expandable table of the highest-severity records |
| **All Incident Records table** | Full scrollable table with colour-coded Severity column, all 9 unified fields |
| **Download CSV button** | Exports the currently filtered rows as a CSV file |

---

## Refreshing the data

If you re-run any of the five extraction pipelines and want the dashboard to
reflect the updated outputs, regenerate the master dataset first:

```bash
# from the repo root
python integration/build_dataset.py
```

Then relaunch (or just reload the browser tab if Streamlit is already running —
it detects file changes automatically):

```bash
streamlit run integration/dashboard.py
```

---

## Unified dataset schema

The dashboard reads `master_incident_dataset.csv`, which has this schema:

| Column | Description |
|--------|-------------|
| `Incident_ID` | Modality-prefixed unique ID (`AUD-001`, `DOC-001`, `IMG-001`, `VID-001`, `TXT-001`) |
| `Source` | Modality label — Audio / PDF / Image / Video / Text |
| `Source_Record_ID` | The modality's own record ID (Call_ID, Report_ID, etc.) |
| `Event` | What happened |
| `Location` | Where — best-effort, `N/A` when unavailable |
| `Time` | When — best-effort, `N/A` when unavailable |
| `Severity_Score` | 0–10 numeric score derived from each modality's confidence signal |
| `Severity_Level` | Low (0–3) / Medium (3–7) / High (7–10) |
| `Details` | Modality-specific context (transcript excerpt, officer name, detected objects, etc.) |

**Current dataset stats:** 1,031 total rows — Audio 707, Video 164, Text 115, PDF 42, Image 3.
Severity split — Medium 733, High 174, Low 124.
