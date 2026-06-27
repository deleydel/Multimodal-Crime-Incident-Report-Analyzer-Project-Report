"""
Multimodal Crime / Incident Report Analyzer
Integration & Data Orchestration Module
===========================================

This is the central data-engineering layer of the project. It takes the five
independent structured outputs (one CSV per modality) and programmatically
merges them into a single unified incident dataset.

Pipeline (matches the project specification):

  Step 1  Assign a unique modality-prefixed Incident_ID per row
          (AUD- / DOC- / IMG- / VID- / TXT-), keeping each modality's own
          source ID column as `Source_Record_ID`.

  Step 2  Concatenate (UNION) all five DataFrames into one common schema with
          pandas.concat — one row per source record. The modalities share no
          incidents, so this is a vertical union, NOT a key join.

  Step 3  Handle missing values: any field a modality does not provide is
          filled with "N/A".

  Step 4  Derive a final severity classification (Low / Medium / High) from
          each row's own confidence / urgency signal.

Severity scale (documented here, in the demo, and in the report)
----------------------------------------------------------------
    Severity_Score = confidence_signal * 10        (range 0 - 10)

        0  <= score <  3   ->  Low
        3  <= score <  7   ->  Medium
        7  <= score <= 10  ->  High

Where `confidence_signal` (0.0 - 1.0) is taken from each modality's own
extracted signal:

    Audio  ->  Urgency_Score          (directly)
    Image  ->  Confidence_Score       (directly)
    Video  ->  Confidence             (directly)
    PDF    ->  a document is a record, not an active incident, so it has no
               numeric urgency signal. We rank by document-type priority:
               event-bearing docs (Memorandum / Letter / *Report) -> 0.50
               (Medium); routine administrative paperwork (cover letters,
               certificates, policies, invoices, training docs) -> 0.30 (Low).
    Text   ->  derived from Sentiment {Negative: 0.70, Neutral: 0.40,
               Positive: 0.20}, boosted by +0.15 when the Topic is a violent /
               high-priority crime category (capped at 1.0).

Unified schema (the minimum schema plus a few helpful extra columns)
--------------------------------------------------------------------
    Incident_ID        modality-prefixed unique id  (AUD-001, DOC-001, ...)
    Source             modality label               (Audio / PDF / Image / ...)
    Source_Record_ID   the modality's own id        (Call_ID, Report_ID, ...)
    Event              what happened
    Location           where (best-effort, may be N/A)
    Time               when (best-effort, may be N/A)
    Severity_Score     0 - 10 numeric score
    Severity_Level     Low / Medium / High
    Details            extra modality-specific context

Usage:
    python integration/build_dataset.py
    python integration/build_dataset.py --output some/other/path.csv
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths — everything is resolved relative to the repository root so the script
# runs the same no matter the current working directory.
# ---------------------------------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parent.parent

MODALITY_CSVS = {
    "Audio": REPO_ROOT / "audio" / "audio_extracted_report.csv",
    "PDF":   REPO_ROOT / "pdf"   / "output.csv",
    "Image": REPO_ROOT / "images" / "image_report.csv",
    "Video": REPO_ROOT / "video" / "output" / "video_events.csv",
    "Text":  REPO_ROOT / "text"  / "crime_text_report.csv",
}

DEFAULT_OUTPUT = REPO_ROOT / "integration" / "master_incident_dataset.csv"

NA = "N/A"

# ---------------------------------------------------------------------------
# Severity configuration (single source of truth)
# ---------------------------------------------------------------------------

LOW_MAX = 3.0     # score < 3            -> Low
MED_MAX = 7.0     # 3 <= score < 7       -> Medium ; score >= 7 -> High

# PDF documents are records, not active incidents, so they carry no numeric
# urgency signal. We rank by the *document type's* priority instead: a memo or
# letter may flag something actionable (Medium), while certificates, invoices,
# policies and training paperwork are routine administrative records (Low).
PDF_ELEVATED_TYPES = {
    "memorandum", "incident report", "crime report", "arrest report", "letter",
}
PDF_ELEVATED_CONFIDENCE = 0.50     # -> 5.0 -> Medium
PDF_ROUTINE_CONFIDENCE = 0.25      # -> 2.5 -> Low  (other/unrecognized doc types)

TEXT_SENTIMENT_CONFIDENCE = {
    "Negative": 0.70,
    "Neutral":  0.40,
    "Positive": 0.20,
}
TEXT_VIOLENT_TOPICS = {"Assault / Violence", "Theft / Robbery"}
TEXT_VIOLENT_BOOST = 0.15

UNIFIED_COLUMNS = [
    "Incident_ID", "Source", "Source_Record_ID", "Event", "Location",
    "Time", "Severity_Score", "Severity_Level", "Details",
]


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _clean(value) -> str:
    """Standardise a single cell to a trimmed string, or N/A when empty."""
    if value is None:
        return NA
    if isinstance(value, float) and pd.isna(value):
        return NA
    text = str(value).strip()
    if text == "" or text.lower() in {"nan", "none", "unknown", "unknown location"}:
        return NA
    return re.sub(r"\s+", " ", text)


def _confidence_to_severity(confidence: float) -> tuple[float, str]:
    """Map a 0.0-1.0 confidence/urgency signal to (score 0-10, Low/Medium/High)."""
    confidence = max(0.0, min(1.0, float(confidence)))
    score = round(confidence * 10, 2)
    if score < LOW_MAX:
        level = "Low"
    elif score < MED_MAX:
        level = "Medium"
    else:
        level = "High"
    return score, level


def _assign_ids(prefix: str, n: int) -> list[str]:
    """AUD-001, AUD-002, ... — modality-prefixed, zero-padded, unique per row."""
    width = max(3, len(str(n)))
    return [f"{prefix}-{i:0{width}d}" for i in range(1, n + 1)]


def _read(source: str) -> pd.DataFrame:
    path = MODALITY_CSVS[source]
    if not path.exists():
        raise FileNotFoundError(
            f"[{source}] expected modality output not found: {path}\n"
            f"Run that modality's extraction script first."
        )
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    print(f"  [{source:5}] {len(df):4d} rows  <-  {path.relative_to(REPO_ROOT)}")
    return df


def _text_locations(entities: str) -> str:
    """Pull the '(Location)'-tagged entities out of the text modality's
    free-form Entities column, e.g. 'Chicago (Location); Monday (Date)'."""
    if not entities or entities == NA:
        return NA
    locs = re.findall(r"([A-Za-z][\w'’.\- ]+?)\s*\(Location\)", entities)
    locs = [l.strip() for l in locs if l.strip()]
    return "; ".join(dict.fromkeys(locs)) if locs else NA


def _truncate(text: str, limit: int = 160) -> str:
    if text == NA or len(text) <= limit:
        return text
    return text[:limit].rsplit(" ", 1)[0].rstrip(" ,.;:") + " ..."


# ---------------------------------------------------------------------------
# Per-modality normalisers: each returns rows in the UNIFIED schema.
# ---------------------------------------------------------------------------

def normalize_audio(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["Incident_ID"] = _assign_ids("AUD", len(df))
    out["Source"] = "Audio"
    out["Source_Record_ID"] = df["Call_ID"].map(_clean)
    out["Event"] = df["Extracted_Event"].map(_clean)
    out["Location"] = df["Location"].map(_clean)
    out["Time"] = NA                              # 911 calls carry no timestamp
    scores = df["Urgency_Score"].astype(float).map(_confidence_to_severity)
    out["Severity_Score"] = [s for s, _ in scores]
    out["Severity_Level"] = [l for _, l in scores]
    out["Details"] = [
        f"Sentiment: {_clean(sent)} | Transcript: {_truncate(_clean(tr))}"
        for sent, tr in zip(df["Sentiment"], df["Transcript"])
    ]
    return out


def _pdf_confidence(incident_type: str) -> float:
    """Document-type priority: event-bearing docs (memo/letter/report) rank
    Medium; routine administrative paperwork ranks Low."""
    if _clean(incident_type).lower() in PDF_ELEVATED_TYPES:
        return PDF_ELEVATED_CONFIDENCE
    return PDF_ROUTINE_CONFIDENCE


def normalize_pdf(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["Incident_ID"] = _assign_ids("DOC", len(df))
    out["Source"] = "PDF"
    out["Source_Record_ID"] = df["Report_ID"].map(_clean)
    out["Event"] = df["Incident_Type"].map(_clean)
    out["Location"] = df["Location"].map(_clean)
    out["Time"] = df["Date"].map(_clean)          # best-effort date as the time field
    scores = [_confidence_to_severity(_pdf_confidence(t)) for t in df["Incident_Type"]]
    out["Severity_Score"] = [s for s, _ in scores]
    out["Severity_Level"] = [l for _, l in scores]
    out["Details"] = [
        f"Officer: {_clean(off)} | Summary: {_truncate(_clean(summ))}"
        for off, summ in zip(df["Officer"], df["Summary"])
    ]
    return out


def normalize_image(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["Incident_ID"] = _assign_ids("IMG", len(df))
    out["Source"] = "Image"
    out["Source_Record_ID"] = df["Image_ID"].map(_clean)
    out["Event"] = df["Scene_Type"].map(_clean)
    out["Location"] = NA                           # no geolocation from a still image
    out["Time"] = NA
    scores = df["Confidence_Score"].astype(float).map(_confidence_to_severity)
    out["Severity_Score"] = [s for s, _ in scores]
    out["Severity_Level"] = [l for _, l in scores]
    out["Details"] = [
        f"Objects: {_clean(obj)} | OCR: {_clean(txt)}"
        for obj, txt in zip(df["Objects_Detected"], df["Text_Extracted"])
    ]
    return out


def normalize_video(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["Incident_ID"] = _assign_ids("VID", len(df))
    out["Source"] = "Video"
    out["Source_Record_ID"] = df["Frame_ID"].map(_clean)
    out["Event"] = df["Event_Detected"].map(_clean)
    out["Location"] = NA                           # surveillance clip, no geolocation
    out["Time"] = df["Timestamp"].map(_clean)      # HH:MM:SS within the clip
    scores = df["Confidence"].astype(float).map(_confidence_to_severity)
    out["Severity_Score"] = [s for s, _ in scores]
    out["Severity_Level"] = [l for _, l in scores]
    out["Details"] = [f"Objects: {_clean(obj)}" for obj in df["Objects"]]
    return out


def _text_confidence(sentiment: str, topic: str) -> float:
    base = TEXT_SENTIMENT_CONFIDENCE.get(_clean(sentiment), 0.40)
    if _clean(topic) in TEXT_VIOLENT_TOPICS:
        base = min(1.0, base + TEXT_VIOLENT_BOOST)
    return base


def normalize_text(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame()
    out["Incident_ID"] = _assign_ids("TXT", len(df))
    out["Source"] = "Text"
    out["Source_Record_ID"] = df["Text_ID"].map(_clean)
    out["Event"] = df["Topic"].map(_clean)
    out["Location"] = df["Entities"].map(_clean).map(_text_locations)
    out["Time"] = NA
    scores = [
        _confidence_to_severity(_text_confidence(sent, topic))
        for sent, topic in zip(df["Sentiment"], df["Topic"])
    ]
    out["Severity_Score"] = [s for s, _ in scores]
    out["Severity_Level"] = [l for _, l in scores]
    out["Details"] = [
        f"Handle: {_clean(src)} | Sentiment: {_clean(sent)} | Text: {_truncate(_clean(raw))}"
        for src, sent, raw in zip(df["Source"], df["Sentiment"], df["Raw_Text"])
    ]
    return out


NORMALIZERS = {
    "Audio": normalize_audio,
    "PDF":   normalize_pdf,
    "Image": normalize_image,
    "Video": normalize_video,
    "Text":  normalize_text,
}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def build_dataset() -> pd.DataFrame:
    print("[1/4] Reading five modality outputs and assigning prefixed IDs ...")
    frames = []
    for source, normalize in NORMALIZERS.items():
        frames.append(normalize(_read(source)))

    print("[2/4] UNION via pandas.concat (one row per source record) ...")
    master = pd.concat(frames, ignore_index=True)
    master = master[UNIFIED_COLUMNS]

    print("[3/4] Filling missing values with N/A ...")
    master = master.replace({"": NA, "nan": NA, "None": NA})
    master = master.fillna(NA)

    print("[4/4] Severity already derived per row (Severity_Score / Severity_Level).")
    return master


def main() -> None:
    ap = argparse.ArgumentParser(description="Merge the five modality CSVs into "
                                             "one unified incident dataset.")
    ap.add_argument("--output", default=str(DEFAULT_OUTPUT),
                    help="path for the merged master CSV")
    args = ap.parse_args()

    master = build_dataset()

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    master.to_csv(out_path, index=False)

    print("\n" + "=" * 60)
    print(f"Unified dataset written: {out_path}")
    print(f"Total incidents (rows): {len(master)}")
    print("\nRows per source:")
    print(master["Source"].value_counts().to_string())
    print("\nSeverity distribution:")
    print(master["Severity_Level"].value_counts().to_string())
    print("=" * 60)


if __name__ == "__main__":
    main()
