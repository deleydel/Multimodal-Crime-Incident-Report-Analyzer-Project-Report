"""
Police Report PDF Analyst — v2
Extracts structured records from a bundle PDF that contains many distinct
official sub-documents (cover letters, SOPs, lesson plans, invoices, forms, …).

Pipeline:
  Phase 1  PDF -> text   (text-first via fitz; robust OCR fallback with OpenCV
                          preprocessing, dual-PSM confidence selection;
                          pdfplumber for tables; photo/non-text detection)
  Phase 2  Segmentation  (fuzzy signature library -> one record per sub-document)
  Phase 3  Extraction    (per-field candidate scoring: regex + spaCy NER +
                          positional heuristics; dates normalised to ISO)
  Phase 4  Output        (output.csv + segments.json + ocr_raw.txt)

OCR is cached to pages_cache.json so Phase 2/3 can be re-tuned without re-OCR.
Bump OCR_VERSION to force a fresh OCR pass.
"""

import io
import re
import csv
import json
import os
import argparse
from datetime import datetime

import fitz
import pdfplumber
import pytesseract
from pytesseract import Output
import spacy
import numpy as np
import cv2
from PIL import Image
from rapidfuzz import fuzz
from dateutil import parser as dateparser
from spellchecker import SpellChecker

# Paths are relative to this script; the PDF defaults to ~/Downloads and can be
# overridden with --pdf. (Set in main() from CLI args.)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PDF_PATH = os.path.expanduser("~/Downloads/policereport.pdf")
OUTPUT_CSV = os.path.join(SCRIPT_DIR, "output.csv")
RAW_TEXT_PATH = os.path.join(SCRIPT_DIR, "ocr_raw.txt")
SEGMENTS_JSON = os.path.join(SCRIPT_DIR, "segments.json")
CACHE_PATH = os.path.join(SCRIPT_DIR, "pages_cache.json")

OCR_VERSION = "v4-otsu-nodeskew-dualpsm-rotate-400dpi"
# When True, also write inspection artifacts (ocr_raw.txt, segments.json).
# Default off so a normal run produces only output.csv.
WRITE_DEBUG = False
DPI = 400
TEXT_PAGE_MIN_CHARS = 50          # fitz text >= this ⇒ treat page as text-based
NONTEXT_CONF_MAX = 50             # mean OCR conf below this ...
NONTEXT_ALNUM_MAX = 400          # ... and few alnum chars ⇒ photo/non-text page

FIELDNAMES = [
    "Report_ID", "Incident_Type", "Date", "Location",
    "Officer", "Summary", "Suspect_Description", "Outcome",
]


# ════════════════════════════════════════════════════════════════════════════
# Phase 1 — PDF → text
# ════════════════════════════════════════════════════════════════════════════

def _preprocess(pil_img: Image.Image) -> Image.Image:
    """Grayscale → denoise → Otsu threshold (OpenCV).

    Note: an automatic deskew step was tried and removed — cv2.minAreaRect on
    raw text pixels produced unreliable angles and rotated upright pages up to
    90°, corrupting OCR. These scans are near-upright, so deskew is unnecessary.
    """
    gray = np.array(pil_img.convert("L"))
    gray = cv2.fastNlMeansDenoising(gray, h=10)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(th)


def _reconstruct_text(data: dict) -> str:
    """Rebuild line-structured text from image_to_data output."""
    lines = {}
    for i, word in enumerate(data["text"]):
        if not word.strip():
            continue
        if int(data["conf"][i]) < 0:
            continue
        key = (data["block_num"][i], data["par_num"][i], data["line_num"][i])
        lines.setdefault(key, []).append(word)
    return "\n".join(" ".join(words) for _, words in sorted(lines.items()))


def _ocr(pil_img: Image.Image):
    """Dual-PSM OCR; keep the variant with higher mean word confidence."""
    best_text, best_conf = "", -1.0
    for psm in (3, 6):
        config = f"--oem 1 --psm {psm}"
        data = pytesseract.image_to_data(pil_img, config=config,
                                         output_type=Output.DICT)
        confs = [int(c) for c in data["conf"] if int(c) >= 0]
        mean_conf = sum(confs) / len(confs) if confs else 0.0
        if mean_conf > best_conf:
            best_conf, best_text = mean_conf, _reconstruct_text(data)
    return best_text, best_conf


ROTATE_RETRY_CONF = 60     # below this, the page may be rotated — try orientations


def _ocr_best(pil_img: Image.Image):
    """OCR upright; if confidence is low the page may be rotated (e.g. the
    upside-down certificates on pp 58-61), so try 180/90/270 and keep the best."""
    text, conf = _ocr(pil_img)
    rotation = 0
    if conf < ROTATE_RETRY_CONF:
        for rot in (180, 90, 270):
            t, c = _ocr(pil_img.rotate(rot, expand=True))
            if c > conf:
                text, conf, rotation = t, c, rot
    return text, conf, rotation


def _clean_ocr(text: str) -> str:
    """Drop garbage lines (<50% alphanumeric) and collapse blank runs."""
    keep = []
    for line in text.splitlines():
        s = line.strip()
        if not s:
            keep.append("")
            continue
        ratio = sum(c.isalnum() or c.isspace() for c in s) / len(s)
        if ratio >= 0.5:
            keep.append(line)
    return re.sub(r"\n{3,}", "\n\n", "\n".join(keep)).strip()


def _table_text(pdf_path: str, page_index: int) -> str:
    """Extract tables from a page via pdfplumber, rendered as pipe-joined rows."""
    try:
        with pdfplumber.open(pdf_path) as pdf:
            page = pdf.pages[page_index]
            tables = page.extract_tables()
    except Exception:
        return ""
    out = []
    for tbl in tables or []:
        for row in tbl:
            cells = [c.strip() for c in row if c and c.strip()]
            if cells:
                out.append(" | ".join(cells))
    return "\n".join(out)


def pdf_to_pages(pdf_path: str) -> list[dict]:
    """Return per-page dicts; cache to disk keyed by OCR_VERSION."""
    if os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            cache = json.load(f)
        if cache.get("version") == OCR_VERSION:
            pages = cache["pages"]
            # Recompute non_text from cached conf/text so threshold tuning
            # doesn't require a full re-OCR pass.
            for p in pages:
                if p["method"] == "ocr":
                    alnum = sum(c.isalnum() for c in p["text"])
                    p["non_text"] = (p["conf"] < NONTEXT_CONF_MAX
                                     and alnum < NONTEXT_ALNUM_MAX)
            print(f"  Loaded cached OCR ({len(pages)} pages)")
            return pages

    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        native = page.get_text().strip()
        if len(native) >= TEXT_PAGE_MIN_CHARS:
            text = native
            tbl = _table_text(pdf_path, i)
            if tbl:
                text += "\n[TABLE]\n" + tbl
            rec = {"page": i + 1, "method": "text", "conf": 100.0,
                   "non_text": False, "text": text}
        else:
            mat = fitz.Matrix(DPI / 72, DPI / 72)
            pix = page.get_pixmap(matrix=mat)
            img = Image.open(io.BytesIO(pix.tobytes("png")))
            img = _preprocess(img)
            raw, conf, rotation = _ocr_best(img)
            text = _clean_ocr(raw)
            alnum = sum(c.isalnum() for c in text)
            non_text = conf < NONTEXT_CONF_MAX and alnum < NONTEXT_ALNUM_MAX
            rec = {"page": i + 1, "method": "ocr", "conf": round(conf, 1),
                   "rotation": rotation, "non_text": non_text, "text": text}
        pages.append(rec)
        flag = " [NON-TEXT]" if rec["non_text"] else ""
        print(f"  Page {i+1:3d}/{len(doc)} [{rec['method']}] "
              f"conf={rec['conf']:5.1f} chars={len(rec['text']):4d}{flag}")

    with open(CACHE_PATH, "w") as f:
        json.dump({"version": OCR_VERSION, "pages": pages}, f)
    return pages


# ════════════════════════════════════════════════════════════════════════════
# Phase 2 — Sub-document segmentation
# ════════════════════════════════════════════════════════════════════════════

# Each signature: (doc_type, [(pattern, weight, kind)])
#   kind "regex" → re.search (IGNORECASE);  kind "fuzzy" → rapidfuzz partial_ratio
# Structured doc types detected by header keyword signatures.
# Cover Letter and Training Certificate are handled by dedicated detectors below
# (they need positional / letterhead logic that flat keyword scoring can't do).
SIGNATURES = [
    ("Memorandum", [
        ("memorandum", 3, "fuzzy"),
        (r"^\s*to\s*:", 1, "regex"),
        (r"^\s*from\s*:", 1, "regex"),
    ]),
    ("Letter", [
        (r"^\s*dear\s+\w", 4, "regex"),
    ]),
    ("Policies and Procedures", [
        ("policies and procedures", 5, "fuzzy"),
    ]),
    ("Department Policy", [
        ("date of origin", 3, "fuzzy"),
        (r"effective\s*:", 2, "regex"),
        (r"\bi\.\s*purpose", 1, "regex"),
        ("this policy is for internal use only", 4, "fuzzy"),
    ]),
    ("Standard Operating Procedure", [
        ("standard operating procedure", 5, "fuzzy"),
    ]),
    ("Divisional Operating Procedure", [
        ("divisional operating procedure", 6, "fuzzy"),
    ]),
    ("Training Lesson Plan", [
        (r"course\s*:", 2, "regex"),
        (r"lesson\s+title\s*:", 3, "regex"),
        (r"duration\s*:", 1, "regex"),
        (r"training\s+level\s*:", 1, "regex"),
        (r"prepared\s+by\s*:", 1, "regex"),
        (r"method\s+of\s+presentation", 1, "regex"),
        ("continuing education course", 2, "fuzzy"),
        ("lesson plan", 3, "fuzzy"),
    ]),
    ("Course Cover Sheet", [
        ("cover sheet", 5, "fuzzy"),
    ]),
    ("Course Summary Sheet", [
        ("course summary sheet", 6, "fuzzy"),
    ]),
    ("Course Outline", [
        ("course outline", 6, "fuzzy"),
    ]),
    ("Road Course Evaluation", [
        ("road course", 4, "fuzzy"),
        ("evaluated driving", 3, "fuzzy"),
    ]),
    # NOTE: "Reference Materials" (TRAINEE/INSTRUCTOR REFERENCE) is intentionally
    # NOT an opener — it is a section *inside* a lesson plan (e.g. pp 18, 33, 68),
    # so it should attach to the preceding Training Lesson Plan as a continuation.
    ("Invoice", [
        (r"invoice\s*#", 5, "regex"),
    ]),
    ("Aircraft Request Form", [
        ("aircraft request", 6, "fuzzy"),
    ]),
]

OPENER_THRESHOLD = 3       # min score for a page to open a new segment
HEADER_CHARS = 900         # how much of the page top to inspect for openers

# Phrases identifying a letter addressed to the federal 1033 / LESO program
_ADDRESSEES = ("1033 program", "law enforcement support office",
               "arkansas leso", "program manager")
_DATE_RE = re.compile(
    r"(?:january|february|march|april|may|june|july|august|september|october"
    r"|november|december)\s+\d{1,2},?\s+\d{4}|\d{1,2}/\d{1,2}/\d{2,4}",
    re.IGNORECASE)


def _has_letterhead(text: str) -> bool:
    """True if one of the first lines is a short dept-name heading (a real
    letterhead), not a body sentence that merely mentions a department."""
    for line in text[:400].splitlines()[:4]:
        s = line.strip()
        # A letterhead is a short heading: starts uppercase, no trailing
        # punctuation. Excludes list items ("13.") and wrapped body fragments
        # ("obtained for the police department.").
        if not s or len(s) > 40 or not s[0].isupper() or s.endswith((".", ",")):
            continue
        low = s.lower()
        if re.search(r"\b(police|sheriff)\b", low):
            return True
        if "county" in low and re.search(r"\b(police|sheriff|office)\b", low):
            return True
        if "office of the sheriff" in low:
            return True
    return False


def _cover_letter_score(text: str) -> int:
    """Detect cover letters, including ones that omit 'To Whom It May Concern'
    but carry a dept letterhead + a 1033/LESO addressee + a date.

    The weak path (no 'To Whom'/From:/Ref:) requires a genuine letterhead
    heading so SOP/policy body pages that merely mention 'LESO' or 'police
    department' in prose do not get mis-split into letters.
    """
    head = text[:500]
    low = head.lower()
    top = text[:250].lower()              # addressee/date must be near the top
    score = 0
    strong = fuzz.partial_ratio("to whom it may concern", low) >= 88
    if strong:
        score += 4
    letterhead = _has_letterhead(text)
    if letterhead:
        score += 2
    addressee = (any(fuzz.partial_ratio(a, top) >= 88 for a in _ADDRESSEES)
                 or re.search(r"\bjames\s+ray\b", top) or re.search(r"\bleso\b", top))
    has_from = bool(re.search(r"\bfrom\s*:", head, re.IGNORECASE))
    has_ref = bool(re.search(r"\bref\s*:", head, re.IGNORECASE))
    if addressee:
        score += 2
    if _DATE_RE.search(top):
        score += 1
    if has_from:
        score += 1
    if has_ref:
        score += 1
    # Weak path needs a real letterhead to count at all
    if not (strong or has_from or has_ref) and not letterhead:
        return 0
    return score


def _certificate_score(text: str) -> int:
    """A real certificate has the cert language at the very top of a short page;
    a cover letter merely mentions 'Joint Program Office' deep in its body."""
    if (fuzz.partial_ratio("joint program office", text[:150].lower()) >= 88
            and fuzz.partial_ratio("successful completion",
                                   text[:300].lower()) >= 88):
        return 6
    return 0


def _signature_scores(zone: str) -> dict:
    """Score only the structured keyword signatures against a text zone."""
    low = zone.lower()
    scores = {}
    for doc_type, patterns in SIGNATURES:
        total = 0
        for pat, weight, kind in patterns:
            if kind == "regex":
                if re.search(pat, zone, re.IGNORECASE | re.MULTILINE):
                    total += weight
            else:
                if fuzz.partial_ratio(pat, low) >= 85:
                    total += weight
        if total:
            scores[doc_type] = total
    return scores


def _score_doc_types(text: str) -> dict:
    scores = _signature_scores(text[:HEADER_CHARS])
    cl = _cover_letter_score(text)
    if cl:
        scores["Cover Letter"] = cl
    cert = _certificate_score(text)
    if cert:
        scores["Training Certificate"] = cert
    return scores


BOTTOM_OPENER_MIN = 5      # strong anchor needed for a bottom-of-page opener
# Types that are a single multi-page document: a repeated header on the next
# page is a continuation, not a new record (e.g. the 2-page LEA aircraft form).
# (Cover letters / certificates are NOT here — consecutive ones are distinct.)
SAME_INSTANCE_TYPES = {"Aircraft Request Form"}


def segment_pages(pages: list[dict]) -> list[dict]:
    """Group pages into per-sub-document segments using the signature library."""
    segments = []
    current = None

    def flush():
        nonlocal current
        if current:
            current["page_end"] = current["pages"][-1]
            current["text"] = "\n\n".join(current["_chunks"]).strip()
            del current["_chunks"]
            segments.append(current)
            current = None

    pending = None        # a doc opener seen at the bottom of the previous page

    for pg in pages:
        # Photo / non-text pages: their own grouped Non-text segment
        if pg["non_text"]:
            pending = None
            if current is None or current["doc_type"] != "Photo/Non-text":
                flush()
                current = {"doc_type": "Photo/Non-text",
                           "page_start": pg["page"], "pages": [],
                           "score": 0, "_chunks": []}
            current["pages"].append(pg["page"])
            current["_chunks"].append(pg["text"])
            continue

        scores = _score_doc_types(pg["text"])
        best_type, best_score = (max(scores.items(), key=lambda kv: kv[1])
                                 if scores else (None, 0))

        opens_new = best_score >= OPENER_THRESHOLD
        # Don't split a multi-page single form into two records
        if (opens_new and current is not None
                and best_type == current["doc_type"]
                and best_type in SAME_INSTANCE_TYPES):
            opens_new = False

        if opens_new:
            flush()
            current = {"doc_type": best_type, "page_start": pg["page"],
                       "pages": [], "score": best_score, "_chunks": []}
        elif pending and current is not None:
            # Previous page's bottom started a new doc that this page continues
            # (handles an opener that appears near the foot of a page).
            flush()
            current = {"doc_type": pending, "page_start": pg["page"],
                       "pages": [], "score": BOTTOM_OPENER_MIN, "_chunks": []}
        elif current is None:
            current = {"doc_type": "Unclassified", "page_start": pg["page"],
                       "pages": [], "score": 0, "_chunks": []}

        current["pages"].append(pg["page"])
        current["_chunks"].append(pg["text"])

        # Look at the foot of this page: a strong, different-typed opener here
        # means the next page belongs to that new document.
        bsig = _signature_scores(pg["text"][-400:])
        bt, bs = (max(bsig.items(), key=lambda kv: kv[1]) if bsig else (None, 0))
        pending = bt if (bs >= BOTTOM_OPENER_MIN
                         and bt != current["doc_type"]) else None

    flush()

    # Assign Report_IDs only to real records (exclude Photo/Non-text)
    rid = 0
    for seg in segments:
        if seg["doc_type"] == "Photo/Non-text":
            seg["report_id"] = None
        else:
            rid += 1
            seg["report_id"] = f"RPT_{rid:03d}"
    return segments


# ════════════════════════════════════════════════════════════════════════════
# Phase 3 — Field extraction (candidate scoring)
# ════════════════════════════════════════════════════════════════════════════

_LE_KEYWORDS = ("police", "sheriff", "county", "constable", "marshal")
_ORG_KEYWORDS = ("department", "office", "bureau", "agency", "commission",
                 "division", "authority", "service")
_NAME_STOP = {
    "department", "police", "sheriff", "county", "office", "city", "division",
    "unit", "chief", "deputy", "captain", "lieutenant", "sergeant", "corporal",
    "major", "colonel", "director", "coordinator", "supervisor", "swat", "team",
    "bureau", "agency", "instructor", "operators", "operator",
    "revisions", "revision", "subject", "subj", "date", "duration", "course",
}
_TITLE_RE = re.compile(
    r"^(?:Mr|Mrs|Ms|Dr|Lt|Cpl|Sgt|Capt|Sheriff|Deputy|Officer|Chief|Major)\b",
    re.IGNORECASE,
)


# Street / address words that disqualify a candidate from being a person name
_ADDRESS_RE = re.compile(
    r"\b(ave|avenue|st|street|ln|lane|rd|road|blvd|dr|drive|hwy|highway|box|"
    r"markham|huntwick|dixieland|woodrow|roya)\b", re.IGNORECASE)
# Leading rank/courtesy titles to strip off the front of a name
_TITLE_TOKENS = {
    "sheriff", "sherif", "deputy", "officer", "ofc", "cpl", "corporal", "sgt",
    "sergeant", "lt", "lieutenant", "capt", "captain", "major", "colonel",
    "col", "chief", "mr", "mrs", "ms", "dr", "det", "detective", "asst", "sir",
}
# AR place names that spaCy sometimes mistags as PERSON (e.g. "Van Buren")
_CITY_NAMES = {
    "van buren", "little rock", "bentonville", "bryant", "cabot", "fort smith",
    "hot springs", "jacksonville", "lonoke", "rogers", "el dorado", "luxora",
    "texarkana", "batesville", "arkansas", "pulaski",
}
# Obvious non-name tokens from OCR / NER noise
_NONNAME = {"shut", "down", "power", "kill", "none", "whom", "concern", "system"}


def _plausible_name(name: str) -> bool:
    """Reject OCR-garbled 'names' (e.g. 'Fa ee eee', 'se Bae CY', 'DQ Maf')."""
    toks = name.split()
    if not (2 <= len(toks) <= 3):
        return False
    if not any(len(t.strip(".")) >= 3 for t in toks):
        return False
    for t in toks:
        t2 = t.strip(".")
        if len(t2) < 2 or not t2.isalpha():
            return False
        if not t2[0].isupper():                 # name tokens start capitalized
            return False
        if t2.isupper() and len(t2) == 2:        # 2-letter all-caps = initials/noise
            return False
        if not re.search(r"[aeiouAEIOU]", t2):   # every name token has a vowel
            return False
        if re.search(r"(.)\1\1", t2):            # 3+ repeated chars = OCR noise
            return False
    return True


def _clean_name(raw: str) -> str:
    if _ADDRESS_RE.search(raw):
        return ""                          # looks like a street address, not a name
    raw = re.sub(r"[^a-zA-Z\s\.\-]", "", raw)
    toks = [t for t in raw.split() if len(t.strip(".")) > 1]
    while toks and toks[0].lower().strip(".") in _TITLE_TOKENS:
        toks.pop(0)                        # strip leading rank/courtesy titles
    out = []
    for t in toks:
        if t.lower().strip(".") in _NAME_STOP:
            break                          # stop at an org word (Department, …)
        out.append(t)
    name = " ".join(out[:2]).strip()       # First Last (middle initials dropped)
    low = name.lower()
    if low in _CITY_NAMES or any(w in _NONNAME for w in low.split()):
        return ""
    return name if _plausible_name(name) else ""


def _iso_date(raw: str) -> str | None:
    """Parse a date to ISO. Rejects partial strings (e.g. year-only) that
    dateutil would otherwise complete with TODAY's month/day. Parsing with two
    different defaults exposes any field that came from the default, not the text.
    """
    try:
        d1 = dateparser.parse(raw, fuzzy=True, default=datetime(1900, 1, 1))
        d2 = dateparser.parse(raw, fuzzy=True, default=datetime(2002, 6, 15))
    except Exception:
        return None
    if not d1 or not d2:
        return None
    # year/month/day must come from the text (identical under both defaults)
    if (d1.year, d1.month, d1.day) != (d2.year, d2.month, d2.day):
        return None
    if not (1990 <= d1.year <= 2026):
        return None
    return d1.strftime("%Y-%m-%d")


def extract_date(text: str) -> str:
    # Prefer labeled date fields inside the segment
    for label in (r"date\s+of\s+origin", r"effective\s+date", r"\beffective",
                  r"\bdate"):
        m = re.search(label + r"\s*[:\-]?\s*([A-Za-z0-9,\/\-\. ]{6,30})",
                      text, re.IGNORECASE)
        if m:
            iso = _iso_date(m.group(1).strip())
            if iso:
                return iso
    # Any explicit date pattern
    for pat in (
        r"(?:January|February|March|April|May|June|July|August|September"
        r"|October|November|December)\s+\d{1,2},?\s+\d{4}",
        r"\d{1,2}\s+(?:January|February|March|April|May|June|July|August"
        r"|September|October|November|December)\s+\d{4}",
        r"\d{1,2}[\/\-]\d{1,2}[\/\-]\d{2,4}",
    ):
        m = re.search(pat, text, re.IGNORECASE)
        if m:
            iso = _iso_date(m.group(0))
            if iso:
                return iso
    # Lenient: OCR can space out digits ("on 1 8 July 201 4" → 18 July 2014)
    m = re.search(
        r"\b(\d\s?\d?)\s+(January|February|March|April|May|June|July|August"
        r"|September|October|November|December)\s+(\d\s?\d\s?\d\s?\d)\b",
        text, re.IGNORECASE)
    if m:
        iso = _iso_date(f"{m.group(1).replace(' ', '')} {m.group(2)} "
                        f"{m.group(3).replace(' ', '')}")
        if iso:
            return iso
    return "N/A"


_DEPT_RE = re.compile(
    r"([A-Z][A-Za-z'’\.]+(?:\s+[A-Za-z'’\.]+){0,3}\s+"
    r"(?:Police\s+Department|Sheriff['’]?s?\s+(?:Office|Department)))",
    re.IGNORECASE)
# Federal program / recipient orgs — never the sender agency for Location
_LOC_BLOCKLIST = (
    "law enforcement support office", "leso", "1033 program",
    "dla disposition", "department of career education", "program manager",
)


def _is_blocked_loc(s: str) -> bool:
    low = s.lower()
    return any(b in low for b in _LOC_BLOCKLIST)


# A candidate starting with a rank/courtesy title is a recipient block, not a sender
_LEAD_TITLE_RE = re.compile(
    r"^(?:corporal|cpl|officer|ofc|sergeant|sgt|lt|lieutenant|captain|capt"
    r"|major|deputy|mr|mrs|ms|dr)\b", re.IGNORECASE)


def _letterhead_location(text: str) -> str | None:
    """Pull the sender's department name from the letterhead / opening lines."""
    head = "\n".join(text.splitlines()[:12])
    # "County of X ... Sheriff" → "X County Sheriff's Department"
    m = re.search(r"County of ([A-Z][a-z]+)", head)
    if m and re.search(r"sheriff", head, re.IGNORECASE):
        return f"{m.group(1)} County Sheriff's Department"
    # First real department name that isn't a recipient block or program org
    for mt in _DEPT_RE.finditer(head):
        cand = re.sub(r"^(?:[a-z]+\s+)+", "", mt.group(1).strip()).strip()
        if not cand or _LEAD_TITLE_RE.match(cand):
            continue
        if len(cand.split()) >= 2 and not _is_blocked_loc(cand):
            return cand
    return None


def _normalize_location(s: str) -> str:
    if not s or s == "N/A":
        return "N/A"
    s = re.sub(r"^\s*the\s+", "", s, flags=re.IGNORECASE)
    s = re.sub(r"\s+", " ", s).strip(" .,")
    # cut trailing OCR fragments after the org keyword ("…Department M" → "…Department")
    m = re.search(r"^(.*?\b(?:Office|Department|Bureau|Division|Academy))\b", s,
                  re.IGNORECASE)
    if m:
        s = m.group(1)
    s = re.sub(r"\s+[A-Z]$", "", s)
    # title-case ALL-CAPS words (ARKANSAS → Arkansas), keep short acronyms (AR)
    s = " ".join(w.title() if (w.isupper() and len(w) > 3) else w
                 for w in s.split())
    return s.strip() or "N/A"


def _normalize_name(s: str) -> str:
    if not s or s == "N/A":
        return "N/A"
    return s.title() if s.isupper() else s


def extract_location(text: str, ner: dict) -> str:
    # 1. Sender department from the letterhead (most reliable)
    lh = _letterhead_location(text)
    if lh:
        return _normalize_location(lh)
    # 2. Labeled "From:" sender line
    m = re.search(r"from\s*[:\-]\s*(.{4,80})", text, re.IGNORECASE)
    if m:
        val = re.sub(r"\s*\n\s*", " ", m.group(1)).strip(" .,")
        if len(val) > 4 and not _TITLE_RE.match(val) \
                and not re.match(r"p\.?o\.?\s*box", val, re.IGNORECASE) \
                and not _is_blocked_loc(val):
            return _normalize_location(val)
    # 3. spaCy ORG, law-enforcement orgs ranked first (skip recipient/program orgs)
    le, generic = [], []
    for org in ner["ORG"]:
        clean = re.sub(r"\s*\n\s*", " ", org).strip()
        if len(clean.split()) < 2 or _is_blocked_loc(clean):
            continue
        low = clean.lower()
        if any(k in low for k in _LE_KEYWORDS):
            le.append(clean)
        elif any(k in low for k in _ORG_KEYWORDS):
            generic.append(clean)
    for org in le + generic:
        return _normalize_location(org)
    for gpe in ner["GPE"]:
        if len(gpe) > 3 and not _is_blocked_loc(gpe):
            return _normalize_location(gpe)
    return "N/A"


# Leadership titles that identify the head/signer of an agency letter
_HEAD_TITLE = r"(?:Sheriff|Chief\s+of\s+Police)"
_NAME = r"[A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+"
_RECIPIENT_NAMES = {"james ray"}      # the federal program contact (a recipient)


def _recipients(text: str) -> set:
    """Names addressed to (To:/Dear/Mr.) — these are recipients, not the officer."""
    out = set(_RECIPIENT_NAMES)
    for m in re.finditer(
            r"(?:^|\n)\s*(?:to|dear|attn)\s*[:\-]?\s*"
            r"((?:mr|mrs|ms|cpl|corporal|capt|lt|sgt|officer|chief)\.?\s*)?"
            r"([A-Z][a-zA-Z\.]+(?:\s+[A-Z][a-zA-Z\.]+){0,2})",
            text[:900], re.IGNORECASE):
        n = _clean_name(m.group(2)).lower()
        if n:
            out.add(n)
    return out


def extract_officer(text: str, ner: dict) -> str:
    head = text[:700]
    recipients = _recipients(text)

    def ok(name: str) -> bool:
        return bool(name) and len(name.split()) >= 2 and name.lower() not in recipients

    # 1. Head title → name  (e.g. "Sheriff Kelley Cradduck")
    m = re.search(_HEAD_TITLE + r"\s*[:\-]?\s*(" + _NAME + r")", head)
    if m:
        name = _clean_name(m.group(1))
        if ok(name):
            return name
    # 2. Name → head title  (e.g. "Mark Kizer, Chief of Police")
    m = re.search(r"(" + _NAME + r")\s*[,\n]?\s*" + _HEAD_TITLE + r"\b", head)
    if m:
        name = _clean_name(m.group(1))
        if ok(name):
            return name
    # 3. Closing signature
    m = re.search(
        r"(?:respectfully|sincerely|regards)[,\.]?\s*\n[\s\S]{0,80}?(" + _NAME + r")",
        text, re.IGNORECASE)
    if m:
        name = _clean_name(m.group(1))
        if ok(name):
            return name
    # 4. "PREPARED BY:" (single line — don't run into the next field)
    m = re.search(r"prepared\s+by\s*[:\-]\s*([^\n]{3,40})", text, re.IGNORECASE)
    if m:
        name = _clean_name(m.group(1))
        if ok(name):
            return name
    # 5. spaCy PERSON (excluding recipients)
    for person in ner["PERSON"]:
        name = _clean_name(person)
        if ok(name):
            return name
    return "N/A"


_NOISE_LINE = re.compile(
    r"^\s*(?:\(?\d{3}\)?[\s\.\-]\d{3}[\s\.\-]\d{4}|p\.?o\.?\s*box"
    r"|fax\s*[:\(#]|www\.|\[table\])", re.IGNORECASE)
_MEMO_HEADER = re.compile(r"^(?:to|from|date|ref|re|subject)\s*[:\-]",
                          re.IGNORECASE)
_ADDR_LINE = re.compile(r"^\s*\d{2,}\s+[A-Za-z]")          # "312 Roya Lane", "700 West…"
_NAMETITLE = re.compile(r",\s*(?:chief|sheriff|chief of police|chief deputy"
                        r"|director)\b", re.IGNORECASE)     # "Mark Kizer, Chief of Police"
_PURPOSE_HDR = re.compile(r"^\s*(?:\d+\.?|[ivx]+\.?|[a-z]\.)?\s*purpose\s*:?\s*$",
                          re.IGNORECASE)
# Form-field labels in lesson plans / course sheets — stripped so the summary
# picks the real sentence, not "COURSE: DURATION: 5.0 Hours …"
_FIELD_LABEL = re.compile(
    r"^\s*(?:course\s+title|course|lesson\s+title|duration\s+of\s+class|duration"
    r"|training\s+level|prepared\s+by|number\s+of\s+trainees"
    r"|method\s+of\s+presentation|instructor|target\s+group"
    r"|level\s+of\s+instruction|date\s+of\s+instruction|learning\s+goal"
    r"|performance\s+objectives?|references|training\s+aids|revisions?|subject"
    r"|purpose\s+of\s+the\s+course|primary\s+purpose|required\s+training"
    r"|driver\s+requirements|approved\s+by)\s*:\s*", re.IGNORECASE)
# A sentence stating the document's purpose makes the best summary
_PURPOSE_RE = re.compile(
    r"(intended to document|purpose of (?:the|this)|the purpose|this policy"
    r"|this lesson|provides? (?:clear )?guidelines|will be utilized"
    r"|is to (?:be )?(?:used|utilized|familiarize)|for the successful completion"
    r"|to familiarize|designed to|intends? to|in response to"
    r"|student will (?:become|be able)|will become familiar)", re.IGNORECASE)


def _is_letterhead_sentence(s: str) -> bool:
    """A 'sentence' that is really a letterhead/title block: mostly Capitalized
    words, few lowercase connectors (i.e. no real prose)."""
    words = [w for w in s.split() if w.isalpha()]
    if len(words) < 4:
        return True
    caps = sum(1 for w in words if w[0].isupper())
    return caps / len(words) > 0.6


def _is_letterhead_line(line: str) -> bool:
    """Drop a line that is letterhead: org/title block, address, name+title."""
    s = line.strip()
    if not s:
        return True
    if _NOISE_LINE.match(s) or _MEMO_HEADER.match(s) or _ADDR_LINE.match(s):
        return True
    if _NAMETITLE.search(s) and len(s) < 55:
        return True
    words = [w for w in s.split() if any(c.isalpha() for c in w)]
    if len(words) >= 2 and len(s) < 60:
        caps = sum(1 for w in words if w[0].isupper())
        if caps / len(words) > 0.8:               # ALL-CAPS / title block
            return True
    return False


_SALUTATION_RE = re.compile(
    r"^\s*(?:t[oa]\s*:?\s*whom\s+it\s+may\s+concern|dear\s+[^,]{0,40}"
    r"|re\s*:[^\n]{0,30})[,.:]?\s*", re.IGNORECASE)


def _finalize_summary(s: str) -> str:
    """Clean a chosen summary: drop a leading salutation, fix OCR spelling,
    trim trailing stray list markers ('… Protection 1.'), end cleanly."""
    s = _SALUTATION_RE.sub("", s)
    s = re.sub(r"^\([A-Za-z]{1,4}\s+", "", s)   # drop leading OCR junk like "(Cy "
    s = _correct_ocr(s, aggressive=True)
    s = re.sub(r"(?<=\s)[^\w\s]{1,3}(?=\s)", " ", s)   # drop stray symbol tokens ("=~", "[]")
    s = re.sub(r"\s+", " ", s).strip()
    for _ in range(3):                       # peel trailing markers like " 1." " a."
        s = re.sub(r"[\s,:;]+$", "", s)
        s = re.sub(r"[\s:]+[A-Za-z0-9]\.?$", "", s)
    s = s.strip(" ,:;")
    if len(s) > 297:                         # truncate at a word boundary, not mid-word
        s = s[:297].rsplit(" ", 1)[0].rstrip(" ,:;")
    if s:
        s = s[0].upper() + s[1:]             # capitalize first character
    if s and not s.endswith((".", "!", "?")):
        s += "."
    return s


def _first_good_sentence(blob: str) -> str | None:
    blob = re.sub(r"^\s*[A-Za-z]\.\s*", "", blob).strip()   # drop a leading "A." marker
    for s in re.split(r"(?<=[.!?])\s+", blob):
        s = s.strip()
        if len(s) > 40 and not _is_letterhead_sentence(s):
            return s
    return None


def extract_summary(text: str, doc_type: str, location: str, date: str) -> str:
    raw = text.splitlines()
    # 1. SOP/policy: the sentence right after a "Purpose" header line
    for i, l in enumerate(raw):
        if _PURPOSE_HDR.match(l):
            s = _first_good_sentence(" ".join(raw[i + 1:i + 6]))
            if s:
                return _finalize_summary(s)
            break
    # Strip form-field labels, drop letterhead + short fragment lines, keep prose
    lines = []
    for l in raw:
        l2 = _FIELD_LABEL.sub("", l).strip()
        if len(l2) < 20 or _is_letterhead_line(l2):
            continue
        lines.append(l2)
    clean = re.sub(r"\s+", " ", " ".join(lines)).strip()
    sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", clean)
                 if len(s.strip()) > 40]
    # 2. Prefer a sentence stating the document's purpose; trim any letterhead
    #    or date prefix that sits before the purpose phrase.
    for s in sentences:
        m = _PURPOSE_RE.search(s)
        if m and not _is_letterhead_sentence(s):
            prefix = s[:m.start()]
            # Only trim a *substantial* letterhead/date prefix; keep short
            # sentence subjects like "The student " / "The MRAP ".
            if len(prefix.strip()) > 15 and (re.search(r"\d", prefix)
                                             or _is_letterhead_sentence(prefix)):
                return _finalize_summary(s[m.start():])
            return _finalize_summary(s)
    # 3. Otherwise the first real prose sentence
    for s in sentences:
        if not _is_letterhead_sentence(s):
            return _finalize_summary(s)
    # 4. Templated fallback for form-like docs with no prose
    loc = location if location != "N/A" else "unknown agency"
    dt = f", {date}" if date != "N/A" else ""
    return f"{doc_type} from {loc}{dt}."


def extract_suspect(text: str) -> str:
    m = re.search(r"suspect\s*[:\-]\s*(.{20,250})", text, re.IGNORECASE)
    return m.group(1).strip().split("\n")[0] if m else "N/A"


def extract_outcome(text: str) -> str:
    m = re.search(r"(?:outcome|result|conclusion|approved|denied)\s*[:\-]\s*"
                  r"(.{15,200})", text, re.IGNORECASE)
    if m:
        val = m.group(1).strip().split("\n")[0]
        if len(val) > 15:
            return val
    return "N/A"


# ── OCR spell-correction post-processing ─────────────────────────────────────

# Deterministic fixes for recurring OCR errors seen in this corpus. Applied to
# every text field (safe, high-precision). Extend as new errors are found.
_OCR_FIX = {
    "resistatant": "Resistant", "resistataant": "Resistant",
    "resistatant": "Resistant", "dacumentation": "documentation",
    "documentaton": "documentation", "ammual": "annual", "frorn": "from",
    "simith": "Smith", "ofthe": "of the", "stcering": "steering",
    "dahiem": "Dahlem", "bovd": "Boyd", "employes": "employs",
}
# Tokens the general spell pass must never "correct" (acronyms / domain terms)
_SPELL_PROTECT = {
    "mrap", "swat", "leso", "cleet", "clest", "arv", "sort", "mou", "mous",
    "fbi", "dea", "atf", "faa", "cdl", "poc", "dodaac", "pcso", "bnpd",
    "gunfire", "preventative", "rollover", "egress", "deployable",
}
_SPELL = SpellChecker(distance=1)        # edit-distance 1 keeps corrections safe


def _correct_ocr(text: str, aggressive: bool = False) -> str:
    """Correct OCR spelling errors. The curated map always runs; the general
    spell pass (aggressive=True, for summaries) only touches lowercase words and
    skips proper nouns / acronyms so agency and officer names are never altered."""
    if not text or text == "N/A":
        return text
    for bad, good in _OCR_FIX.items():
        text = re.sub(rf"\b{bad}\b", good, text, flags=re.IGNORECASE)
    text = re.sub(r"\b1s\b", "is", text)
    text = re.sub(r"\s\|\s", " I ", text)
    if aggressive:
        out = []
        for tok in text.split():
            core = re.sub(r"[^A-Za-z]", "", tok)
            if (len(core) >= 4 and core.islower() and core not in _SPELL_PROTECT
                    and core in _SPELL.unknown([core])):
                corr = _SPELL.correction(core)
                if corr and corr != core:
                    tok = tok.replace(core, corr)
            out.append(tok)
        text = " ".join(out)
    return text


def _needs_review(row: dict, conf_mean: float) -> str:
    """Flag cells that warrant a human check, with the reason(s)."""
    reasons = []
    letter = row["Incident_Type"] in ("Cover Letter", "Letter", "Memorandum")
    if conf_mean < 70:
        reasons.append("low_OCR")
    if letter and row["Officer"] == "N/A":
        reasons.append("no_officer")
    if row["Location"] == "N/A":
        reasons.append("no_location")
    if letter and row["Date"] == "N/A":
        reasons.append("no_date")
    if " from " in row["Summary"] and row["Summary"].rstrip().endswith("."):
        reasons.append("templated_summary")
    return ";".join(reasons)


def extract_fields(seg: dict, nlp) -> dict:
    text = seg["text"]
    doc = nlp(text[:3000])
    ner = {"PERSON": [], "ORG": [], "GPE": []}
    for ent in doc.ents:
        if ent.label_ in ner:
            ner[ent.label_].append(ent.text.strip())
    date = extract_date(text)
    # curated OCR fixes on names/locations (no general spell pass → proper nouns safe)
    location = _correct_ocr(extract_location(text, ner))
    officer = _correct_ocr(_normalize_name(extract_officer(text, ner)))
    row = {
        "Report_ID": seg["report_id"],
        "Incident_Type": seg["doc_type"],
        "Date": date,
        "Location": location,
        "Officer": officer,
        "Summary": extract_summary(text, seg["doc_type"], location, date),
        "Suspect_Description": extract_suspect(text),
        "Outcome": extract_outcome(text),
    }
    row["Needs_Review"] = _needs_review(row, seg.get("conf", 100.0))
    return row


# Sub-document types that belong to a department submission and may inherit the
# agency/location from that submission's cover letter.
_INHERIT_TYPES = {
    "Training Lesson Plan", "Standard Operating Procedure",
    "Divisional Operating Procedure", "Department Policy",
    "Policies and Procedures", "Course Cover Sheet", "Course Summary Sheet",
    "Course Outline", "Reference Materials", "Road Course Evaluation",
}
# Standalone docs whose location must NOT seed the inherited agency
_NON_SOURCE_TYPES = {"Training Certificate", "Invoice"}


def inherit_locations(rows: list[dict]) -> None:
    """Fill a sub-document's missing/partial Location from the agency established
    by the most recent department record (usually its cover letter).

    Only fills 'N/A' or completes a partial match (e.g. 'Smith Police Department'
    → 'Fort Smith Police Department'); never overrides a genuinely different
    agency (Benton County Sheriff's Office vs Benton Police Department).
    """
    current = None
    for r in rows:
        loc, typ = r["Location"], r["Incident_Type"]
        if typ in _INHERIT_TYPES and current:
            if loc == "N/A" or (loc != current and loc in current):
                r["Location"] = current
                if "no_location" in r["Needs_Review"]:
                    r["Needs_Review"] = ";".join(
                        p for p in r["Needs_Review"].split(";") if p != "no_location")
        if r["Location"] != "N/A" and typ not in _NON_SOURCE_TYPES:
            current = r["Location"]


# ════════════════════════════════════════════════════════════════════════════
# Main
# ════════════════════════════════════════════════════════════════════════════

def _load_nlp():
    """Prefer the larger, more accurate model; fall back to the small one."""
    for model in ("en_core_web_lg", "en_core_web_sm"):
        try:
            nlp = spacy.load(model)
            print(f"Loaded spaCy model: {model}")
            return nlp
        except OSError:
            continue
    raise RuntimeError("No spaCy English model installed")


def main():
    global PDF_PATH, WRITE_DEBUG
    ap = argparse.ArgumentParser(description="Extract structured records from a "
                                             "bundled police-report PDF.")
    ap.add_argument("--pdf", default=PDF_PATH, help="path to the source PDF")
    ap.add_argument("--debug", action="store_true",
                    help="also write ocr_raw.txt and segments.json")
    args = ap.parse_args()
    PDF_PATH = args.pdf
    WRITE_DEBUG = WRITE_DEBUG or args.debug

    nlp = _load_nlp()

    print("\n[Phase 1] PDF → text")
    pages = pdf_to_pages(PDF_PATH)

    if WRITE_DEBUG:
        with open(RAW_TEXT_PATH, "w", encoding="utf-8") as f:
            for p in pages:
                f.write(f"\n{'='*60}\nPAGE {p['page']}  "
                        f"[{p['method']} conf={p['conf']} "
                        f"non_text={p['non_text']}]\n{'='*60}\n{p['text']}\n")
        print(f"  Raw text → {RAW_TEXT_PATH}")

    print("\n[Phase 2] Segmenting into sub-documents")
    segments = segment_pages(pages)
    records = [s for s in segments if s["report_id"]]
    skipped = [s for s in segments if not s["report_id"]]
    print(f"  {len(records)} records, {len(skipped)} non-text group(s) skipped")
    for s in segments:
        rid = s["report_id"] or "  (skip) "
        print(f"  {rid} | p{s['page_start']:>2}-{s['pages'][-1]:<2} "
              f"| score={s['score']:<2} | {s['doc_type']}")

    if WRITE_DEBUG:
        with open(SEGMENTS_JSON, "w") as f:
            json.dump([{k: v for k, v in s.items() if k != "text"}
                       for s in segments], f, indent=2)
        print(f"  Segment map → {SEGMENTS_JSON}")

    print("\n[Phase 3] Extracting fields")
    conf_by_page = {p["page"]: p["conf"] for p in pages}
    for s in records:
        cs = [conf_by_page.get(p, 100.0) for p in s["pages"]]
        s["conf"] = sum(cs) / len(cs) if cs else 100.0
    rows = [extract_fields(s, nlp) for s in records]
    inherit_locations(rows)   # fill sub-doc locations from their cover letter

    with open(OUTPUT_CSV, "w", newline="", encoding="utf-8") as f:
        # extrasaction="ignore" → the internal Needs_Review key is dropped from CSV
        w = csv.DictWriter(f, fieldnames=FIELDNAMES, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"  {len(rows)} records → {OUTPUT_CSV}")

    # QA aid: report records worth a manual check (console only, not in the CSV)
    flagged = [r for r in rows if r.get("Needs_Review")]
    if flagged:
        print(f"\n  {len(flagged)} record(s) worth review:")
        for r in flagged:
            print(f"    {r['Report_ID']}: {r['Needs_Review']}")


if __name__ == "__main__":
    main()
