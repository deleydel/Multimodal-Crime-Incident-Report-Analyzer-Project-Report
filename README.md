# Multimodal Crime / Incident Report Analyzer

An automated, end-to-end data processing pipeline and analytical workspace that ingests heterogeneous, unstructured 
streams (audio, text, PDF, images, and video footage) and programmatically serializes them into a unified, schema-aligned 
incident database.


## Project Architecture & Flow

Every day, city emergency management teams receive incident data in highly disjointed formats. This prototype replaces 
manual cross-referencing by processing each data stream through dedicated AI and heuristic extraction layers, merging the 
outputs into a common relational table.

* **Ingestion:** Raw artifacts are mapped into modular subdirectories.
* **Extraction:** Separate computational models extract critical keys: Event Type, Location, Timeline, and Priority Signals.
* **Integration:** A centralized pandas orchestration layer combines individual outputs using a vertical union technique, resolves missing parameters, and applies a dynamic severity scoring logic.
* **Downstream View:** Data is served dynamically through an interactive dashboard built using Streamlit.


##  Repository Directory Structure

The repository isolates data modalities and code bases into clean folders to maintain reproducible runtime settings:

```text
├── audio/
│   ├── audio_analysis_pipeline.ipynb   # Audio speech-to-text processing script
│   └── audio_extracted_report.csv      # Extracted audio metadata output
├── pdf/
│   ├── pdf_processing_pipeline.ipynb   # Document parsing and OCR notebook
│   └── pdf_extracted_report.csv        # Segmented official record output
├── images/
│   ├── image_analysis_node.py          # YOLOv8 object detection script
│   └── image_extracted_report.csv      # Scene metadata output
├── video/
│   ├── video_surveillance_node.py       # Frame extraction & difference script
│   └── video_extracted_report.csv      # Timestamped activity log output
├── text/
│   ├── text_nlp_engine.py              # Zero-shot classification script
│   └── text_extracted_report.csv       # Social/news text output
├── integration/
│   ├── app.py                          # Main Streamlit dashboard application
│   ├── master_orchestration.py         # Table alignment & combination script
│   └── master_incident_dataset.csv     # Merged cross-modality master dataset
├── README.md                           # Master workspace documentation
└── requirements.txt                    # Consolidated system dependencies
