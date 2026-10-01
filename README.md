# Second Brain

**A local-first research workspace for literature discovery and manuscript drafting.**

Second Brain brings a personal paper library, source-grounded Q&A, lightweight bibliometrics, and an editable manuscript workspace together in one Python app. It runs with models served by the user's own Ollama installation.

[Open the HTML getting-started guide](README.html)

## What it does

- Imports PDF, DOCX, XLSX, PPTX, PNG, JPG, WEBP, TIFF, TXT, and CSV sources.
- Extracts PDF pages, Office text and tables, spreadsheet cells and chart series, and presentation content.
- Uses Qwen-VL to analyze selected visual content and up to two scanned pages per PDF.
- Stores source copies, extracted passages, bibliographic metadata, and manuscript drafts in local SQLite.
- Searches full text and filters the library by publication year and affiliation country.
- Shows publication trends, author-keyword themes, country counts, and an affiliation-centroid map.
- Corrects metadata in an editable grid; asks Qwen questions and drafts/paraphrases manuscript sections.
- Provides a page preview, local autosave, and DOCX export.

## Quick start

Requirements: Python 3.12 and Ollama installed and running.

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

Open `http://localhost:8501`, then set model names in the sidebar to match `ollama list`.

## Models

```powershell
ollama pull qwen3:8b
ollama pull qwen3-vl:8b
ollama pull nomic-embed-text
```

`qwen3:8b` handles Q&A and writing. `qwen3-vl:8b` reads visual content. `nomic-embed-text` is optional; without it, retrieval falls back to keyword matching. `qwen3-coder` is not required at runtime.

## Privacy and research integrity

Imported papers, extracted text, metadata, indexes, and drafts live under `data/`. Model requests go to the Ollama address configured in the app; use a local address if documents must not leave this computer. The map uses Carto basemap tiles, but paper text is not sent to Carto. `data/`, `Referensi/`, `.venv/`, `.env`, and `.streamlit/secrets.toml` are excluded by Git ignore rules.

Country and keyword metadata are heuristic detections and should be reviewed before bibliometric interpretation. Map pins represent country centroids from author affiliations, not research or experiment sites. Qwen output is a draft: verify claims and citations against the original papers. Study design, data, results, and novelty must be supplied and reviewed by the researcher.

## Limitations

- Automatic visual analysis is capped at two pages per PDF; larger scans need selective review.
- Spreadsheet and presentation charts are indexed from underlying series data; visual styling and annotations may be lost.
- Citations identify source file and page/sheet/slide, but do not yet produce a full APA/IEEE bibliography or DOI metadata.
- The preview is not a fully paginated WYSIWYG Word editor.

## Tests

```powershell
python -m unittest discover -s tests -v
```
