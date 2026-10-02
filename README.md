# Ilocano Hymn Lexicon Benchmark

A reproducible, lightweight, classical-ML Python project that benchmarks how well Ilocano lexicons help a system recognise different kinds of Ilocano text.

## Quick Start

### Install

```bash
py -3.13 -m pip install -r requirements.txt
```

### GUI (easiest way)

```bash
py -3.13 gui.py
```

A desktop window opens with tabs for:

| Tab | What you can do |
|-----|-----------------|
| **Run benchmark** | One-click run (fast or full mode) with live log output |
| **Report** | Read `results/REPORT.md` |
| **Figures** | Browse every PNG plot; open files/folders |
| **Metrics** | Browse every CSV table (`task_c_*`, coverage, corpus stats, …) |
| **Song search** | Type a phrase → top matching songs (Task C retrieval) |
| **Logs** | Read `logs/benchmark_run.log`, corpus inspection, normalisation rules |

The GUI always runs from the project folder (`ilocano-hymn-benchmark/`) and uses the same commands as the CLI below.

### One-command run (CLI)

```bash
py -3.13 -m src.run_benchmark --config config.yaml
```

### Fast mode (fewer folds/repeats for quick checks)

```bash
py -3.13 -m src.run_benchmark --config config.yaml --fast
```

### Run tests

```bash
py -3.13 -m pytest tests/ -v
```

## Data Placement

### Input files (already placed)
- `data/raw/corpus_ilocano_liturgy.txt` — Main corpus
- `data/raw/unique_words.txt` — 672 unique words
- `data/lexicon_house/*.csv` — House-made annotated lexicon CSVs

### Adding house lexicon CSVs
Drop additional `.csv` files into `data/lexicon_house/`. The loader automatically:
- Globs all `*.csv` files (sorted by filename)
- Tolerates different column orders and extra columns
- Reports duplicates and coverage

### Adding genre data
Create `.txt` files in the appropriate genre directory:
- `data/genres/basic_text/<source>.txt` — Plain Ilocano text
- `data/genres/conversation/<source>.txt` — Dialogue/spoken text
- `data/genres/poetry/<source>.txt` — Ilocano poetry

Format: one passage per blank-line-separated block.

Also update `data/genres/manifest.csv` with the source metadata.

### External lexicons
- Wiktionary: auto-downloaded on first run (or place `wiktionary_ilocano.jsonl` in `data/lexicon_external/`)
- PanLex: auto-queried on first run (or place `panlex_ilocano.json` in `data/lexicon_external/`)
- User-supplied: place headword list in `data/lexicon_external/user_supplied/`

## Retrieval CLI

Search for a song by phrase:

```bash
py -3.13 -m src.retrieval "Apo caasiannacami"
py -3.13 -m src.retrieval "Gloria coma iti Ama" -k 3
```

## Reading Results

- `results/REPORT.md` — Full benchmark report
- `results/metrics/` — CSV files with per-fold and aggregated metrics
- `results/figures/` — PNG + SVG visualisations
- `results/errors/` — Misclassified samples with error analysis
- `logs/` — Detailed run logs, corpus inspection, normalisation rules

## Project Structure

```
ilocano-hymn-benchmark/
├── AGENTS.md              # Full project specification
├── PLAN.md                # Plan and assumptions
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── config.yaml            # Configuration
├── Makefile               # Build automation
├── gui.py                 # Simple desktop GUI (run this for easiest use)
├── data/                  # Input data
├── src/                   # Source code
│   ├── data_loader.py     # Corpus parsing, boundary detection
│   ├── preprocess.py      # Normalisation, variant mapping
│   ├── lexicons.py        # Lexicon interface and loaders
│   ├── genres.py          # Genre data loading
│   ├── features.py        # Feature extraction
│   ├── models.py          # Model zoo
│   ├── splits.py          # CV splitters
│   ├── evaluate.py        # Metrics and significance tests
│   ├── retrieval.py       # Song retrieval (Task C)
│   ├── analysis.py        # Error analysis and ablations
│   ├── plots.py           # Visualisations
│   └── run_benchmark.py   # Main entry point
├── tests/                 # Test suite
├── results/               # Output
└── logs/                  # Run logs
```
