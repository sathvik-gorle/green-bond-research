# about

## Layout

| Path | Purpose |
|------|---------|
| `manuscript/` | `researchpaperfinal.pdf`, LaTeX source, and figure PDFs used by the manuscript |
| `sources/researchpaper.Rmd` | R Markdown used to weave narrative, tables, and figures |
| `paper/` | Supporting materials for Rmd (e.g. `references.bib`) |
| `data/raw/` | Inputs for `features.csv`; **ACRI modeling** uses **`ACRI_loss`** (Climate Risk Loss index) for `ACRI`/`ACRI_z`; **`ACRI_scaled`** in `acri_*.csv` is archival-only (mis-scaled composite export) |
| `data/processed/` | Processed panel (`features.csv`) and `.RData` objects from volatility models |
| `data/exports/tables/` | Regression and diagnostic tables (CSV) |
| `data/exports/figures/` | VAR / GARCH / case-study figures (PDF) |
| `code/replication.py` | Python driver: sample construction, core regressions, VAR appendix, vignettes |
| `code/garch_models.R` | Univariate / multivariate volatility models (`Rscript`) |
| `code/configs/config.yaml` | Sample window, paths, and model options |
| `code/requirements.txt` | Python dependencies |
| `code/install_r_deps.R` | One-shot CRAN installer for R packages used by GARCH / Rmd |
| `code/validate_inputs.py` | Checks raw inputs and `features.csv` before a long run |
| `scripts/reproduce.sh` | End-to-end replication (pip + Python full + R GARCH) |
| `MANIFEST.sha256` | Integrity checksums for tracked files (see exclusion rules below) |

Paths in code resolve to the **repository root** (`greenbondfinalfolder/`): processed data lives in `data/`, never via a symlink under `code/`.

## Requirements

**Python** 3 with packages from `code/requirements.txt`.

**R** with CRAN packages for GARCH and optional R Markdown rendering. Install once:

```bash
Rscript code/install_r_deps.R
```

## One-command replication

From the repository root (`greenbondfinalfolder/`):

```bash
bash scripts/reproduce.sh
```

This installs Python deps and runs `python3 code/replication.py --stage full`. The Python `full` stage invokes the R GARCH script as part of the run.

(Optional) Validate inputs **after** building `features.csv` (fail-fast on empty or corrupt panel):

```bash
python3 code/validate_inputs.py
```

## Run empirical pipeline manually

Always run from **repository root** (not inside `code/`), unless you explicitly `cd` and adjust paths:

```bash
python3 -m pip install -r code/requirements.txt
python3 code/replication.py --stage full
```

Use `--stage` with one of: `features`, `nlp`, `regressions`, `garch`, `var`, `case_studies`, or `full`. Outputs accumulate under `data/exports/` and `data/processed/`; nothing is deleted automatically.

## Rebuild the manuscript PDF

```bash
cd manuscript
pdflatex -interaction=nonstopmode Final_Paper_Submission_Final.tex
pdflatex -interaction=nonstopmode Final_Paper_Submission_Final.tex
```

## Check file integrity

Regenerate the manifest (excludes `.git/`, this file, `code/logs/`, `__pycache__/`, and `.pyc`):

```bash
find . -type f \
  -not -path "./.git/*" \
  -not -path "./MANIFEST.sha256" \
  -not -path "./code/logs/*" \
  -not -path "*/__pycache__/*" \
  -not -name "*.pyc" \
  -exec shasum -a 256 {} \; > MANIFEST.sha256
```

Verify:

```bash
shasum -a 256 -c MANIFEST.sha256
```
