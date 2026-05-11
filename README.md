# Green bond research reproduction

Everything needed to rebuild the empirical outputs and manuscript lives in this directory: data, code, exported tables and figures, and document sources.

## Layout

| Path | Purpose |
|------|---------|
| `manuscript/` | `researchpaperfinal.pdf`, LaTeX source, and figure PDFs used by the manuscript |
| `sources/` | R Markdown used to weave narrative, tables, and figures |
| `data/raw/` | Inputs for constructing `features.csv` |
| `data/processed/` | Processed panel (`features.csv`) and `.RData` objects from volatility models |
| `data/exports/tables/` | Regression and diagnostic tables (CSV) |
| `data/exports/figures/` | VAR / GARCH / case-study figures (PDF) |
| `code/replication.py` | Python driver: sample construction, core regressions, VAR appendix, vignettes |
| `code/garch_models.R` | Univariate / multivariate volatility models (`Rscript`) |
| `code/configs/config.yaml` | Sample window, paths, and model options |
| `code/requirements.txt` | Python dependencies |
| `MANIFEST.sha256` | Integrity checksums (every file listed except this manifest path) |

## Requirements

**Python** 3 with packages from `requirements.txt`. **R** with `rugarch`, `rmgarch`, `ggplot2`, and `yaml` for the GARCH step.

`code/data` is a symlink to `../data`.

## Run empirical pipeline

```bash
cd code
python3 -m pip install -r requirements.txt
python3 replication.py --stage full
```

Use `--stage` with one of: `features`, `nlp`, `regressions`, `garch`, `var`, `case_studies`, or `full`. Outputs accumulate under `data/exports/` and `data/processed/`; nothing is deleted automatically.

## Rebuild the manuscript PDF

```bash
cd manuscript
pdflatex -interaction=nonstopmode Final_Paper_Submission_Final.tex
pdflatex -interaction=nonstopmode Final_Paper_Submission_Final.tex
```

## Check file integrity

From this directory:

```bash
shasum -a 256 -c MANIFEST.sha256
```
