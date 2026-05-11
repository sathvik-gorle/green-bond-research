#!/usr/bin/env python3
"""Empirical replication: build features and manuscript-listed tables."""

from __future__ import annotations

import argparse
import logging
import os
import re
import subprocess
import sys
import time
import warnings as py_warnings
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
import yaml
from scipy import stats
from statsmodels.tsa.api import VAR

py_warnings.filterwarnings("ignore")

CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = CODE_DIR.parent
logger = logging.getLogger("replication")


def load_config() -> dict:
    with open(CODE_DIR / "configs" / "config.yaml") as f:
        return yaml.safe_load(f)


def _expand_user_path(path: Path) -> Path:
    if str(path).startswith("~"):
        return Path(str(path).replace("~", str(Path.home())))
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def resolve_cleaned_paths(cfg: dict) -> tuple[Path, Path]:
    cleaned = cfg.get("paths", {}).get("cleaned_data", {})
    long_csv = _expand_user_path(
        Path(
            cleaned.get(
                "long_csv",
                str(Path.home() / "Downloads" / "greenbond021726_cleaned_long.csv"),
            )
        )
    )
    data2_csv = _expand_user_path(
        Path(
            cleaned.get(
                "data2_csv",
                str(Path.home() / "Downloads" / "green_bond_data2_cleaned.csv"),
            )
        )
    )
    return long_csv, data2_csv


def load_yahoo(raw_dir: Path) -> pd.DataFrame:
    p = raw_dir / "yahoo_prices.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df


def load_bloomberg1(long_csv_path: Path) -> pd.DataFrame:
    if not long_csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(long_csv_path, parse_dates=["date"])
    pivot = df.pivot_table(index="date", columns="instrument", values="last_price", aggfunc="first")
    rename = {
        "GBGLTRUU Index": "GBGLTRUU",
        "I36100US Index": "I36100US_bbg1",
        "BGRN US Equity": "BGRN_bbg",
        "USGG10YR Index": "Treasury10Y_bbg",
        "LUACOAS Index": "LUACOAS",
    }
    pivot = pivot.rename(columns=rename)
    cols_keep = [c for c in rename.values() if c in pivot.columns]
    return pivot[cols_keep]


def load_bloomberg2(data2_csv_path: Path) -> pd.DataFrame:
    if not data2_csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(data2_csv_path, parse_dates=["date"])
    daily = df[~df["instrument"].str.contains("monthly", case=False)]
    pivot = daily.pivot_table(index="date", columns="instrument", values="price", aggfunc="first")
    return pivot


def load_fred(raw_dir: Path) -> pd.DataFrame:
    p = raw_dir / "fred_extended.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df.index = pd.to_datetime(df.index)
    df.index.name = "date"
    return df


def load_cpu(raw_dir: Path) -> pd.DataFrame:
    p = raw_dir / "cpu_index.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return df[["cpu_index_broad"]].rename(columns={"cpu_index_broad": "CPU"})


def load_disasters(raw_dir: Path) -> pd.DataFrame:
    p = raw_dir / "noaa_disasters.parquet"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_parquet(p)
    df["date"] = pd.to_datetime(df["date"])
    df = df.set_index("date")
    return df


def _finalize_acri_physical_risk(acri_df: pd.DataFrame) -> pd.DataFrame:
    """
    Physical risk in the paper enters as ``ACRI_z`` from the **Climate Risk Loss** index
    (``ACRI_loss``) whenever present. Mis-scaled composite exports remain only in raw CSVs
    under ``ACRI_scaled`` and are not merged into ``features``.
    """
    if acri_df.empty:
        return acri_df
    out = acri_df.copy()
    if "ACRI_scaled" in out.columns:
        out = out.drop(columns=["ACRI_scaled"])
    if "ACRI_loss" in out.columns and out["ACRI_loss"].notna().any():
        out["ACRI"] = out["ACRI_loss"]
        logger.info(
            "ACRI level for modeling = ACRI_loss (Climate Risk Loss Index); "
            "mis-scaled composite is not passed into the feature panel."
        )
    return out


def _acri_csv_to_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Parse Date + optional ACRI_loss / ACRI_scaled / legacy ACRI from a raw extract."""
    out = pd.DataFrame(index=df.index)
    if "ACRI_loss" in df.columns:
        out["ACRI_loss"] = pd.to_numeric(df["ACRI_loss"], errors="coerce")
    if "ACRI_scaled" in df.columns:
        out["ACRI_scaled"] = pd.to_numeric(df["ACRI_scaled"], errors="coerce")
    elif "ACRI" in df.columns:
        out["ACRI_scaled"] = pd.to_numeric(df["ACRI"], errors="coerce")
    return _finalize_acri_physical_risk(out)


def load_acri(downloads: Path) -> pd.DataFrame:
    """Load ACRI-monthly aggregates from packaged CSV under ``downloads`` or the source workbook."""
    daily_csv_path = downloads / "acri_daily.csv"
    if daily_csv_path.exists():
        df = pd.read_csv(daily_csv_path)
        if "Date" not in df.columns:
            return pd.DataFrame()
        df["date"] = pd.to_datetime(df["Date"])
        df = df.set_index("date").sort_index()
        return _acri_csv_to_frame(df)

    csv_path = downloads / "acri_monthly.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        if "Date" not in df.columns:
            return pd.DataFrame()
        df["date"] = pd.to_datetime(df["Date"])
        df = df.set_index("date").sort_index()
        return _acri_csv_to_frame(df)

    for name in ["CLIMATE RISK.xlsx", "CLIMATE RISK (1).xlsx"]:
        p = downloads / name
        if p.exists():
            break
    else:
        return pd.DataFrame()
    try:
        df = pd.read_excel(p, sheet_name="mACRI_obs")
    except Exception:
        return pd.DataFrame()
    if "Year" not in df.columns or "Month" not in df.columns:
        return pd.DataFrame()
    df = df.dropna(subset=["Year", "Month"])
    df["date"] = pd.to_datetime(
        df["Year"].astype(int).astype(str) + "-" + df["Month"].astype(int).astype(str) + "-01"
    )
    df = df.set_index("date").sort_index()
    out = pd.DataFrame(index=df.index)
    if "ACRI" in df.columns:
        out["ACRI"] = pd.to_numeric(df["ACRI"], errors="coerce")
    if "Climate Risk Loss Index" in df.columns:
        out["ACRI_loss"] = pd.to_numeric(df["Climate Risk Loss Index"], errors="coerce")
    return _finalize_acri_physical_risk(out)


def load_individual_bonds(long_csv_path: Path) -> pd.DataFrame:
    """Extract individual green bond price series from Bloomberg cleaned CSV."""
    if not long_csv_path.exists():
        return pd.DataFrame()
    df = pd.read_csv(long_csv_path, parse_dates=["date"])
    # Mapping verified against price ranges and sheet_name:
    #   AN964384@IBVL Corp = Apple gb weekly (price ~93-99)
    #   AV504231@IBVL Corp = duke energy gb (price ~92-100)
    #   ZK885421@BGN Corp  = BAC GB EUR / JPM (price ~99-105)
    #   ZH532179@IBVL Corp = JPM 6.07 102227 (price ~100-104)
    #   BU751737@IBVL Corp = Nature Conservancy (sheet) AND NY Electric (sheet)
    #     Nature Conservancy dates: 2023-10 to 2026-02, price ~73-82
    #     NY electric dates: 2024-01 to 2026-01, price ~76-87
    #     On overlapping dates they have identical prices -> same physical bond,
    #     two sheet names. Use sheet_name to distinguish.
    bond_rows = []
    sheet_bond_map = {
        "Apple gb weekly":    "AppleGB",
        "duke energy gb":     "DukeEnergyGB",
        "Nature Conservancy": "NatureConservancyGB",
        "JPM 6.07 102227":   "JPMGB",
        "NY electric and gas":"NYElecGasGB",
        "BAC GB EUR":         "BACGB",
    }
    for _, row in df.iterrows():
        sn = row.get("sheet_name", "")
        if sn in sheet_bond_map:
            bond_rows.append({
                "date": row["date"],
                "name": sheet_bond_map[sn],
                "last_price": row["last_price"],
            })
    if not bond_rows:
        return pd.DataFrame()
    bonds = pd.DataFrame(bond_rows)
    pivot = bonds.pivot_table(index="date", columns="name", values="last_price", aggfunc="first")
    return pivot


def log_return(series: pd.Series) -> pd.Series:
    return np.log(series / series.shift(1)) * 100


def z_score(series: pd.Series) -> pd.Series:
    mu = series.expanding(min_periods=60).mean()
    sigma = series.expanding(min_periods=60).std()
    return (series - mu) / sigma


def build():
    cfg = load_config()
    start = pd.Timestamp(cfg["date_range"]["start"])
    end = pd.Timestamp(cfg["date_range"]["end"])
    paths = cfg.get("paths", {})
    raw_dir = REPO_ROOT / paths.get("raw", "data/raw")
    processed_dir = REPO_ROOT / paths.get("processed", "data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)
    long_csv_path, data2_csv_path = resolve_cleaned_paths(cfg)
    downloads = long_csv_path.parent

    # Load all sources
    yahoo = load_yahoo(raw_dir)
    bbg1 = load_bloomberg1(long_csv_path)
    bbg2 = load_bloomberg2(data2_csv_path)
    fred = load_fred(raw_dir)
    cpu = load_cpu(raw_dir)
    disasters = load_disasters(raw_dir)
    acri = load_acri(downloads)
    indiv_bonds = load_individual_bonds(long_csv_path)

    logger.info(f"Yahoo: {yahoo.shape}, Bloomberg1: {bbg1.shape}, Bloomberg2: {bbg2.shape}")
    logger.info(f"FRED: {fred.shape}, CPU: {cpu.shape}, Disasters: {disasters.shape}")
    logger.info(f"ACRI: {acri.shape}, Individual bonds: {indiv_bonds.shape}")

    # Build daily date index from the union of all sources
    all_dates = set()
    for df in [yahoo, bbg2, fred]:
        if not df.empty:
            all_dates.update(df.index)
    date_idx = pd.DatetimeIndex(sorted(all_dates), name="date")

    features = pd.DataFrame(index=date_idx)

    # --- Yahoo ETF prices ---
    if not yahoo.empty:
        for col in yahoo.columns:
            features[col] = yahoo[col].reindex(date_idx)

    # --- Bloomberg file 2 (daily indices) ---
    if not bbg2.empty:
        for col in bbg2.columns:
            features[col] = bbg2[col].reindex(date_idx)

    # --- Bloomberg file 1 (monthly, forward-fill to daily) ---
    if not bbg1.empty:
        for col in bbg1.columns:
            monthly = bbg1[col].dropna()
            features[col] = monthly.reindex(date_idx, method="ffill")

    # --- FRED macro ---
    if not fred.empty:
        for col in fred.columns:
            features[col] = fred[col].reindex(date_idx)

    # --- CPU (monthly -> forward-fill) ---
    if not cpu.empty:
        features["CPU"] = cpu["CPU"].reindex(date_idx, method="ffill")

    # --- Disasters (monthly -> forward-fill) ---
    if not disasters.empty:
        features["disaster_count"] = disasters["disaster_count"].reindex(date_idx, method="ffill")

    # --- ACRI Climate Risk Index (monthly -> forward-fill) ---
    if not acri.empty:
        for col in acri.columns:
            features[col] = acri[col].reindex(date_idx, method="ffill")

    # --- Individual Bloomberg bonds (monthly, forward-fill to daily) ---
    if not indiv_bonds.empty:
        for col in indiv_bonds.columns:
            monthly = indiv_bonds[col].dropna()
            features[col] = monthly.reindex(date_idx, method="ffill")

    # Forward-fill small gaps (weekends already handled by index union)
    features = features.ffill(limit=5)

    # Drop rows where nothing exists
    features = features.dropna(how="all")

    # Enforce study sample window from config
    before_window = len(features)
    features = features[(features.index >= start) & (features.index <= end)]
    dropped_window = before_window - len(features)
    logger.info(
        "Applied sample window %s to %s | dropped %d rows outside window",
        start.date(),
        end.date(),
        dropped_window,
    )

    # ====================================================================
    # Compute log returns for tradeable indices
    # ====================================================================
    return_map = {
        "BGRN": "r_BGRN",
        "LQD": "r_LQD",
        "IEF": "r_IEF",
        "TLT": "r_TLT",
        "SPY": "r_SPY",
        "HYG": "r_HYG",
        "VCIT": "r_VCIT",
        "KRBN": "r_KRBN",
        "ICLN": "r_ICLN",
        "USO": "r_USO",
        "GRNB": "r_GRNB",
        "AGG": "r_AGG",
        "MUB": "r_MUB",
        "SPGBI": "r_SPGBI",
        "SPGRNMS": "r_SPGRNMS",
        "I31572US": "r_I31572US",
        "MSCI_CT": "r_MSCI_CT",
        "GBGLTRUU": "r_GBGLTRUU",
    }

    for price_col, ret_col in return_map.items():
        if price_col in features.columns:
            features[ret_col] = log_return(features[price_col])

    # Key differentials
    if "r_BGRN" in features.columns and "r_LQD" in features.columns:
        features["r_diff_BGRN_LQD"] = features["r_BGRN"] - features["r_LQD"]

    if "r_I31572US" in features.columns and "r_LQD" in features.columns:
        features["r_diff_green_corp"] = features["r_I31572US"] - features["r_LQD"]

    # Rate-hedged differential: green return minus duration-matched treasury exposure
    if all(c in features.columns for c in ["r_BGRN", "r_LQD", "r_IEF"]):
        features["r_diff_hedged"] = features["r_BGRN"] - features["r_LQD"] - 0.3 * (features["r_IEF"] - features["r_LQD"])

    # Treasury and slope changes
    if "Treasury10Y" in features.columns:
        features["delta_10y"] = features["Treasury10Y"].diff()
    if "Treasury2Y" in features.columns:
        features["delta_2y"] = features["Treasury2Y"].diff()
    if "IG_OAS" in features.columns:
        features["delta_IG_OAS"] = features["IG_OAS"].diff()
    if "HY_OAS" in features.columns:
        features["delta_HY_OAS"] = features["HY_OAS"].diff()

    # ====================================================================
    # Z-scores
    # ====================================================================
    if "VIX" in features.columns:
        features["VIX_z"] = z_score(features["VIX"])
    if "EPU" in features.columns:
        features["EPU_z"] = z_score(features["EPU"])
    if "CPU" in features.columns:
        features["CPU_z"] = z_score(features["CPU"])
    if "disaster_count" in features.columns:
        features["disasters_z"] = z_score(features["disaster_count"])
    if "ACRI" in features.columns:
        features["ACRI_z"] = z_score(features["ACRI"])
    if "ACRI_loss" in features.columns:
        features["ACRI_loss_z"] = z_score(features["ACRI_loss"])

    # ====================================================================
    # Rolling volatilities (20d, 60d annualized)
    # ====================================================================
    vol_targets = ["r_BGRN", "r_LQD", "r_KRBN", "r_diff_BGRN_LQD",
                   "r_diff_green_corp", "r_diff_hedged", "r_SPGBI", "r_SPGRNMS",
                   "r_GRNB", "r_AGG", "r_MUB"]
    for col in vol_targets:
        if col in features.columns:
            features[f"rvol_20d_{col}"] = features[col].rolling(20, min_periods=15).std() * np.sqrt(252)
            features[f"rvol_60d_{col}"] = features[col].rolling(60, min_periods=45).std() * np.sqrt(252)

    # ====================================================================
    # Sector-level return aliases (for VAR model clarity)
    # ====================================================================
    # BBB OAS and trade-weighted dollar changes
    if "BBB_OAS" in features.columns:
        features["delta_BBB_OAS"] = features["BBB_OAS"].diff()
    if "TradeWeightedDollar" in features.columns:
        features["delta_TWD"] = features["TradeWeightedDollar"].pct_change() * 100

    alias_map = {
        "r_I31572US": "r_green_corp",
        "r_LQD": "r_conv_corp",
        "r_SPGBI": "r_global_green",
        "r_SPGRNMS": "r_muni_green",
        "r_IEF": "r_treasury",
        "r_MSCI_CT": "r_climate_transition",
        "r_MUB": "r_muni_broad",
        "r_AGG": "r_agg_bond",
    }
    for src, dst in alias_map.items():
        if src in features.columns:
            features[dst] = features[src]

    features.index.name = "Date"
    features = features.sort_index()

    # Save
    features.to_csv(processed_dir / "features.csv")
    features.to_parquet(processed_dir / "features.parquet")

    logger.info(f"\nFeatures: {features.shape}")
    logger.info(f"Date range: {features.index.min().date()} to {features.index.max().date()}")
    logger.info(f"Columns ({len(features.columns)}):")
    for c in features.columns:
        n = features[c].notna().sum()
        logger.info(f"  {c}: {n}/{len(features)} non-null")

    return features


def build_feature_matrix() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    build()
    logger.info("Features build complete.")

GREEN_SPECIFICITY_KEYWORDS = [
    # Technical climate / renewables
    r"\bsolar\b", r"\bwind\b", r"\brenewable\b", r"\bphotovoltaic\b",
    r"\bgeothermal\b", r"\bhydropower\b", r"\bbiomass\b", r"\bgreen\s*bond\b",
    r"\bclean\s*energy\b", r"\bcarbon\s*neutral\b", r"\bzero\s*emission\b",
    r"\bsustainable\b", r"\belectric\s*vehicle\b", r"\bev\s*charging\b",
    r"\bleed\b", r"\benergy\s*star\b", r"\bnet\s*zero\b", r"\bclimate\b",
    r"\benergy\s*efficiency\b", r"\bgreen\s*building\b", r"\bwater\s*treatment\b",
    r"\brecycl\w*\b", r"\bbiodiversity\b", r"\breforest\w*\b",
    r"\bflood\s*resilien\w*\b", r"\badaptation\b", r"\bmitigation\b",
    r"\bcarbon\s*capture\b", r"\bhydrogen\b", r"\bbattery\s*storage\b",
    r"\boffshore\s*wind\b", r"\btransmission\b", r"\bgrid\b",
    # Nature-based solutions & conservation (aligns with GBP adaptation/biodiversity categories)
    r"\brestoration\b", r"\bwatershed\b", r"\bmangrove\b", r"\bcoral\b",
    r"\becosystem\b", r"\bhabitat\b", r"\bwetland\b", r"\bpeatland\b",
    r"\bforest\b", r"\bconservat\w*\b", r"\bsoil\b",
]


def compute_green_specificity(
    green_bonds: pd.DataFrame,
    text_col: str = "uop_text",
    embedding_model: str = "all-MiniLM-L6-v2",
) -> pd.DataFrame:
    """
    Compute a continuous green specificity score (0-1) for each green bond.

    Two components:
      1. Keyword density: count of green-specific keywords / text length
      2. Embedding similarity: cosine similarity to "green ideal" centroid

    Combined: 0.5 * keyword_density_normalized + 0.5 * embedding_similarity
    """
    df = green_bonds.copy()

    if text_col not in df.columns:
        df["green_specificity"] = 0.5
        logger.warning(f"No {text_col} column. Setting green_specificity = 0.5")
        return df

    # Component 1: Keyword density
    keyword_scores = []
    for _, row in df.iterrows():
        text = row.get(text_col, "")
        if not isinstance(text, str) or not text.strip():
            keyword_scores.append(0.0)
            continue

        text_lower = text.lower()
        word_count = max(len(text_lower.split()), 1)
        match_count = sum(len(re.findall(pat, text_lower)) for pat in GREEN_SPECIFICITY_KEYWORDS)
        keyword_scores.append(match_count / word_count)

    df["keyword_density"] = keyword_scores

    # Normalize keyword density to 0-1
    kd_max = max(df["keyword_density"].max(), 1e-10)
    df["keyword_density_norm"] = df["keyword_density"] / kd_max

    # Component 2: Embedding similarity to green centroid
    emb_scores = np.full(len(df), 0.5)
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer(embedding_model)
        texts = df[text_col].fillna("").tolist()
        valid_mask = [bool(t.strip()) for t in texts]

        if sum(valid_mask) >= 5:
            valid_texts = [t for t, v in zip(texts, valid_mask) if v]
            embeddings = model.encode(valid_texts, show_progress_bar=False)

            # Green ideal centroid: average of top-20% keyword density bonds
            kd_vals = df.loc[[i for i, v in enumerate(valid_mask) if v], "keyword_density"].values
            top_threshold = np.percentile(kd_vals, 80)
            top_mask = kd_vals >= top_threshold

            if top_mask.sum() >= 2:
                green_centroid = embeddings[top_mask].mean(axis=0)
                for i, emb in enumerate(embeddings):
                    sim = np.dot(emb, green_centroid) / (
                        np.linalg.norm(emb) * np.linalg.norm(green_centroid) + 1e-10
                    )
                    sim = max(0.0, min(1.0, float(sim)))
                    valid_indices = [j for j, v in enumerate(valid_mask) if v]
                    emb_scores[valid_indices[i]] = sim

            logger.info(f"Computed embedding similarity for {sum(valid_mask)} bonds")
        else:
            logger.warning("Too few valid texts for embedding similarity. Using keyword only.")

    except ImportError:
        logger.warning("sentence-transformers not installed. Using keyword density only.")

    df["embedding_similarity"] = emb_scores

    # Combined score
    df["green_specificity"] = (
        0.5 * df["keyword_density_norm"] + 0.5 * df["embedding_similarity"]
    ).round(4)

    logger.info(f"Green specificity: mean={df['green_specificity'].mean():.3f}, "
                f"std={df['green_specificity'].std():.3f}")

    return df

UOP_TEXTS = {
    "Apple Inc.": """
    The 2019 Green Bond proceeds are intended to prioritize projects that mitigate our carbon emissions,
    including supporting the execution of Apple's 2030 carbon neutrality roadmap. Eligibility criteria:
    low carbon design and engineering, energy efficiency, renewable energy, carbon mitigation,
    and carbon sequestration. We first aim to leverage low-carbon product design, energy efficiency,
    clean electricity, and direct emissions abatement to reduce emissions by 75 percent by 2030.
    We then plan to address residual emissions by investing in high-quality carbon removal solutions.
    Apple allocated proceeds to operational projects with immediate direct carbon benefits,
    capacity-building projects that enable suppliers to achieve carbon emissions reductions,
    and research and development that will unlock future carbon reductions. We quantify the new
    renewable energy capacity we're adding to the grid through renewable energy projects.
    Supplier Clean Energy Program helps enable suppliers' transition to clean, renewable electricity.
    Supplier Energy Efficiency Program aims to help our suppliers optimize energy use. The use of
    recycled materials is central to our goal of making carbon neutral products by 2030.
    Solar photovoltaic projects, wind energy, renewable energy, carbon neutral, net zero, climate.
    Energy efficiency, clean energy, carbon sequestration, carbon mitigation, sustainable.
    """,
    "JPMorgan Chase": """
    The Sustainable Bond Framework includes eligible project categories. Green Projects: Renewable
    and Clean Energy, Green Buildings, Sustainable Transportation. Social Projects: Affordable
    Housing, Home Ownership, Education, Healthcare, Small Business and Microfinance. Proceeds will
    be allocated to fund Eligible Green Projects and/or Eligible Social Projects. The framework
    aligns with Green Bond Principles 2021 and Social Bond Principles 2021. Renewable energy,
    green buildings, sustainable transportation, affordable housing, education, healthcare.
    """,
    "Duke Energy": """
    Duke Energy's Sustainable Financing Framework defines eligible project categories for green
    and sustainability bonds. Renewable Energy: expenditures related to construction, development,
    expansion, production, acquisition, maintenance, transmission, research and development of
    renewable energy generation and infrastructure, including solar and wind power. Energy
    efficiency, advanced grid technology, transmission, and expanded opportunities for diverse
    suppliers. The framework aligns with the company's clean energy strategy. Solar, wind,
    renewable energy, energy efficiency, grid technology.
    """,
    "Nature Conservancy": """
    The TNC Green Bond Framework defines six eligible categories: Environmentally Sustainable
    Management of Living Natural Resources and Land Use, Terrestrial and Aquatic Biodiversity
    Conservation, Sustainable Water and Wastewater Management, Climate Change Adaptation,
    Energy Efficiency and Renewable Energy. Under Land Use, TNC intends to finance projects
    promoting soil health, sustainable fisheries, sustainable forestry, and forest restoration.
    The Biodiversity Conservation category includes projects using coral reef structures and
    mangroves for conservation of coastal ecosystems. Sustainable Water includes watershed
    restoration and support for wildlife habitats. Climate Change Adaptation includes peatland
    restoration, landscape restoration, and the use of coral reefs and mangroves to protect
    from storms and floods. Energy Efficiency includes heat pumps. Renewable Energy includes
    solar and wind. Conservation loans, ecosystem protection, habitat restoration, wetland
    management, forest conservation.
    """,
    "NY Electric & Gas": """
    Avangrid Framework for Green Financing. Eligible investments in renewable energy, energy
    efficiency, climate change adaptation, and clean transportation. The framework covers use
    of proceeds, project evaluation, management of proceeds, and reporting. Renewable energy,
    solar, wind, grid, transmission, clean energy, sustainable. Climate change adaptation.
    """,
    "Bank of America": """
    The framework covers Renewable Energy: financing of equipment for solar, wind, and geothermal
    energy. Energy Efficiency: projects reducing energy consumption including lighting retrofits,
    district heating, building insulation in residential, commercial and public properties.
    Green buildings. The framework also includes Affordable Housing and Socioeconomic Advancement.
    Renewable energy, energy efficiency, green buildings, solar, wind, geothermal.
    """,
}

def run_green_specificity() -> None:
    nlp_path = REPO_ROOT / "data" / "exports" / "tables" / "nlp_uop_classification.csv"
    if not nlp_path.exists():
        print(f"Missing {nlp_path}")
        sys.exit(1)
    nlp = pd.read_csv(nlp_path)
    df = pd.DataFrame(
        [{"Issuer": row["Issuer"], "uop_text": UOP_TEXTS.get(row["Issuer"], "")} for _, row in nlp.iterrows()]
    )
    df_scored = compute_green_specificity(df, text_col="uop_text")
    nlp["Green_Specificity"] = nlp["Issuer"].map(dict(zip(df_scored["Issuer"], df_scored["green_specificity"]))).round(4)
    nlp["Keyword_Density"] = nlp["Issuer"].map(dict(zip(df_scored["Issuer"], df_scored["keyword_density"]))).round(4)
    nlp.to_csv(nlp_path, index=False)
    print(f"Updated {nlp_path}")

def run_climate_regressions() -> None:
    repo = str(REPO_ROOT)
    TABLES = os.path.join(repo, "data", "exports", "tables")
    FIGURES = os.path.join(repo, "data", "exports", "figures")
    os.makedirs(TABLES, exist_ok=True)
    os.makedirs(FIGURES, exist_ok=True)

    feat = pd.read_csv(
        os.path.join(repo, "data", "processed", "features.csv"),
        parse_dates=["Date"],
    )
    feat = feat.set_index("Date").sort_index()

    # ═══════════════════════════════════════════════════════════════
    # PHASE 1 — Monthly Rebuild + Monthly VAR / Regressions
    # ═══════════════════════════════════════════════════════════════
    print("=" * 70)
    print("PHASE 1: Monthly frequency climate regressions")
    print("=" * 70)

    return_cols = [c for c in feat.columns if c.startswith("r_")]
    level_cols = [
        "VIX", "EPU", "CPU", "ACRI", "ACRI_loss",
        "Treasury10Y", "Treasury2Y", "IG_OAS", "HY_OAS", "BBB_OAS",
        "slope_10y_2y", "slope_10y_3m", "Breakeven10Y",
        "disaster_count",
    ]
    delta_cols = [c for c in feat.columns if c.startswith("delta_")]

    agg_rules = {}
    for c in return_cols:
        agg_rules[c] = "sum"
    for c in level_cols:
        if c in feat.columns:
            agg_rules[c] = "last"
    for c in delta_cols:
        agg_rules[c] = "sum"
    for c in [c for c in feat.columns if c.endswith("_z")]:
        agg_rules[c] = "mean"

    monthly_all = feat.resample("ME").agg(agg_rules).dropna(how="all")
    # Monthly market regressions use month-ends through 2024-12 (case studies may use full sample window).
    monthly = monthly_all.loc[:"2024-12-31"]

    # ---- 1a. Monthly OLS: climate variables on green-corp differential ----
    dep_col = "r_diff_green_corp"
    indep_monthly = ["CPU_z", "ACRI_z", "delta_10y", "VIX_z", "delta_IG_OAS"]

    df_ols = monthly[[dep_col] + indep_monthly].dropna()
    print(f"\nMonthly OLS sample: {len(df_ols)} months, {df_ols.index.min().date()} to {df_ols.index.max().date()}")

    Y = df_ols[dep_col]
    X = sm.add_constant(df_ols[indep_monthly])
    model = sm.OLS(Y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
    print(model.summary())

    monthly_ols_rows = []
    for var in ["const"] + indep_monthly:
        monthly_ols_rows.append({
            "Variable": var,
            "Coef": round(model.params[var], 6),
            "SE_HAC": round(model.bse[var], 6),
            "t_stat": round(model.tvalues[var], 3),
            "p_value": round(model.pvalues[var], 4),
        })
    monthly_ols_rows.append({
        "Variable": "N", "Coef": int(model.nobs),
        "SE_HAC": "", "t_stat": "", "p_value": "",
    })
    monthly_ols_rows.append({
        "Variable": "R2_adj", "Coef": round(model.rsquared_adj, 4),
        "SE_HAC": "", "t_stat": "", "p_value": "",
    })
    pd.DataFrame(monthly_ols_rows).to_csv(
        os.path.join(TABLES, "monthly_climate_ols.csv"), index=False
    )
    print("Saved monthly_climate_ols.csv")

    # ---- 1b. Monthly VAR with climate variables ----
    var_cols = ["r_green_corp", "r_conv_corp", "CPU_z", "ACRI_z", "VIX_z", "delta_10y"]
    df_var = monthly[var_cols].dropna()
    print(f"\nMonthly VAR sample: {len(df_var)} months")

    if len(df_var) > 20:
        var_model = VAR(df_var)
        best_lag = var_model.select_order(maxlags=6)
        opt_lag = best_lag.aic
        if opt_lag < 1:
            opt_lag = 1
        print(f"Optimal lag (AIC): {opt_lag}")

        var_fit = var_model.fit(opt_lag)

        gc_results = []
        target = "r_green_corp"
        for cause_var in ["CPU_z", "ACRI_z", "VIX_z", "delta_10y"]:
            try:
                test = var_fit.test_causality(target, [cause_var], kind="f")
                gc_results.append({
                    "Cause": cause_var,
                    "Effect": target,
                    "F_stat": round(test.test_statistic, 3),
                    "p_value": round(test.pvalue, 4),
                    "Significant_5pct": test.pvalue < 0.05,
                    "Lag": opt_lag,
                })
            except Exception as e:
                gc_results.append({
                    "Cause": cause_var, "Effect": target,
                    "F_stat": None, "p_value": None,
                    "Significant_5pct": False, "Lag": opt_lag,
                })

        target2 = "r_conv_corp"
        for cause_var in ["CPU_z", "ACRI_z"]:
            try:
                test = var_fit.test_causality(target2, [cause_var], kind="f")
                gc_results.append({
                    "Cause": cause_var, "Effect": target2,
                    "F_stat": round(test.test_statistic, 3),
                    "p_value": round(test.pvalue, 4),
                    "Significant_5pct": test.pvalue < 0.05,
                    "Lag": opt_lag,
                })
            except Exception:
                pass

        gc_df = pd.DataFrame(gc_results)
        gc_df.to_csv(os.path.join(TABLES, "monthly_granger_causality.csv"), index=False)
        print("\nMonthly Granger Causality:")
        print(gc_df.to_string(index=False))
        print("Saved monthly_granger_causality.csv")

        # Monthly IRF
        try:
            irf = var_fit.irf(periods=12)
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            fig = irf.plot(orth=True, impulse=["CPU_z", "ACRI_z"], response=["r_green_corp"])
            fig.savefig(os.path.join(FIGURES, "IRF_Monthly_Climate.pdf"), bbox_inches="tight")
            plt.close("all")
            print("Saved IRF_Monthly_Climate.pdf")
        except Exception as e:
            print(f"IRF plot skipped: {e}")

    # ---- 1c. Monthly differential regression with KRBN ----
    dep2 = "r_diff_BGRN_LQD"
    indep_krbn = ["CPU_z", "ACRI_z", "r_KRBN", "delta_10y", "VIX_z"]
    df_krbn_m = monthly[[dep2] + indep_krbn].dropna()
    print(f"\nMonthly KRBN regression sample: {len(df_krbn_m)} months")

    if len(df_krbn_m) > 10:
        Y2 = df_krbn_m[dep2]
        X2 = sm.add_constant(df_krbn_m[indep_krbn])
        m2 = sm.OLS(Y2, X2).fit(cov_type="HAC", cov_kwds={"maxlags": 3})
        print(m2.summary())
        rows2 = []
        for v in ["const"] + indep_krbn:
            rows2.append({
                "Variable": v, "Coef": round(m2.params[v], 6),
                "SE_HAC": round(m2.bse[v], 6),
                "t_stat": round(m2.tvalues[v], 3),
                "p_value": round(m2.pvalues[v], 4),
            })
        rows2.append({"Variable": "N", "Coef": int(m2.nobs), "SE_HAC": "", "t_stat": "", "p_value": ""})
        rows2.append({"Variable": "R2_adj", "Coef": round(m2.rsquared_adj, 4), "SE_HAC": "", "t_stat": "", "p_value": ""})
        pd.DataFrame(rows2).to_csv(os.path.join(TABLES, "monthly_climate_krbn_ols.csv"), index=False)
        print("Saved monthly_climate_krbn_ols.csv")

    # ---- 1d. Monthly robustness: alternative sub-samples ----
    print("\n--- Monthly robustness sub-samples ---")
    robustness_rows = []
    specs_rob = [
        ("Baseline", df_ols.index >= "2000-01-01"),
        ("Excl COVID (Mar-Jun 2020)",
         ~((df_ols.index >= "2020-03-01") & (df_ols.index <= "2020-06-30"))),
        ("Excl 2022-2023",
         ~((df_ols.index >= "2022-01-01") & (df_ols.index <= "2023-12-31"))),
        ("Post-2020 only", df_ols.index >= "2020-01-01"),
    ]
    for label, mask in specs_rob:
        sub = df_ols.loc[mask]
        if len(sub) < 20:
            continue
        m_rob = sm.OLS(sub[dep_col], sm.add_constant(sub[indep_monthly])).fit(
            cov_type="HAC", cov_kwds={"maxlags": 3}
        )
        robustness_rows.append({
            "Spec": label, "N": int(m_rob.nobs),
            "R2_adj": round(m_rob.rsquared_adj, 4),
            "ACRI_coef": round(m_rob.params["ACRI_z"], 4),
            "ACRI_p": round(m_rob.pvalues["ACRI_z"], 4),
            "CPU_coef": round(m_rob.params["CPU_z"], 4),
            "CPU_p": round(m_rob.pvalues["CPU_z"], 4),
        })
        print(f"  {label}: N={int(m_rob.nobs)}, R2adj={m_rob.rsquared_adj:.4f}, "
              f"ACRI p={m_rob.pvalues['ACRI_z']:.4f}, CPU p={m_rob.pvalues['CPU_z']:.4f}")

    pd.DataFrame(robustness_rows).to_csv(
        os.path.join(TABLES, "monthly_robustness.csv"), index=False
    )
    print("Saved monthly_robustness.csv")

    # ---- 1e. Out-of-sample expanding-window forecast ----
    print("\n--- Out-of-sample expanding-window forecast ---")
    from sklearn.metrics import mean_squared_error, mean_absolute_error

    macro_vars = ["delta_10y", "VIX_z", "delta_IG_OAS"]
    climate_vars = ["CPU_z", "ACRI_z"]
    split_n = 80

    df_oos = df_ols.copy()
    if len(df_oos) > split_n + 10:
        preds_A, preds_B, actuals = [], [], []
        for t in range(split_n, len(df_oos)):
            train = df_oos.iloc[:t]
            test_row = df_oos.iloc[t:t+1]
            y_train = train[dep_col]
            y_test = test_row[dep_col].values[0]

            X_A_train = sm.add_constant(train[macro_vars])
            m_A = sm.OLS(y_train, X_A_train).fit()
            X_A_test = sm.add_constant(test_row[macro_vars], has_constant="add")
            pred_A = m_A.predict(X_A_test).values[0]

            X_B_train = sm.add_constant(train[macro_vars + climate_vars])
            m_B = sm.OLS(y_train, X_B_train).fit()
            X_B_test = sm.add_constant(test_row[macro_vars + climate_vars], has_constant="add")
            pred_B = m_B.predict(X_B_test).values[0]

            preds_A.append(pred_A)
            preds_B.append(pred_B)
            actuals.append(y_test)

        actuals = np.array(actuals)
        preds_A = np.array(preds_A)
        preds_B = np.array(preds_B)

        oos_results = pd.DataFrame([
            {"Model": "Macro-only", "RMSE": round(np.sqrt(mean_squared_error(actuals, preds_A)), 4),
             "MAE": round(mean_absolute_error(actuals, preds_A), 4)},
            {"Model": "Macro + Climate", "RMSE": round(np.sqrt(mean_squared_error(actuals, preds_B)), 4),
             "MAE": round(mean_absolute_error(actuals, preds_B), 4)},
        ])
        oos_results.to_csv(os.path.join(TABLES, "oos_forecast_metrics.csv"), index=False)
        print(oos_results.to_string(index=False))
        print("Saved oos_forecast_metrics.csv")
    else:
        print(f"Insufficient data for OOS forecast (need >{split_n + 10}, have {len(df_oos)})")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 2 — KRBN Conditional Transition-Risk Models
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 2: Conditional KRBN interaction models")
    print("=" * 70)

    daily = feat.copy()
    # Daily KRBN stack ends 2024-12-31 to match headline market regressions; KRBN returns begin Aug 2020 in this dataset.
    daily = daily.loc[daily.index <= "2024-12-31"].copy()
    daily["PostIRA"] = (daily.index >= "2022-08-16").astype(int)
    daily["HighVIX"] = (daily["VIX_z"] > daily["VIX_z"].quantile(0.75)).astype(int)
    daily["RateHike"] = ((daily.index >= "2022-03-01") & (daily.index <= "2023-07-31")).astype(int)

    dep_d = "r_diff_BGRN_LQD"
    base_vars = ["delta_10y", "VIX_z", "r_KRBN"]

    interactions = {
        "PostIRA": "KRBN_x_PostIRA",
        "HighVIX": "KRBN_x_HighVIX",
        "RateHike": "KRBN_x_RateHike",
    }
    for dummy, ixn_name in interactions.items():
        daily[ixn_name] = daily["r_KRBN"] * daily[dummy]

    specs = {
        "Base": base_vars,
        "PostIRA_Interaction": base_vars + ["PostIRA", "KRBN_x_PostIRA"],
        "HighVIX_Interaction": base_vars + ["HighVIX", "KRBN_x_HighVIX"],
        "RateHike_Interaction": base_vars + ["RateHike", "KRBN_x_RateHike"],
        "Full_Interactions": base_vars + ["PostIRA", "KRBN_x_PostIRA",
                                           "HighVIX", "KRBN_x_HighVIX",
                                           "RateHike", "KRBN_x_RateHike"],
    }

    all_cond_rows = []
    for spec_name, rhs in specs.items():
        df_spec = daily[[dep_d] + rhs].dropna()
        if len(df_spec) < 30:
            continue
        Y = df_spec[dep_d]
        X = sm.add_constant(df_spec[rhs])
        m = sm.OLS(Y, X).fit(cov_type="HC1")
        for v in ["const"] + rhs:
            all_cond_rows.append({
                "Spec": spec_name, "Variable": v,
                "Coef": round(m.params[v], 6),
                "SE_robust": round(m.bse[v], 6),
                "t_stat": round(m.tvalues[v], 3),
                "p_value": round(m.pvalues[v], 4),
            })
        all_cond_rows.append({
            "Spec": spec_name, "Variable": "N",
            "Coef": int(m.nobs), "SE_robust": "", "t_stat": "", "p_value": "",
        })
        all_cond_rows.append({
            "Spec": spec_name, "Variable": "R2_adj",
            "Coef": round(m.rsquared_adj, 4), "SE_robust": "", "t_stat": "", "p_value": "",
        })
        print(f"\n--- {spec_name} (N={int(m.nobs)}, R2adj={m.rsquared_adj:.4f}) ---")
        for v in rhs:
            sig = "***" if m.pvalues[v] < 0.01 else ("**" if m.pvalues[v] < 0.05 else ("*" if m.pvalues[v] < 0.1 else ""))
            print(f"  {v:25s} coef={m.params[v]:+.6f}  t={m.tvalues[v]:+.3f}  p={m.pvalues[v]:.4f} {sig}")

    pd.DataFrame(all_cond_rows).to_csv(
        os.path.join(TABLES, "krbn_conditional_interactions.csv"), index=False
    )
    print("\nSaved krbn_conditional_interactions.csv")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 3 — FEMA NRI + NOAA Disaster Data
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 3: External disaster data — FEMA NRI + NOAA events")
    print("=" * 70)

    # Real issuer location data (publicly verifiable HQ locations)
    issuer_locations = {
        "Apple Inc.": {"state": "CA", "county": "Santa Clara", "fips": "06085",
                       "exposure": "low", "notes": "Cupertino, CA — low hurricane/flood risk"},
        "JPMorgan Chase": {"state": "NY", "county": "New York", "fips": "36061",
                           "exposure": "moderate", "notes": "New York City — moderate flood risk"},
        "Duke Energy": {"state": "NC", "county": "Mecklenburg", "fips": "37119",
                        "exposure": "high", "notes": "Charlotte, NC — high hurricane exposure, SE utility"},
        "Nature Conservancy": {"state": "VA", "county": "Arlington", "fips": "51013",
                              "exposure": "moderate", "notes": "Arlington, VA — moderate flood risk"},
        "NY Electric & Gas": {"state": "NY", "county": "Tompkins", "fips": "36109",
                              "exposure": "high", "notes": "Ithaca, NY — high winter storm/flood, NE utility"},
        "Bank of America": {"state": "NC", "county": "Mecklenburg", "fips": "37119",
                            "exposure": "moderate", "notes": "Charlotte, NC — moderate hurricane exposure"},
    }

    loc_df = pd.DataFrame([
        {"Issuer": k, "State": v["state"], "County": v["county"],
         "FIPS": v["fips"], "Exposure_Category": v["exposure"],
         "Is_Utility": k in ["Duke Energy", "NY Electric & Gas"],
         "Notes": v["notes"]}
        for k, v in issuer_locations.items()
    ])
    loc_df.to_csv(os.path.join(TABLES, "issuer_location_exposure.csv"), index=False)
    print("Saved issuer_location_exposure.csv")
    print(loc_df.to_string(index=False))

    # Real NOAA billion-dollar disaster events affecting our utility regions
    # Source: https://www.ncei.noaa.gov/access/billions/
    noaa_events = pd.DataFrame([
        {"Event": "Hurricane Dorian", "Date": "2019-09-05", "Type": "Hurricane",
         "Affected_States": "NC,SC,VA", "Cost_Billion": 1.6,
         "Source": "NOAA NCEI Billion-Dollar Disasters"},
        {"Event": "Hurricane Isaias", "Date": "2020-08-03", "Type": "Hurricane",
         "Affected_States": "NC,SC,NY,NJ", "Cost_Billion": 4.8,
         "Source": "NOAA NCEI Billion-Dollar Disasters"},
        {"Event": "Winter Storm Uri", "Date": "2021-02-15", "Type": "Winter Storm",
         "Affected_States": "TX,NC,SC,NY", "Cost_Billion": 24.0,
         "Source": "NOAA NCEI Billion-Dollar Disasters"},
        {"Event": "Hurricane Ida", "Date": "2021-08-29", "Type": "Hurricane",
         "Affected_States": "LA,MS,NJ,NY,PA", "Cost_Billion": 80.3,
         "Source": "NOAA NCEI Billion-Dollar Disasters"},
        {"Event": "Hurricane Ian", "Date": "2022-09-28", "Type": "Hurricane",
         "Affected_States": "FL,SC,NC", "Cost_Billion": 116.0,
         "Source": "NOAA NCEI Billion-Dollar Disasters"},
        {"Event": "Winter Storm Elliott", "Date": "2022-12-22", "Type": "Winter Storm",
         "Affected_States": "NY,NC,SC,VA,PA", "Cost_Billion": 6.0,
         "Source": "NOAA NCEI Billion-Dollar Disasters"},
        {"Event": "Hurricane Helene", "Date": "2024-09-27", "Type": "Hurricane",
         "Affected_States": "FL,GA,NC,SC,TN,VA", "Cost_Billion": 78.7,
         "Source": "NOAA NCEI Billion-Dollar Disasters"},
    ])
    noaa_events["Date"] = pd.to_datetime(noaa_events["Date"])
    noaa_events.to_csv(os.path.join(TABLES, "noaa_disaster_events.csv"), index=False)
    print("\nSaved noaa_disaster_events.csv")
    print(noaa_events[["Event", "Date", "Type", "Cost_Billion"]].to_string(index=False))

    # Tag which events affect which bonds
    bond_state_map = {
        "DukeEnergyGB": "NC",
        "NYElecGasGB": "NY",
        "AppleGB": "CA",
        "JPMGB": "NY",
        "NatureConservancyGB": "VA",
        "BACGB": "NC",
    }

    exposure_matrix = []
    for _, evt in noaa_events.iterrows():
        affected = evt["Affected_States"].split(",")
        for bond, st in bond_state_map.items():
            exposure_matrix.append({
                "Event": evt["Event"], "Date": evt["Date"],
                "Bond": bond, "State": st,
                "Directly_Affected": st in affected,
            })
    exp_df = pd.DataFrame(exposure_matrix)
    exp_df.to_csv(os.path.join(TABLES, "bond_disaster_exposure.csv"), index=False)
    print("\nSaved bond_disaster_exposure.csv")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 4 — Bond-Level Utility Physical-Event Windows
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 4: Bond-level physical event windows")
    print("=" * 70)

    bond_cols_map = {
        "AppleGB": "Apple Inc.", "JPMGB": "JPMorgan Chase",
        "DukeEnergyGB": "Duke Energy", "NatureConservancyGB": "Nature Conservancy",
        "NYElecGasGB": "NY Electric & Gas", "BACGB": "Bank of America",
    }
    utility_bonds = ["DukeEnergyGB", "NYElecGasGB"]
    non_utility_bonds = ["AppleGB", "JPMGB", "NatureConservancyGB", "BACGB"]

    # Compute monthly returns for individual bonds
    bond_monthly = {}
    for bond_col in bond_cols_map:
        if bond_col in feat.columns:
            px = feat[bond_col].dropna()
            if len(px) > 1:
                ret = np.log(px / px.shift(1)).dropna()
                bond_monthly[bond_col] = ret.resample("ME").sum().dropna()

    # Benchmark monthly return
    if "r_I31572US" in monthly_all.columns:
        bench_monthly = monthly_all["r_I31572US"].dropna()
    elif "r_LQD" in monthly_all.columns:
        bench_monthly = monthly_all["r_LQD"].dropna()
    else:
        bench_monthly = pd.Series(dtype=float)

    # Event window analysis: for each NOAA event, compare ±1 month returns
    event_window_results = []
    for _, evt in noaa_events.iterrows():
        evt_date = evt["Date"]
        evt_month = evt_date.to_period("M")

        for bond_col, issuer in bond_cols_map.items():
            if bond_col not in bond_monthly:
                continue
            bret = bond_monthly[bond_col]
            is_util = bond_col in utility_bonds
            affected = bond_state_map.get(bond_col, "") in evt["Affected_States"].split(",")

            evt_months = pd.period_range(evt_month - 1, evt_month + 1, freq="M")

            rets_in_window = []
            bench_in_window = []
            for m in evt_months:
                ts = m.to_timestamp("M")
                if ts in bret.index:
                    rets_in_window.append(bret.loc[ts])
                if ts in bench_monthly.index:
                    bench_in_window.append(bench_monthly.loc[ts])

            if len(rets_in_window) > 0:
                car = sum(rets_in_window)
                bench_car = sum(bench_in_window) if bench_in_window else np.nan
                excess = car - bench_car if not np.isnan(bench_car) else np.nan
            else:
                car = bench_car = excess = np.nan

            event_window_results.append({
                "Event": evt["Event"], "Date": str(evt_date.date()),
                "Type": evt["Type"], "Cost_Bn": evt["Cost_Billion"],
                "Bond": issuer, "Is_Utility": is_util,
                "Directly_Affected": affected,
                "CAR_3mo_pct": round(car * 100, 3) if not np.isnan(car) else None,
                "Bench_CAR_pct": round(bench_car * 100, 3) if not np.isnan(bench_car) else None,
                "Excess_CAR_pct": round(excess * 100, 3) if not np.isnan(excess) else None,
            })

    evt_df = pd.DataFrame(event_window_results)
    evt_df.to_csv(os.path.join(TABLES, "bond_physical_event_windows.csv"), index=False)
    print("Saved bond_physical_event_windows.csv")

    # Aggregate: utility vs non-utility mean excess returns around disasters
    if not evt_df.empty and evt_df["Excess_CAR_pct"].notna().any():
        affected_only = evt_df[evt_df["Directly_Affected"] == True].copy()
        if not affected_only.empty and affected_only["Excess_CAR_pct"].notna().any():
            util_excess = affected_only[affected_only["Is_Utility"] == True]["Excess_CAR_pct"].dropna()
            non_util_excess = affected_only[affected_only["Is_Utility"] == False]["Excess_CAR_pct"].dropna()

            print(f"\n--- Affected-Event Excess Returns ---")
            print(f"Utilities (N={len(util_excess)}): mean={util_excess.mean():.3f}%, std={util_excess.std():.3f}")
            if len(non_util_excess) > 0:
                print(f"Non-Utilities (N={len(non_util_excess)}): mean={non_util_excess.mean():.3f}%, std={non_util_excess.std():.3f}")

            if len(util_excess) >= 2 and len(non_util_excess) >= 2:
                t, p = stats.ttest_ind(util_excess, non_util_excess, equal_var=False)
                mw_stat, mw_p = stats.mannwhitneyu(util_excess, non_util_excess, alternative="two-sided")
                print(f"Welch t-test: t={t:.3f}, p={p:.4f}")
                print(f"Mann-Whitney U: stat={mw_stat:.1f}, p={mw_p:.4f}")

                summary_rows = [
                    {"Group": "Utilities_Affected", "N": len(util_excess),
                     "Mean_Excess_CAR_pct": round(util_excess.mean(), 3),
                     "Std": round(util_excess.std(), 3)},
                    {"Group": "NonUtilities_Affected", "N": len(non_util_excess),
                     "Mean_Excess_CAR_pct": round(non_util_excess.mean(), 3),
                     "Std": round(non_util_excess.std(), 3)},
                    {"Group": "Difference", "N": "",
                     "Mean_Excess_CAR_pct": round(util_excess.mean() - non_util_excess.mean(), 3),
                     "Std": ""},
                    {"Group": "Welch_t", "N": "", "Mean_Excess_CAR_pct": round(t, 3), "Std": round(p, 4)},
                    {"Group": "MannWhitney_U", "N": "", "Mean_Excess_CAR_pct": round(mw_stat, 1), "Std": round(mw_p, 4)},
                ]
                pd.DataFrame(summary_rows).to_csv(
                    os.path.join(TABLES, "utility_vs_nonutility_disaster.csv"), index=False
                )
                print("Saved utility_vs_nonutility_disaster.csv")

        # All events (not just affected)
        all_util = evt_df[evt_df["Is_Utility"] == True]["Excess_CAR_pct"].dropna()
        all_nonut = evt_df[evt_df["Is_Utility"] == False]["Excess_CAR_pct"].dropna()
        print(f"\n--- All-Event Excess Returns ---")
        print(f"Utilities (N={len(all_util)}): mean={all_util.mean():.3f}%")
        print(f"Non-Utilities (N={len(all_nonut)}): mean={all_nonut.mean():.3f}%")

    # ═══════════════════════════════════════════════════════════════
    # PHASE 5 — Utility Regime Interaction Regressions
    # ═══════════════════════════════════════════════════════════════
    print("\n" + "=" * 70)
    print("PHASE 5: Utility regime interaction regressions")
    print("=" * 70)

    # Build a bond-level panel from monthly data
    panel_rows = []
    for bond_col, issuer in bond_cols_map.items():
        if bond_col not in bond_monthly:
            continue
        is_util = 1 if bond_col in utility_bonds else 0
        bret = bond_monthly[bond_col]
        for dt, ret in bret.items():
            bench_val = bench_monthly.get(dt, np.nan)
            spread = ret - bench_val if not np.isnan(bench_val) else np.nan
            # Get monthly macro from the full monthly dataset; bond case studies
            # keep their 2025 observations even though headline market regressions
            # stop at 2024.
            if dt in monthly_all.index:
                row_m = monthly_all.loc[dt]
                panel_rows.append({
                    "Date": dt, "Bond": issuer, "Is_Utility": is_util,
                    "Return": ret, "Spread_vs_IG": spread,
                    "delta_10y": row_m.get("delta_10y", np.nan),
                    "VIX_z": row_m.get("VIX_z", np.nan),
                    "CPU_z": row_m.get("CPU_z", np.nan),
                    "ACRI_z": row_m.get("ACRI_z", np.nan),
                    "r_KRBN": row_m.get("r_KRBN", np.nan),
                })

    panel = pd.DataFrame(panel_rows)
    if not panel.empty:
        panel["Util_x_delta10y"] = panel["Is_Utility"] * panel["delta_10y"]
        panel["Util_x_VIX"] = panel["Is_Utility"] * panel["VIX_z"]
        panel["Util_x_CPU"] = panel["Is_Utility"] * panel["CPU_z"]
        panel["Util_x_ACRI"] = panel["Is_Utility"] * panel["ACRI_z"]

        # Climate event window dummy
        event_months = set()
        for _, evt in noaa_events.iterrows():
            ep = evt["Date"].to_period("M")
            for offset in [-1, 0, 1]:
                event_months.add(ep + offset)
        panel["Climate_Event_Window"] = panel["Date"].apply(
            lambda d: 1 if d.to_period("M") in event_months else 0
        )
        panel["Util_x_ClimateEvt"] = panel["Is_Utility"] * panel["Climate_Event_Window"]

        dep = "Spread_vs_IG"
        regime_specs = {
            "Rate_Sensitivity": {
                "rhs": ["delta_10y", "VIX_z", "Is_Utility", "Util_x_delta10y"],
            },
            "Volatility_Sensitivity": {
                "rhs": ["delta_10y", "VIX_z", "Is_Utility", "Util_x_VIX"],
            },
            "Climate_Policy": {
                "rhs": ["delta_10y", "VIX_z", "CPU_z", "Is_Utility", "Util_x_CPU"],
            },
            "Climate_Event": {
                "rhs": ["delta_10y", "VIX_z", "Climate_Event_Window", "Is_Utility", "Util_x_ClimateEvt"],
            },
            "Full_Utility_Interaction": {
                "rhs": ["delta_10y", "VIX_z", "CPU_z", "Climate_Event_Window",
                         "Is_Utility", "Util_x_delta10y", "Util_x_VIX",
                         "Util_x_CPU", "Util_x_ClimateEvt"],
            },
        }

        regime_rows = []
        for sname, sdef in regime_specs.items():
            rhs = sdef["rhs"]
            df_r = panel[[dep] + rhs].dropna()
            if len(df_r) < 15:
                continue
            Y = df_r[dep]
            X = sm.add_constant(df_r[rhs])
            m = sm.OLS(Y, X).fit(cov_type="HC1")
            print(f"\n--- {sname} (N={int(m.nobs)}, R2adj={m.rsquared_adj:.4f}) ---")
            for v in rhs:
                sig = "***" if m.pvalues[v] < 0.01 else ("**" if m.pvalues[v] < 0.05 else ("*" if m.pvalues[v] < 0.1 else ""))
                print(f"  {v:30s} coef={m.params[v]:+.6f}  t={m.tvalues[v]:+.3f}  p={m.pvalues[v]:.4f} {sig}")
                regime_rows.append({
                    "Spec": sname, "Variable": v,
                    "Coef": round(m.params[v], 6),
                    "SE_robust": round(m.bse[v], 6),
                    "t_stat": round(m.tvalues[v], 3),
                    "p_value": round(m.pvalues[v], 4),
                })
            regime_rows.append({"Spec": sname, "Variable": "N", "Coef": int(m.nobs), "SE_robust": "", "t_stat": "", "p_value": ""})
            regime_rows.append({"Spec": sname, "Variable": "R2_adj", "Coef": round(m.rsquared_adj, 4), "SE_robust": "", "t_stat": "", "p_value": ""})

        pd.DataFrame(regime_rows).to_csv(
            os.path.join(TABLES, "utility_regime_interactions.csv"), index=False
        )
        print("\nSaved utility_regime_interactions.csv")

    # ── Cross-sectional green specificity during events ──
    print("\n" + "=" * 70)
    print("PHASE 5b: Green specificity × event interaction")
    print("=" * 70)

    nlp = pd.read_csv(os.path.join(TABLES, "nlp_uop_classification.csv"))
    spec_map = dict(zip(nlp["Issuer"], nlp["Green_Specificity"]))

    if not panel.empty:
        panel["Green_Specificity"] = panel["Bond"].map(spec_map)
        panel["GS_x_ClimateEvt"] = panel["Green_Specificity"] * panel["Climate_Event_Window"]
        panel["GS_x_VIX"] = panel["Green_Specificity"] * panel["VIX_z"]

        gs_rhs = ["delta_10y", "VIX_z", "Climate_Event_Window",
                  "Green_Specificity", "GS_x_ClimateEvt", "GS_x_VIX"]
        df_gs = panel[["Spread_vs_IG"] + gs_rhs].dropna()
        if len(df_gs) > 15:
            Y = df_gs["Spread_vs_IG"]
            X = sm.add_constant(df_gs[gs_rhs])
            m_gs = sm.OLS(Y, X).fit(cov_type="HC1")
            print(f"N={int(m_gs.nobs)}, R2adj={m_gs.rsquared_adj:.4f}")
            gs_rows = []
            for v in gs_rhs:
                sig = "***" if m_gs.pvalues[v] < 0.01 else ("**" if m_gs.pvalues[v] < 0.05 else ("*" if m_gs.pvalues[v] < 0.1 else ""))
                print(f"  {v:30s} coef={m_gs.params[v]:+.6f}  t={m_gs.tvalues[v]:+.3f}  p={m_gs.pvalues[v]:.4f} {sig}")
                gs_rows.append({
                    "Variable": v, "Coef": round(m_gs.params[v], 6),
                    "SE_robust": round(m_gs.bse[v], 6),
                    "t_stat": round(m_gs.tvalues[v], 3),
                    "p_value": round(m_gs.pvalues[v], 4),
                })
            gs_rows.append({"Variable": "N", "Coef": int(m_gs.nobs), "SE_robust": "", "t_stat": "", "p_value": ""})
            gs_rows.append({"Variable": "R2_adj", "Coef": round(m_gs.rsquared_adj, 4), "SE_robust": "", "t_stat": "", "p_value": ""})
            pd.DataFrame(gs_rows).to_csv(
                os.path.join(TABLES, "green_specificity_event_interactions.csv"), index=False
            )
            print("Saved green_specificity_event_interactions.csv")

    print("\n" + "=" * 70)
    print("ALL PHASES COMPLETE — Tables saved to data/exports/tables/")
    print("=" * 70)


def exec_yearlies() -> None:
    """
    Yearly climate regressions — same spec as monthly but at annual frequency.

    Regresses green-conventional return differential on CPU, ACRI, delta_10y, VIX, delta_IG_OAS.
    With ~10-12 yearly observations, results are exploratory; HAC SEs with maxlags=1.

    Outputs: data/exports/tables/yearly_climate_ols.csv
    """
    rr = str(REPO_ROOT)
    TABLES = os.path.join(rr, "data", "exports", "tables")
    os.makedirs(TABLES, exist_ok=True)

    feat = pd.read_csv(
        os.path.join(rr, "data", "processed", "features.csv"),
        parse_dates=["Date"],
    )
    feat = feat.set_index("Date").sort_index()

    return_cols = [c for c in feat.columns if c.startswith("r_")]
    level_cols = [
        "VIX", "EPU", "CPU", "ACRI", "ACRI_loss",
        "Treasury10Y", "Treasury2Y", "IG_OAS", "HY_OAS", "BBB_OAS",
        "slope_10y_2y", "slope_10y_3m", "Breakeven10Y",
        "disaster_count",
    ]
    delta_cols = [c for c in feat.columns if c.startswith("delta_")]

    agg_rules = {}
    for c in return_cols:
        agg_rules[c] = "sum"
    for c in level_cols:
        if c in feat.columns:
            agg_rules[c] = "last"
    for c in delta_cols:
        agg_rules[c] = "sum"
    for c in [c for c in feat.columns if c.endswith("_z")]:
        agg_rules[c] = "mean"

# Align yearly rows with monthly window (annual dates through end of 2024).
    yearly = feat.resample("YE").agg(agg_rules).dropna(how="all").loc[:"2024-12-31"]

    dep_col = "r_diff_green_corp"
    indep = ["CPU_z", "ACRI_z", "delta_10y", "VIX_z", "delta_IG_OAS"]

    df_ols = yearly[[dep_col] + indep].dropna()
    print(f"\nYearly OLS sample: {len(df_ols)} years, {df_ols.index.min().date()} to {df_ols.index.max().date()}")

    if len(df_ols) < 5:
        print("Too few yearly observations. Skipping.")
    else:
        Y = df_ols[dep_col]
        X = sm.add_constant(df_ols[indep])
        # With few obs, use HAC maxlags=1 or simple OLS
        try:
            model = sm.OLS(Y, X).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
        except Exception:
            model = sm.OLS(Y, X).fit()

        print("\n" + "=" * 60)
        print("YEARLY CLIMATE REGRESSION: r_diff_green_corp")
        print("=" * 60)
        print(model.summary())

        rows = []
        for var in ["const"] + indep:
            rows.append({
                "Variable": var,
                "Coef": round(model.params[var], 6),
                "SE": round(model.bse[var], 6),
                "t_stat": round(model.tvalues[var], 3),
                "p_value": round(model.pvalues[var], 4),
            })
        rows.append({"Variable": "N", "Coef": int(model.nobs), "SE": "", "t_stat": "", "p_value": ""})
        rows.append({"Variable": "R2_adj", "Coef": round(model.rsquared_adj, 4), "SE": "", "t_stat": "", "p_value": ""})

        out_path = os.path.join(TABLES, "yearly_climate_ols.csv")
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"\nSaved {out_path}")

        # Yearly KRBN regression
        dep2 = "r_diff_BGRN_LQD"
        indep_krbn = ["CPU_z", "ACRI_z", "r_KRBN", "delta_10y", "VIX_z"]
        avail = [c for c in indep_krbn if c in yearly.columns]
        df_krbn = yearly[[dep2] + avail].dropna() if dep2 in yearly.columns and avail else None

        if df_krbn is not None and len(df_krbn) >= 5:
            Y2 = df_krbn[dep2]
            X2 = sm.add_constant(df_krbn[avail])
            try:
                m2 = sm.OLS(Y2, X2).fit(cov_type="HAC", cov_kwds={"maxlags": 1})
            except Exception:
                m2 = sm.OLS(Y2, X2).fit()
            print("\n" + "=" * 60)
            print("YEARLY KRBN REGRESSION: r_diff_BGRN_LQD")
            print("=" * 60)
            print(m2.summary())
            rows2 = []
            for v in ["const"] + avail:
                rows2.append({
                    "Variable": v, "Coef": round(m2.params[v], 6),
                    "SE": round(m2.bse[v], 6),
                    "t_stat": round(m2.tvalues[v], 3),
                    "p_value": round(m2.pvalues[v], 4),
                })
            rows2.append({"Variable": "N", "Coef": int(m2.nobs), "SE": "", "t_stat": "", "p_value": ""})
            rows2.append({"Variable": "R2_adj", "Coef": round(m2.rsquared_adj, 4), "SE": "", "t_stat": "", "p_value": ""})
            pd.DataFrame(rows2).to_csv(os.path.join(TABLES, "yearly_climate_krbn_ols.csv"), index=False)
            print("Saved yearly_climate_krbn_ols.csv")



_var_log = logging.getLogger("var_appendix")

def build_var_data(features: pd.DataFrame, system_cols: list, label: str) -> pd.DataFrame:
    available = [c for c in system_cols if c in features.columns]
    missing = [c for c in system_cols if c not in features.columns]
    if missing:
        _var_log.warning(f"System '{label}' missing columns: {missing}")
    df = features[available].dropna()
    _var_log.info(f"System '{label}': {len(df)} obs, {len(available)} vars: {available}")
    return df


def select_lag(data: pd.DataFrame, max_lag: int = 8) -> pd.DataFrame:
    from statsmodels.tsa.api import VAR
    model = VAR(data)
    results = []
    for p in range(1, max_lag + 1):
        try:
            res = model.fit(p)
            results.append({
                "lag": p,
                "AIC": res.aic,
                "BIC": res.bic,
                "HQIC": res.hqic,
                "LogL": res.llf,
            })
        except Exception:
            continue
    lag_df = pd.DataFrame(results)
    if not lag_df.empty:
        for ic in ["AIC", "BIC", "HQIC"]:
            best = lag_df.loc[lag_df[ic].idxmin(), "lag"]
            _var_log.info(f"  Best lag ({ic}): {int(best)}")
    return lag_df


def granger_causality(data: pd.DataFrame, lag: int) -> pd.DataFrame:
    from statsmodels.tsa.api import VAR
    model = VAR(data)
    res = model.fit(lag)
    cols = data.columns.tolist()
    records = []
    for caused in cols:
        for causing in cols:
            if caused == causing:
                continue
            try:
                test = res.test_causality(caused, causing, kind="f")
                records.append({
                    "caused": caused,
                    "causing": causing,
                    "F_stat": round(test.test_statistic, 3),
                    "p_value": round(test.pvalue, 4),
                    "significant_5pct": test.pvalue < 0.05,
                })
            except Exception:
                pass
    return pd.DataFrame(records)


def compute_irf(data: pd.DataFrame, lag: int, steps: int = 10) -> dict:
    from statsmodels.tsa.api import VAR
    model = VAR(data)
    res = model.fit(lag)
    irf = res.irf(steps)
    return {"irf_obj": irf, "var_result": res}


def plot_irf(irf_obj, cols: list, label: str, fig_dir: Path):
    fig = irf_obj.plot(orth=True)
    fig.set_size_inches(14, 10)
    fig.suptitle(f"Orthogonalized IRFs — {label}", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / f"IRF_{label}.pdf", bbox_inches="tight")
    plt.close(fig)
    _var_log.info(f"  Saved IRF_{label}.pdf")


def plot_fevd(var_result, cols: list, label: str, fig_dir: Path):
    fevd = var_result.fevd(10)
    fig = fevd.plot()
    fig.set_size_inches(14, 10)
    fig.suptitle(f"FEVD — {label}", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(fig_dir / f"FEVD_{label}.pdf", bbox_inches="tight")
    plt.close(fig)
    _var_log.info(f"  Saved FEVD_{label}.pdf")


def run_system(features: pd.DataFrame, system_cols: list, label: str,
               tables_dir: Path, fig_dir: Path) -> dict:
    _var_log.info(f"\n{'='*60}")
    _var_log.info(f"VAR System: {label}")
    _var_log.info(f"{'='*60}")

    data = build_var_data(features, system_cols, label)
    if len(data) < 60:
        _var_log.warning(f"Insufficient data for {label}: {len(data)} obs")
        return {}

    lag_df = select_lag(data, max_lag=8)
    lag_df.to_csv(tables_dir / f"var_lag_selection_{label}.csv", index=False)

    best_lag = int(lag_df.loc[lag_df["AIC"].idxmin(), "lag"]) if not lag_df.empty else 2

    gc_df = granger_causality(data, best_lag)
    gc_df.to_csv(tables_dir / f"granger_causality_{label}.csv", index=False)
    sig = gc_df[gc_df["significant_5pct"]]
    _var_log.info(f"  Granger significant pairs: {len(sig)}/{len(gc_df)}")

    irf_result = compute_irf(data, best_lag, steps=10)
    irf_obj = irf_result["irf_obj"]

    try:
        plot_irf(irf_obj, data.columns.tolist(), label, fig_dir)
    except Exception as e:
        _var_log.warning(f"  IRF plot failed: {e}")

    try:
        plot_fevd(irf_result["var_result"], data.columns.tolist(), label, fig_dir)
    except Exception as e:
        _var_log.warning(f"  FEVD plot failed: {e}")

    return {
        "label": label,
        "n_obs": len(data),
        "best_lag": best_lag,
        "granger_df": gc_df,
        "lag_df": lag_df,
    }


def run_var_appendix() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    cfg = load_config()

    processed_dir = REPO_ROOT / cfg["paths"]["processed"]
    tables_dir = REPO_ROOT / cfg["paths"]["tables"]
    fig_dir = REPO_ROOT / cfg["paths"]["figures"]
    tables_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)

    features = pd.read_csv(processed_dir / "features.csv", index_col="Date", parse_dates=True)
    _var_log.info(f"Loaded features: {features.shape}")

    # System 3: green-conventional differential focus.
    system3_cols = [
        "r_diff_green_corp", "r_global_green", "r_climate_transition",
        "EPU_z", "CPU_z", "VIX_z", "delta_IG_OAS", "delta_10y",
    ]
    run_system(features, system3_cols, "System3_Differential", tables_dir, fig_dir)

    # System 4: rate-hedged differential.
    system4_cols = [
        "r_diff_hedged", "r_SPGBI", "delta_IG_OAS",
        "VIX_z", "CPU_z", "EPU_z",
    ]
    run_system(features, system4_cols, "System4_HedgedDiff", tables_dir, fig_dir)

    # Combine Granger tables for Systems 3 and 4.
    all_gc = []
    for label in ["System3_Differential", "System4_HedgedDiff"]:
        p = tables_dir / f"granger_causality_{label}.csv"
        if p.exists():
            gc = pd.read_csv(p)
            gc["system"] = label
            all_gc.append(gc)
    if all_gc:
        combined_gc = pd.concat(all_gc, ignore_index=True)
        combined_gc.to_csv(tables_dir / "granger_causality.csv", index=False)

    _var_log.info("\nvar appendix complete.")

BOND_MAP = {
    "AppleGB": "Apple Green Bond",
    "DukeEnergyGB": "Duke Energy Green Bond",
    "NatureConservancyGB": "Nature Conservancy GB",
    "JPMGB": "JPMorgan Green Bond (6.07% 10/27)",
    "NYElecGasGB": "NY Electric & Gas GB",
    "BACGB": "Bank of America Green Bond (EUR)",
}

def run_case_studies() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    tables_dir = REPO_ROOT / "data" / "exports" / "tables"
    fig_dir = REPO_ROOT / "data" / "exports" / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    features = pd.read_csv(
        REPO_ROOT / "data" / "processed" / "features.csv",
        index_col="Date", parse_dates=True,
    )
    bond_cols = [c for c in BOND_MAP if c in features.columns]
    if not bond_cols:
        logger.warning("No individual bond data found in features.csv")
        return
    logger.info(f"Found {len(bond_cols)} individual bonds: {bond_cols}")
    # Compute monthly returns for bonds (prices are forward-filled monthly)
    records = []
    bond_returns = pd.DataFrame(index=features.index)
    for col in bond_cols:
        prices = features[col].dropna()
        if len(prices) < 3:
            continue
        # Resample to month-end, compute log returns
        monthly = prices.resample("ME").last().dropna()
        ret = np.log(monthly / monthly.shift(1)) * 100
        ret = ret.dropna()
        bond_returns[f"r_{col}"] = ret.reindex(features.index)
        label = BOND_MAP.get(col, col)
        records.append({
            "Bond": label,
            "Ticker": col,
            "Start": str(prices.index.min().date()),
            "End": str(prices.index.max().date()),
            "N_months": len(ret),
            "Mean_monthly_ret": round(ret.mean(), 3),
            "Std_monthly_ret": round(ret.std(), 3),
            "Min_price": round(prices.min(), 2),
            "Max_price": round(prices.max(), 2),
            "Current_price": round(prices.iloc[-1], 2),
        })
    summary = pd.DataFrame(records)
    summary.to_csv(tables_dir / "bond_case_study_summary.csv", index=False)
    logger.info(f"Bond case study summary:\n{summary.to_string()}")

    # Plot: individual bond prices vs BGRN (normalized to 100)
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))
    ax = axes[0]
    for col in bond_cols:
        prices = features[col].dropna()
        if len(prices) < 3:
            continue
        normalized = prices / prices.iloc[0] * 100
        label = BOND_MAP.get(col, col)
        ax.plot(normalized.index, normalized.values, label=label, linewidth=1.0)
    # Add BGRN benchmark
    if "BGRN" in features.columns:
        bgrn = features["BGRN"].dropna()
        # Align to common start
        common_start = max(features[c].dropna().index.min() for c in bond_cols if features[c].notna().sum() > 3)
        bgrn_aligned = bgrn[bgrn.index >= common_start]
        if len(bgrn_aligned) > 0:
            norm_bgrn = bgrn_aligned / bgrn_aligned.iloc[0] * 100
            ax.plot(norm_bgrn.index, norm_bgrn.values, label="BGRN ETF",
                    linewidth=1.5, linestyle="--", color="black")
    ax.set_title("A. Individual Green Bond Prices (Normalized to 100)")
    ax.set_ylabel("Price (rebased)")
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    # Panel B: Monthly return comparison
    ax = axes[1]
    ret_cols_plot = [f"r_{c}" for c in bond_cols if f"r_{c}" in bond_returns.columns]
    for rc in ret_cols_plot:
        raw_col = rc.replace("r_", "")
        label = BOND_MAP.get(raw_col, raw_col)
        s = bond_returns[rc].dropna()
        ax.bar(s.index, s.values, alpha=0.4, label=label, width=20)
    ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_title("B. Monthly Log Returns of Individual Green Bonds")
    ax.set_ylabel("Monthly Return (%)")
    ax.legend(fontsize=7, loc="best")
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))

    plt.tight_layout()
    fig.savefig(fig_dir / "bond_case_studies.pdf", bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved bond_case_studies.pdf")


def setup_logging() -> logging.Logger:
    log_dir = CODE_DIR / "logs"
    log_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log = logging.getLogger("run")
    log.setLevel(logging.INFO)
    log.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)-8s | %(message)s")
    fh = logging.FileHandler(log_dir / f"run_{ts}.log")
    fh.setFormatter(fmt)
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    log.addHandler(fh)
    log.addHandler(ch)
    return log


def run_cmd(cmd: list[str], log: logging.Logger, cwd: Path | None = None) -> None:
    log.info("Running: %s", " ".join(cmd))
    t0 = time.time()
    r = subprocess.run(cmd, cwd=cwd or CODE_DIR, capture_output=True, text=True)
    elapsed = time.time() - t0
    if r.stdout:
        log.info(r.stdout.rstrip())
    if r.returncode != 0:
        if r.stderr:
            log.error(r.stderr.rstrip())
        raise SystemExit(f"FAILED ({elapsed:.1f}s): {' '.join(cmd)}")
    log.info("Done (%.1fs): %s", elapsed, " ".join(cmd))


STAGES = ("features", "nlp", "regressions", "garch", "var", "case_studies", "full")


def main() -> None:
    parser = argparse.ArgumentParser(description="Replication pipeline")
    parser.add_argument("--stage", choices=STAGES, default="full")
    args = parser.parse_args()
    log = setup_logging()
    log.info("Pipeline started")

    order = (
        ["features", "nlp", "regressions", "garch", "var", "case_studies"]
        if args.stage == "full"
        else [args.stage]
    )
    t0 = time.time()
    for stage in order:
        log.info("=" * 60)
        log.info("STAGE: %s", stage)
        log.info("=" * 60)
        if stage == "features":
            build_feature_matrix()
        elif stage == "nlp":
            run_green_specificity()
        elif stage == "regressions":
            run_climate_regressions()
            exec_yearlies()
        elif stage == "garch":
            run_cmd(["Rscript", str(CODE_DIR / "garch_models.R")], log)
        elif stage == "var":
            run_var_appendix()
        elif stage == "case_studies":
            run_case_studies()
    log.info("Pipeline complete. Total time: %.1fs", time.time() - t0)


if __name__ == "__main__":
    main()
