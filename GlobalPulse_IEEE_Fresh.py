# -*- coding: utf-8 -*-
"""
GLOBALPULSE IEEE FRESH RUN  (derived from GlobalPulse_V6.py)
=============================================================
Global Economic Development & Risk Intelligence - World Bank WDI -> country-year
panel -> forecasting, classification, segmentation, dimensionality reduction and
ensembles, with a publication-style figure set and a one-page executive dashboard.

PURPOSE OF THIS FILE
--------------------
A separate, clean and reproducible execution of GlobalPulse for the IEEE research
paper. It is GlobalPulse_V6.py with ONLY these changes:

  * All output goes to one new directory (OUTPUT_DIR below), organised as
        data/  figures/  tables/  reports/  logs/
  * The old ``globalpulse_out`` folder is never read, written, or deleted.
    A safety check refuses to run if OUTPUT_DIR points at/inside it.
  * The dataset cache lives in OUTPUT_DIR/data. With FRESH_DOWNLOAD = True
    (default) the cache is ignored and EVERY indicator plus the country metadata
    is downloaded again from the World Bank API (1960-2025).
  * The download is verified. If the API fails (or an indicator is only partially
    downloaded) the script reports the failure and stops; it never claims fresh
    data it did not get. A download_manifest.json documents provenance.
  * Everything printed is also saved to logs/execution_log.txt.
  * The absolute output directory is printed at start; generated files are
    listed at the end (also when the run fails part-way).

NOT CHANGED: indicators, features, preprocessing, models, hyper-parameters,
validation/test splits, random seeds, and all figure code.

DATA SOURCE
-----------
World Bank World Development Indicators (WDI), Indicators API v2. No API key.

NOTES
-----
- Target: next-year real GDP growth. Features come from year t (or earlier).
- Chronological split by target year; newest years are an untouched final test.
- Region dummies are used; the World Bank income label is NOT used as a predictor
  because it is a recent classification that reflects past growth (look-ahead).
- Educational macroeconomic analysis, not a credit rating or a causal model.
"""

# %% 0. CONFIGURATION ----------------------------------------------------------
import os
import sys
import json
import time
import hashlib
import platform
import textwrap
import warnings
import inspect
import traceback
from datetime import datetime
from pathlib import Path

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)
RUN_STARTED = time.time()

START_YEAR = 1960
END_YEAR = 2025

# --- Fresh-download controls (IEEE run) --------------------------------------
FRESH_DOWNLOAD = True           # True  = ignore any cache and download everything again from the API
                                # False = may reuse a cache that is INSIDE OUTPUT_DIR/data only
STRICT_FRESH_DOWNLOAD = True    # True  = stop if any indicator FAILED / is PARTIAL (recommended for the paper)
REFRESH_DATA = FRESH_DOWNLOAD   # name used by the unchanged V6 download code
RUN_LABEL = "IEEE research-paper run (GlobalPulse_IEEE_Fresh)"

PER_PAGE = 3000
YEAR_CHUNK_SIZE = 10
COUNTRY_BATCH_SIZE = 25
REQUEST_TIMEOUT = 60
MAX_RETRIES = 6
TEST_YEAR_FRACTION = 0.20
VALIDATION_YEAR_FRACTION = 0.20
MIN_FEATURES_PRESENT = 4
MIN_TRAIN_FEATURE_COVERAGE = 0.25
MAX_CLUSTER_SAMPLE = 4000
MAX_HIERARCHICAL_SAMPLE = 450
MIN_CLUSTER_SHARE = 0.05
SNAPSHOT_WINDOW_YEARS = 3       # "latest available value within N years" for clustering
COVID_YEARS = (2020, 2021)      # reported separately in the regression sensitivity check
LABEL_COUNTRIES = ["IND", "CHN", "USA", "DEU", "BRA", "NGA", "JPN", "BGD", "ZAF", "RUS", "IDN", "PAK", "ETH", "MEX", "TUR", "GBR"]
RANDOM_SEED = 42

# --- Output location (completely separate from the V6 'globalpulse_out') ------
# Optional override for testing on another machine: set env var GLOBALPULSE_IEEE_DIR.
OUTPUT_DIR = Path(os.environ.get("GLOBALPULSE_IEEE_DIR", r"C:\Users\Pharhan\Downloads\GlobalPulse_IEEE_Fresh"))


def _assert_safe_output_dir(path):
    """Refuse to run if the output directory is, contains, or sits inside the old V6 folder."""
    resolved = path.resolve()
    legacy = {(Path.cwd() / "globalpulse_out").resolve()}
    try:
        legacy.add((Path(__file__).resolve().parent / "globalpulse_out").resolve())
    except NameError:                       # __file__ is undefined in some interactive consoles
        pass
    for old in legacy:
        if resolved == old or old in resolved.parents or resolved in old.parents:
            raise RuntimeError(f"Unsafe OUTPUT_DIR {resolved}: it overlaps the original V6 folder {old}. "
                               "Choose a separate directory so the reference results stay untouched.")
    if resolved.name.lower() == "globalpulse_out" or any(p.name.lower() == "globalpulse_out" for p in resolved.parents):
        raise RuntimeError(f"Unsafe OUTPUT_DIR {resolved}: it is named/located like the original 'globalpulse_out'.")


_assert_safe_output_dir(OUTPUT_DIR)
PROJECT_DIR = OUTPUT_DIR
DATA_DIR = PROJECT_DIR / "data"
FIG_DIR = PROJECT_DIR / "figures"
TABLE_DIR = PROJECT_DIR / "tables"
REPORT_DIR = PROJECT_DIR / "reports"
LOG_DIR = PROJECT_DIR / "logs"
for folder in (PROJECT_DIR, DATA_DIR, FIG_DIR, TABLE_DIR, REPORT_DIR, LOG_DIR):
    folder.mkdir(parents=True, exist_ok=True)

API_BASE = "https://api.worldbank.org/v2"
WDI_SOURCE_ID = 2

INDICATORS = {
    "GDP_GROWTH": {"code": "NY.GDP.MKTP.KD.ZG", "name": "GDP growth (annual %)", "role": "Target and previous-year predictor"},
    "GDP_PER_CAPITA_GROWTH": {"code": "NY.GDP.PCAP.KD.ZG", "name": "GDP per capita growth (annual %)", "role": "Predictor"},
    "GDP_PER_CAPITA_USD": {"code": "NY.GDP.PCAP.CD", "name": "GDP per capita (current US$)", "role": "Descriptive context only"},
    "GDP_PER_CAPITA_REAL_USD": {"code": "NY.GDP.PCAP.KD", "name": "GDP per capita (constant 2015 US$)", "role": "Real GDP-per-capita predictor"},
    "POPULATION": {"code": "SP.POP.TOTL", "name": "Population, total", "role": "Predictor"},
    "URBAN_POP_PCT": {"code": "SP.URB.TOTL.IN.ZS", "name": "Urban population (% of total population)", "role": "Predictor"},
    "INFLATION": {"code": "FP.CPI.TOTL.ZG", "name": "Inflation, consumer prices (annual %)", "role": "Predictor"},
    "UNEMPLOYMENT": {"code": "SL.UEM.TOTL.ZS", "name": "Unemployment, total (% of labour force; modelled ILO estimate)", "role": "Predictor"},
    "EXPORTS_PCT_GDP": {"code": "NE.EXP.GNFS.ZS", "name": "Exports of goods and services (% of GDP)", "role": "Predictor"},
    "IMPORTS_PCT_GDP": {"code": "NE.IMP.GNFS.ZS", "name": "Imports of goods and services (% of GDP)", "role": "Predictor"},
    "FDI_PCT_GDP": {"code": "BX.KLT.DINV.WD.GD.ZS", "name": "Foreign direct investment, net inflows (% of GDP)", "role": "Predictor"},
    "LIFE_EXPECTANCY": {"code": "SP.DYN.LE00.IN", "name": "Life expectancy at birth, total (years)", "role": "Predictor"},
    "INTERNET_USERS_PCT": {"code": "IT.NET.USER.ZS", "name": "Individuals using the Internet (% of population)", "role": "Predictor"},
    "LITERACY_PCT": {"code": "SE.ADT.LITR.ZS", "name": "Adult literacy rate (% ages 15 and above)", "role": "Predictor; sparse"},
    "HEALTH_EXP_PCT_GDP": {"code": "SH.XPD.CHEX.GD.ZS", "name": "Current health expenditure (% of GDP)", "role": "Predictor"},
    "CO2_PER_CAPITA": {"code": "EN.GHG.CO2.PC.CE.AR5", "name": "Carbon dioxide emissions excluding LULUCF per capita (t CO2e/capita)", "role": "Predictor"},
}

# Region dummies (REGION_*) are appended automatically in create_forecast_frame().
PREDICTOR_KEYS = [
    "GDP_GROWTH", "GDP_PER_CAPITA_GROWTH", "GDP_PER_CAPITA_REAL_USD",
    "LOG_GDP_PER_CAPITA_REAL", "LOG_POPULATION", "URBAN_POP_PCT",
    "INFLATION", "UNEMPLOYMENT", "EXPORTS_PCT_GDP", "IMPORTS_PCT_GDP",
    "FDI_PCT_GDP", "LIFE_EXPECTANCY", "INTERNET_USERS_PCT", "CO2_PER_CAPITA",
    "LITERACY_PCT", "HEALTH_EXP_PCT_GDP",
    "GDP_GROWTH_LAG1", "GDP_GROWTH_LAG2", "GDP_GROWTH_CHANGE_1Y", "GDP_GROWTH_ROLLING3",
    "GDP_PER_CAPITA_GROWTH_LAG1", "INFLATION_LAG1",
    "GDP_GROWTH_VOL5", "COUNTRY_MEAN_GROWTH_TO_DATE",
    "GLOBAL_MEDIAN_GROWTH", "GLOBAL_MEDIAN_GROWTH_LAG1", "GROWTH_VS_GLOBAL",
]

# %% 1. DEPENDENCIES -----------------------------------------------------------
def require_import(module_name, install_name=None):
    try:
        return __import__(module_name)
    except ImportError as exc:
        pkg = install_name or module_name
        raise RuntimeError(
            f"Required package '{pkg}' is missing.\n"
            f"Open Anaconda Prompt, activate your environment and run:\n"
            f"conda install -c conda-forge {pkg} -y\nThen restart the Spyder kernel.\nOriginal error: {exc}"
        ) from exc

for _mod, _pkg in [("requests", None), ("numpy", None), ("pandas", None), ("matplotlib", None),
                   ("sklearn", "scikit-learn"), ("scipy", None), ("statsmodels", None)]:
    require_import(_mod, _pkg)

import matplotlib
if os.environ.get("GLOBALPULSE_HEADLESS") == "1":
    matplotlib.use("Agg")
import requests
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cycler
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle
from matplotlib.lines import Line2D
import statsmodels.api as sm
from scipy.cluster.hierarchy import linkage, dendrogram, cophenet
from scipy.spatial import ConvexHull
from scipy.spatial.distance import pdist
from scipy.stats import gaussian_kde
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, RobustScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression, RidgeCV, HuberRegressor
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import (BaggingClassifier, GradientBoostingClassifier, RandomForestClassifier,
                              RandomForestRegressor, GradientBoostingRegressor)
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.inspection import permutation_importance
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score, accuracy_score, precision_score,
    recall_score, f1_score, log_loss, roc_auc_score, confusion_matrix, classification_report,
    silhouette_score, adjusted_rand_score, roc_curve, precision_recall_curve, average_precision_score,
)

try:
    from mlxtend.frequent_patterns import apriori, association_rules
    MLXTEND_AVAILABLE = True
except ImportError:
    MLXTEND_AVAILABLE = False
    print("[OPTIONAL] mlxtend not found; Apriori will be skipped (conda install -c conda-forge mlxtend -y).")

np.random.seed(RANDOM_SEED)

# %% 2. DESIGN SYSTEM + CONSOLE HELPERS ---------------------------------------
C = dict(navy="#1B2A49", blue="#2F6DB5", sky="#8DB7E8", teal="#1F9E89", green="#5BAA5F",
         amber="#F2B134", orange="#EE8434", red="#D1495B", purple="#7B5EA7",
         grey="#8A94A6", light="#EEF1F6", ink="#22262E", mid="#4A5568")
CAT = [C["blue"], C["teal"], C["orange"], C["purple"], C["red"], C["amber"], C["green"], C["navy"]]
TIER_COLORS = [C["orange"], C["teal"], C["blue"], C["purple"], C["red"], C["amber"]]
SEQ = LinearSegmentedColormap.from_list("gp_seq", ["#F6F8FC", "#BFD4F0", C["blue"], C["navy"]])
DIV = LinearSegmentedColormap.from_list("gp_div", [C["red"], "#FFFFFF", C["blue"]])

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 250, "savefig.bbox": "tight", "savefig.facecolor": "white",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.family": "sans-serif",
    "font.sans-serif": ["Segoe UI", "Helvetica Neue", "Arial", "DejaVu Sans"],
    "font.size": 10, "axes.titlesize": 11.5, "axes.titleweight": "bold", "axes.titlelocation": "left",
    "axes.labelcolor": C["ink"], "axes.labelsize": 9.5, "axes.edgecolor": "#C9D0DB", "axes.linewidth": 0.8,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": "#E6EAF0", "grid.linewidth": 0.7, "axes.axisbelow": True,
    "xtick.color": C["mid"], "ytick.color": C["mid"], "text.color": C["ink"],
    "legend.frameon": False, "legend.fontsize": 8.5,
    "axes.prop_cycle": cycler(color=CAT),
})

LABELS = {
    "GDP_GROWTH": "GDP growth", "GDP_PER_CAPITA_GROWTH": "GDP per-capita growth",
    "GDP_PER_CAPITA_USD": "GDP per capita (current US$)", "GDP_PER_CAPITA_REAL_USD": "Real GDP per capita",
    "POPULATION": "Population", "URBAN_POP_PCT": "Urban population %", "INFLATION": "Inflation",
    "UNEMPLOYMENT": "Unemployment", "EXPORTS_PCT_GDP": "Exports % GDP", "IMPORTS_PCT_GDP": "Imports % GDP",
    "FDI_PCT_GDP": "FDI inflow % GDP", "LIFE_EXPECTANCY": "Life expectancy",
    "INTERNET_USERS_PCT": "Internet users %", "LITERACY_PCT": "Adult literacy %",
    "HEALTH_EXP_PCT_GDP": "Health spend % GDP", "CO2_PER_CAPITA": "CO\u2082 per capita",
    "LOG_GDP_PER_CAPITA_REAL": "log real GDP per capita", "LOG_POPULATION": "log population",
    "GDP_GROWTH_CHANGE_1Y": "GDP growth change (1y)", "GDP_GROWTH_ROLLING3": "GDP growth (3y avg)",
    "GDP_GROWTH_VOL5": "GDP growth volatility (5y)", "COUNTRY_MEAN_GROWTH_TO_DATE": "Country mean growth to date",
    "GLOBAL_MEDIAN_GROWTH": "World median growth", "GLOBAL_MEDIAN_GROWTH_LAG1": "World median growth (t\u22121)",
    "GROWTH_VS_GLOBAL": "Growth gap vs world",
}


def pretty(name):
    n, flag = str(name), ""
    if n.startswith("missingindicator_"):
        n, flag = n[len("missingindicator_"):], " (missing flag)"
    if n in LABELS:
        return LABELS[n] + flag
    if n.startswith("REGION_"):
        return n[7:] + flag
    for suffix, tag in (("_LAG1", " (t\u22121)"), ("_LAG2", " (t\u22122)")):
        if n.endswith(suffix):
            return pretty(n[:-len(suffix)]) + tag + flag
    return n.replace("_", " ").capitalize() + flag


def banner(title):
    print("\n" + "=" * 78 + f"\n  {title}\n" + "=" * 78)


def info(tag, msg):
    print(f"  [{tag}] {msg}")


def finish(fig, name, title, subtitle=None, tight=True):
    """Consistent title block + footer, then save/show/close."""
    h = fig.get_figheight()
    top_in = 0.95 if subtitle else 0.62
    if tight:
        fig.tight_layout(rect=[0, 0.30 / h, 1, 1 - top_in / h])
    else:
        fig.subplots_adjust(left=0.07, right=0.985, top=1 - (top_in + 0.12) / h, bottom=0.06)
    fig.text(0.005, 1 - 0.14 / h, title, ha="left", va="top", fontsize=15, fontweight="bold", color=C["ink"])
    if subtitle:
        fig.text(0.005, 1 - 0.50 / h, subtitle, ha="left", va="top", fontsize=9.5, color=C["mid"])
    fig.text(0.995, 0.004, "GlobalPulse \u00b7 World Bank WDI", ha="right", va="bottom", fontsize=7, color=C["grey"])
    path = FIG_DIR / f"{name}.png"
    fig.savefig(path)
    plt.show()
    plt.close(fig)
    info("FIGURE", path.name)


def style_axis(ax, xgrid=False, ygrid=True):
    ax.grid(xgrid, axis="x")
    ax.grid(ygrid, axis="y")
    ax.tick_params(length=0)


def callout(ax, text, loc="upper right", fontsize=8.5):
    x, y, ha, va = {"upper right": (0.98, 0.97, "right", "top"), "upper left": (0.02, 0.97, "left", "top"),
                    "lower right": (0.98, 0.03, "right", "bottom"), "lower left": (0.02, 0.03, "left", "bottom")}[loc]
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va=va, fontsize=fontsize, color=C["ink"], linespacing=1.5,
            bbox=dict(boxstyle="round,pad=0.45", fc="#F6F8FC", ec="#D5DCE8", lw=0.8))


def fmt_alpha(a):
    return f"{a:g}"

# %% 2b. RUN LOGGING, PROVENANCE AND FILE LISTING (new in the IEEE run) --------
class _Tee:
    """Duplicate everything written to the console into the log file."""

    def __init__(self, stream, log_handle):
        self._stream, self._log = stream, log_handle

    def write(self, text):
        try:
            self._stream.write(text)
        except Exception:
            pass
        try:
            self._log.write(text)
            self._log.flush()
        except Exception:
            pass
        return len(text)

    def flush(self):
        try:
            self._stream.flush()
        except Exception:
            pass

    def __getattr__(self, name):            # delegate everything else (isatty, encoding, ...) to the console stream
        return getattr(self._stream, name)


_LOG_STATE = {"handle": None, "stdout": None, "stderr": None, "path": None}
DATA_PROVENANCE = {}


def _package_versions():
    out = {}
    for mod in ("numpy", "pandas", "scipy", "sklearn", "statsmodels", "matplotlib", "requests", "mlxtend"):
        try:
            out[mod] = getattr(__import__(mod), "__version__", "unknown")
        except Exception:
            out[mod] = "not installed"
    return out


def start_logging():
    log_path = LOG_DIR / "execution_log.txt"
    if log_path.exists():                     # never overwrite an earlier log in the new folder; archive it
        stamp = datetime.fromtimestamp(log_path.stat().st_mtime).strftime("%Y%m%d_%H%M%S")
        archive = LOG_DIR / f"execution_log_previous_{stamp}.txt"
        n = 1
        while archive.exists():
            archive = LOG_DIR / f"execution_log_previous_{stamp}_{n}.txt"
            n += 1
        log_path.rename(archive)
    handle = open(log_path, "w", encoding="utf-8", errors="replace")
    _LOG_STATE.update({"handle": handle, "stdout": sys.stdout, "stderr": sys.stderr, "path": log_path})
    sys.stdout = _Tee(sys.stdout, handle)
    sys.stderr = _Tee(sys.stderr, handle)
    print(f"GlobalPulse IEEE fresh run | log started {datetime.now().astimezone().isoformat(timespec='seconds')}")
    print(f"Python {sys.version.split()[0]} on {platform.platform()}")
    print("Packages: " + ", ".join(f"{k} {v}" for k, v in _package_versions().items()))
    return log_path


def stop_logging():
    if _LOG_STATE["handle"] is None:
        return
    try:
        sys.stdout.flush()
        sys.stderr.flush()
    except Exception:
        pass
    sys.stdout, sys.stderr = _LOG_STATE["stdout"], _LOG_STATE["stderr"]
    try:
        _LOG_STATE["handle"].close()
    finally:
        _LOG_STATE["handle"] = None


def list_generated_files():
    banner("GENERATED FILES")
    print(f"  Output directory (absolute): {PROJECT_DIR.resolve()}")
    total_files = total_bytes = 0
    for folder in (DATA_DIR, FIG_DIR, TABLE_DIR, REPORT_DIR, LOG_DIR):
        files = sorted(p for p in folder.rglob("*") if p.is_file())
        size = sum(p.stat().st_size for p in files)
        print(f"\n  {folder.name}/  ({len(files)} files, {size / 1024:,.1f} KB)")
        for p in files:
            print(f"    {str(p.relative_to(folder)):<58}{p.stat().st_size / 1024:>10,.1f} KB")
        total_files += len(files)
        total_bytes += size
    print(f"\n  TOTAL: {total_files} files, {total_bytes / 1024 / 1024:,.2f} MB in {PROJECT_DIR.resolve()}")


def _sha256(path, chunk=1 << 20):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def write_download_manifest():
    DATA_PROVENANCE["python"] = sys.version.split()[0]
    DATA_PROVENANCE["packages"] = _package_versions()
    DATA_PROVENANCE["output_directory"] = str(PROJECT_DIR.resolve())
    (DATA_DIR / "download_manifest.json").write_text(json.dumps(DATA_PROVENANCE, indent=2, default=str), encoding="utf-8")

# %% 3. WORLD BANK API HELPERS -----------------------------------------------
class PermanentAPIError(RuntimeError):
    """An API response indicates a non-transient request/series problem."""


SESSION = requests.Session()
SESSION.headers.update({"User-Agent": "GlobalPulse-INT234-Student-Research/6.0 (World Bank Indicators API)",
                        "Accept": "application/json"})


def api_get_json(url, params=None):
    last_error = None
    backoffs = [3, 6, 12, 20, 30, 45]
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = SESSION.get(url, params=params, timeout=REQUEST_TIMEOUT)
            if response.status_code in (429, 500, 502, 503, 504):
                retry_after = response.headers.get("Retry-After")
                try:
                    wait = min(float(retry_after), 60) if retry_after else backoffs[min(attempt - 1, 5)]
                except (TypeError, ValueError):
                    wait = backoffs[min(attempt - 1, 5)]
                raise requests.HTTPError(f"Transient HTTP {response.status_code}; retry after {wait:.0f}s", response=response)
            if response.status_code >= 400:
                raise PermanentAPIError(f"Non-retryable World Bank API HTTP {response.status_code}: {(response.text or '')[:300]}")
            response.raise_for_status()
            try:
                payload = response.json()
            except ValueError as exc:
                raise ValueError(f"API response was not JSON: {(response.text or '')[:240]}") from exc
            if isinstance(payload, list) and payload and isinstance(payload[0], dict):
                meta = payload[0]
                if "message" in meta and "pages" not in meta:
                    raise PermanentAPIError(f"World Bank API message: {meta.get('message')}")
            elif isinstance(payload, dict) and payload.get("message"):
                raise PermanentAPIError(f"World Bank API message: {payload.get('message')}")
            return payload
        except PermanentAPIError:
            raise
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            print(f"    [API retry {attempt}/{MAX_RETRIES}] {type(exc).__name__}: {exc}")
            if attempt < MAX_RETRIES:
                time.sleep(backoffs[min(attempt - 1, 5)])
    raise RuntimeError(f"World Bank API request failed after {MAX_RETRIES} attempts: {last_error}")


def fetch_country_metadata():
    url = f"{API_BASE}/country"
    rows, page, total_pages = [], 1, 1
    while page <= total_pages:
        payload = api_get_json(url, {"format": "json", "per_page": 400, "page": page})
        if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[0], dict):
            raise RuntimeError("Unexpected country metadata response from World Bank API.")
        meta, records = payload[0], payload[1]
        records = records if isinstance(records, list) else []
        total_pages = int(meta.get("pages", 1) or 1)
        for item in records:
            region_obj = item.get("region") or {}
            region = region_obj.get("value", "") if isinstance(region_obj, dict) else str(region_obj)
            if region.strip().lower() == "aggregates":
                continue
            iso3 = (item.get("id") or "").upper()
            if len(iso3) != 3:
                continue
            income_obj = item.get("incomeLevel") or {}
            lending_obj = item.get("lendingType") or {}
            rows.append({
                "iso3": iso3, "iso2": (item.get("iso2Code") or "").upper(), "country": item.get("name", ""),
                "region": region.strip(),
                "income_level": income_obj.get("value", "") if isinstance(income_obj, dict) else str(income_obj),
                "lending_type": lending_obj.get("value", "") if isinstance(lending_obj, dict) else str(lending_obj),
                "capital_city": item.get("capitalCity", ""),
            })
        page += 1
    result = pd.DataFrame(rows).drop_duplicates("iso3")
    if result.empty:
        raise RuntimeError("No country metadata returned by World Bank API.")
    return result


def _normalise_indicator_records(records, label, code, iso3_set):
    out = []
    for row in records or []:
        if not isinstance(row, dict):
            continue
        iso3 = (row.get("countryiso3code") or "").upper()
        if not iso3 or iso3 not in iso3_set:
            continue
        try:
            year = int(str(row.get("date"))[:4])
            value = row.get("value")
            if value is None:
                continue
            numeric_value = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(numeric_value):
            continue
        country_obj = row.get("country") or {}
        indicator_obj = row.get("indicator") or {}
        out.append({
            "iso3": iso3, "country": country_obj.get("value", "") if isinstance(country_obj, dict) else str(country_obj),
            "year": year, "indicator_key": label, "indicator_code": code,
            "indicator_name": indicator_obj.get("value", label) if isinstance(indicator_obj, dict) else label,
            "value": numeric_value, "observation_status": row.get("obs_status", ""),
        })
    return out


def _fetch_indicator_window(code, label, iso3_set, start_year, end_year, country_selector="all"):
    url = f"{API_BASE}/country/{country_selector}/indicator/{code}"
    params = {"format": "json", "date": f"{start_year}:{end_year}", "per_page": PER_PAGE, "page": 1, "source": WDI_SOURCE_ID}
    payload = api_get_json(url, params)
    if not isinstance(payload, list) or len(payload) < 2 or not isinstance(payload[0], dict):
        raise ValueError(f"Unexpected response for {label} {start_year}-{end_year}: {repr(payload)[:300]}")
    meta = payload[0]
    if "message" in meta and "pages" not in meta:
        raise ValueError(f"World Bank API message for {label}: {meta.get('message')}")
    total_pages = int(meta.get("pages", 1) or 1)
    records = []
    for page in range(1, total_pages + 1):
        if page == 1:
            page_payload = payload
        else:
            params["page"] = page
            page_payload = api_get_json(url, params)
        if not isinstance(page_payload, list) or len(page_payload) < 2:
            raise PermanentAPIError(f"Malformed paginated response for {label}, page {page}.")
        page_records = page_payload[1]
        if page_records is None:
            continue
        if not isinstance(page_records, list):
            raise PermanentAPIError(f"Unexpected records type for {label}, page {page}.")
        records.extend(_normalise_indicator_records(page_records, label, code, iso3_set))
    return records


def fetch_indicator(code, label, iso3_set):
    all_records, failed_windows, completed_windows = [], [], []

    def fetch_recursive(start_year, end_year):
        try:
            rows = _fetch_indicator_window(code, label, iso3_set, start_year, end_year, "all")
            completed_windows.append(f"{start_year}-{end_year}")
            return rows
        except PermanentAPIError as exc:
            failed_windows.append(f"{start_year}-{end_year}: {exc}")
            print(f"\n    [PERMANENT API RESPONSE] {label}: {exc}")
            return []
        except Exception as exc:
            if start_year < end_year:
                mid = (start_year + end_year) // 2
                print(f"\n    [SPLIT] {label} {start_year}-{end_year} failed; retrying as halves.")
                return fetch_recursive(start_year, mid) + fetch_recursive(mid + 1, end_year)
            print(f"\n    [COUNTRY FALLBACK] {label} {start_year}: {exc}")
            country_codes = sorted(iso3_set)
            fallback_rows, fallback_failed = [], []
            for pos in range(0, len(country_codes), COUNTRY_BATCH_SIZE):
                batch = country_codes[pos:pos + COUNTRY_BATCH_SIZE]
                try:
                    fallback_rows.extend(_fetch_indicator_window(code, label, iso3_set, start_year, end_year, ";".join(batch)))
                except Exception as batch_exc:
                    fallback_failed.append(f"{start_year}:{batch[0]}-{batch[-1]} ({batch_exc})")
                    if pos == 0:
                        break
                time.sleep(0.12)
            failed_windows.extend(fallback_failed)
            if fallback_rows:
                completed_windows.append(f"{start_year}-{end_year} (country batches)")
            else:
                failed_windows.append(f"{start_year}-{end_year}: {exc}")
            return fallback_rows

    windows, year = [], START_YEAR
    while year <= END_YEAR:
        window_end = min(year + YEAR_CHUNK_SIZE - 1, END_YEAR)
        windows.append((year, window_end))
        year = window_end + 1
    for n, (y0, y1) in enumerate(windows, start=1):
        print(f"\n    [WINDOW {n}/{len(windows)}] {y0}-{y1}", end="")
        rows = fetch_recursive(y0, y1)
        all_records.extend(rows)
        print(f" -> {len(rows):,} observations", end="")
        time.sleep(0.15)
    if all_records:
        all_records = pd.DataFrame(all_records).drop_duplicates(["iso3", "year", "indicator_key"], keep="last").to_dict("records")
    if all_records and not failed_windows:
        status = "OK"
    elif all_records:
        status = "PARTIAL"
    elif completed_windows and not failed_windows:
        status = "NO DATA"
    else:
        status = "FAILED"
    return all_records, status, completed_windows, failed_windows


# All cache files live INSIDE the new output directory (OUTPUT_DIR/data).
CACHE_LONG = DATA_DIR / "world_bank_wdi_long.csv"
CACHE_COUNTRIES = DATA_DIR / "world_bank_country_metadata.csv"
CACHE_CONFIG = DATA_DIR / "world_bank_cache_config.json"
CACHE_FETCH_LOG = DATA_DIR / "api_fetch_log.csv"


def _cache_config_compatible(old_config, current_config):
    return (old_config.get("start_year") == current_config.get("start_year")
            and old_config.get("end_year") == current_config.get("end_year")
            and old_config.get("source_id") == current_config.get("source_id"))


def download_wdi_data():
    config_now = {"start_year": START_YEAR, "end_year": END_YEAR,
                  "indicators": {k: v["code"] for k, v in INDICATORS.items()}, "source_id": WDI_SOURCE_ID,
                  "fetch_method": "decade-windows-with-recursive-split-and-country-fallback-v2"}
    long_df, countries = pd.DataFrame(), pd.DataFrame()
    cache_usable, old_status, old_log_by_key = False, {}, {}

    if not REFRESH_DATA and CACHE_LONG.exists() and CACHE_COUNTRIES.exists() and CACHE_CONFIG.exists():
        try:
            cache_usable = _cache_config_compatible(json.loads(CACHE_CONFIG.read_text(encoding="utf-8")), config_now)
        except Exception:
            cache_usable = False

    if cache_usable:
        info("CACHE", f"Loading {CACHE_LONG.name}")
        long_df = pd.read_csv(CACHE_LONG)
        countries = pd.read_csv(CACHE_COUNTRIES)
        if not long_df.empty:
            long_df["year"] = pd.to_numeric(long_df["year"], errors="coerce")
            long_df = long_df.dropna(subset=["year"])
            long_df["year"] = long_df["year"].astype(int)
        if CACHE_FETCH_LOG.exists():
            try:
                old_log = pd.read_csv(CACHE_FETCH_LOG)
                if {"indicator_key", "status"}.issubset(old_log.columns):
                    old_status = dict(zip(old_log["indicator_key"].astype(str), old_log["status"].astype(str)))
                    old_log_by_key = {str(r.get("indicator_key")): r for r in old_log.to_dict("records") if r.get("indicator_key")}
            except Exception:
                old_status = {}
        info("CACHE", f"{len(long_df):,} observations loaded; checking each indicator.")
    else:
        info("API", "Fetching World Bank country metadata ...")
        countries = fetch_country_metadata()
        info("API", f"{len(countries)} non-aggregate economies identified.")

    iso3_set = set(countries["iso3"].astype(str))
    existing_keys = set(long_df["indicator_key"].astype(str)) if not long_df.empty else set()

    if REFRESH_DATA or not cache_usable:
        keys_to_fetch = list(INDICATORS.keys())
    else:
        keys_to_fetch = []
        for key, spec in INDICATORS.items():
            status_text = old_status.get(key, "")
            entry = old_log_by_key.get(key, {})
            has_rows = key in existing_keys and (long_df["indicator_key"].astype(str) == key).any()
            old_code = str(entry.get("indicator_code", "") or "")
            if old_code and old_code != spec["code"]:
                info("CACHE", f"{key}: definition changed {old_code} -> {spec['code']}; refetching.")
                long_df = long_df[long_df["indicator_key"].astype(str) != key].copy()
                keys_to_fetch.append(key)
                continue
            verified_no_data = (status_text == "NO DATA" and bool(str(entry.get("completed_windows", "") or ""))
                                and not str(entry.get("failed_windows", "") or "").strip())
            if verified_no_data:
                continue
            if (not has_rows) or status_text not in ("OK", "COMPLETE"):
                keys_to_fetch.append(key)

    if keys_to_fetch:
        info("API", f"Fetching {len(keys_to_fetch)} indicator(s) for {START_YEAR}-{END_YEAR} ...")
    else:
        info("CACHE", "All indicators cached successfully; no API calls needed.")

    log_rows = []
    if CACHE_FETCH_LOG.exists() and cache_usable and not REFRESH_DATA:
        try:
            prev = pd.read_csv(CACHE_FETCH_LOG)
            log_rows = prev.to_dict("records") if not prev.empty else []
        except Exception:
            log_rows = []
    log_by_key = {str(r.get("indicator_key")): r for r in log_rows if r.get("indicator_key")}

    for index, key in enumerate(keys_to_fetch, start=1):
        spec = INDICATORS[key]
        code = spec["code"]
        print(f"  [{index}/{len(keys_to_fetch)}] {key}: {code}", end="")
        started = time.time()
        try:
            rows, status, completed, failed = fetch_indicator(code, key, iso3_set)
            if rows:
                long_df = pd.concat([long_df, pd.DataFrame(rows)], ignore_index=True)
            years = sorted({r["year"] for r in rows})
            log_by_key[key] = {"indicator_key": key, "indicator_code": code, "indicator_name": spec["name"], "status": status,
                               "observations": len(rows), "countries": len({r["iso3"] for r in rows}),
                               "first_year": min(years) if years else np.nan, "last_year": max(years) if years else np.nan,
                               "completed_windows": ";".join(completed), "failed_windows": " | ".join(failed),
                               "seconds": round(time.time() - started, 2)}
            print(f" -> {len(rows):,} observations; status={status}")
        except Exception as exc:
            log_by_key[key] = {"indicator_key": key, "indicator_code": code, "indicator_name": spec["name"], "status": "FAILED",
                               "observations": 0, "countries": 0, "first_year": np.nan, "last_year": np.nan,
                               "completed_windows": "", "failed_windows": str(exc), "seconds": round(time.time() - started, 2)}
            print(f" -> FAILED: {exc}")

    if long_df.empty:
        raise RuntimeError("The World Bank API returned no usable observations. Check internet access. No fake data was generated.")
    long_df = long_df.drop_duplicates(["iso3", "year", "indicator_key"], keep="last")
    long_df["year"] = long_df["year"].astype(int)
    long_df.to_csv(CACHE_LONG, index=False, encoding="utf-8-sig")
    countries.to_csv(CACHE_COUNTRIES, index=False, encoding="utf-8-sig")
    CACHE_CONFIG.write_text(json.dumps(config_now, indent=2), encoding="utf-8")

    final_log = []
    for key, spec in INDICATORS.items():
        row = log_by_key.get(key)
        if row is None:
            subset = long_df[long_df["indicator_key"].astype(str) == key]
            yrs = sorted(subset["year"].unique())
            row = {"indicator_key": key, "indicator_code": spec["code"], "indicator_name": spec["name"],
                   "status": "OK" if len(subset) else "NO DATA", "observations": len(subset),
                   "countries": subset["iso3"].nunique() if len(subset) else 0,
                   "first_year": min(yrs) if yrs else np.nan, "last_year": max(yrs) if yrs else np.nan,
                   "completed_windows": "cached", "failed_windows": "", "seconds": 0}
        final_log.append(row)
    pd.DataFrame(final_log).to_csv(CACHE_FETCH_LOG, index=False, encoding="utf-8-sig")
    pd.DataFrame([{"indicator_key": k, "indicator_code": v["code"], "indicator_name": v["name"], "role": v["role"]}
                  for k, v in INDICATORS.items()]).to_csv(DATA_DIR / "indicator_dictionary.csv", index=False, encoding="utf-8-sig")
    if "GDP_GROWTH" not in set(long_df["indicator_key"].astype(str)):
        raise RuntimeError("GDP_GROWTH is unavailable. Check data/api_fetch_log.csv; rerun to retry failed series only.")
    info("API", f"{len(long_df):,} observations in local cache.")
    return long_df, countries


def download_fresh_data():
    """Run the (unchanged) V6 download, then verify it and record provenance.
    Never claims a fresh download that did not happen."""
    banner("DATA ACQUISITION")
    cache_existed_before = CACHE_LONG.exists()
    DATA_PROVENANCE.update({
        "run_label": RUN_LABEL, "source": "World Bank WDI Indicators API v2", "api_base": API_BASE,
        "period_requested": f"{START_YEAR}-{END_YEAR}", "indicators_requested": len(INDICATORS),
        "fresh_download_requested": bool(FRESH_DOWNLOAD), "strict_mode": bool(STRICT_FRESH_DOWNLOAD),
        "started_at": datetime.now().astimezone().isoformat(timespec="seconds"), "status": "RUNNING"})
    if FRESH_DOWNLOAD:
        info("DOWNLOAD", "FRESH_DOWNLOAD = True: any cache is ignored; country metadata and all indicators are requested from the API.")
    else:
        info("DOWNLOAD", "FRESH_DOWNLOAD = False: only a cache inside the NEW directory may be reused (the old globalpulse_out is never read).")
    info("DOWNLOAD", f"Cache location: {DATA_DIR.resolve()}")
    t0 = time.time()
    try:
        long_df, countries = download_wdi_data()
    except Exception as exc:
        DATA_PROVENANCE.update({"status": "FAILED", "error": f"{type(exc).__name__}: {exc}",
                                "finished_at": datetime.now().astimezone().isoformat(timespec="seconds")})
        write_download_manifest()
        print("\n" + "!" * 78)
        print("  WORLD BANK DOWNLOAD FAILED - NO FRESH DATA WAS OBTAINED")
        print(f"  Reason: {type(exc).__name__}: {exc}")
        print("  Nothing was substituted from the old project folder and no data was invented.")
        print(f"  Details: {DATA_DIR / 'download_manifest.json'}")
        print("!" * 78)
        raise RuntimeError(f"World Bank download failed: {exc}") from exc

    log_df = pd.read_csv(CACHE_FETCH_LOG) if CACHE_FETCH_LOG.exists() else pd.DataFrame()
    if log_df.empty or "indicator_key" not in log_df.columns:
        raise RuntimeError("api_fetch_log.csv is missing or unreadable; the download cannot be verified.")
    status = dict(zip(log_df["indicator_key"].astype(str), log_df["status"].astype(str)))
    window_text = dict(zip(log_df["indicator_key"].astype(str), log_df["completed_windows"].fillna("").astype(str)))
    failed = [k for k in INDICATORS if status.get(k) in ("FAILED", "PARTIAL")]
    no_data = [k for k in INDICATORS if status.get(k) == "NO DATA"]
    missing = [k for k in INDICATORS if k not in set(long_df["indicator_key"].astype(str))]
    from_cache = [k for k in INDICATORS if window_text.get(k, "") == "cached"]
    mode = "fresh_api_download" if (FRESH_DOWNLOAD or not cache_existed_before) else "cache_reuse_inside_new_directory"

    DATA_PROVENANCE.update({
        "mode": mode, "finished_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "seconds": round(time.time() - t0, 1), "observations": int(len(long_df)), "countries": int(long_df["iso3"].nunique()),
        "year_min": int(long_df["year"].min()), "year_max": int(long_df["year"].max()),
        "long_csv_sha256": _sha256(CACHE_LONG), "indicators_failed_or_partial": failed,
        "indicators_no_data": no_data, "indicators_missing_from_data": missing,
        "indicator_status": log_df[["indicator_key", "indicator_code", "status", "observations", "countries",
                                    "first_year", "last_year"]].to_dict("records")})

    if FRESH_DOWNLOAD and from_cache:        # defensive: a fresh run must not contain cached indicators
        DATA_PROVENANCE["status"] = "NOT_FRESH"
        write_download_manifest()
        raise RuntimeError(f"FRESH_DOWNLOAD requested but these indicators came from cache: {from_cache}")
    if failed or missing:
        DATA_PROVENANCE["status"] = "INCOMPLETE"
        write_download_manifest()
        print("\n" + "!" * 78)
        print("  DOWNLOAD INCOMPLETE")
        for k in failed:
            print(f"    - {k} ({INDICATORS[k]['code']}): status {status.get(k)}")
        for k in missing:
            print(f"    - {k}: no observations in the dataset")
        print("!" * 78)
        if STRICT_FRESH_DOWNLOAD:
            raise RuntimeError("Fresh download incomplete for: " + ", ".join(sorted(set(failed + missing))) +
                               ". Re-run later, or set STRICT_FRESH_DOWNLOAD = False to continue with partial data "
                               "(the report will flag this).")
        info("WARNING", "STRICT_FRESH_DOWNLOAD = False: continuing with partial data; this is flagged in the report.")
    else:
        DATA_PROVENANCE["status"] = "OK"
    if no_data:
        info("NOTE", "Indicators the API reported as having no data: " + ", ".join(no_data))
    write_download_manifest()
    label = "FRESH API DOWNLOAD VERIFIED" if mode == "fresh_api_download" and DATA_PROVENANCE["status"] == "OK" else f"{mode} / {DATA_PROVENANCE['status']}"
    info("DOWNLOAD", f"{label}: {len(long_df):,} observations, {long_df['iso3'].nunique()} countries, "
                     f"{DATA_PROVENANCE['year_min']}-{DATA_PROVENANCE['year_max']}, {DATA_PROVENANCE['seconds']}s")
    info("DOWNLOAD", f"SHA-256 of world_bank_wdi_long.csv: {DATA_PROVENANCE['long_csv_sha256']}")
    return long_df, countries

# %% 4. COUNTRY-YEAR PANEL ----------------------------------------------------
ARTIFACTS = {}   # figures/metrics shared with the executive dashboard


def build_panel(long_df, countries):
    banner("DATA PANEL")
    wide = long_df.pivot_table(index=["iso3", "country", "year"], columns="indicator_key", values="value", aggfunc="first").reset_index()
    wide.columns.name = None
    wide = wide.merge(countries, on="iso3", how="left", suffixes=("", "_meta"))
    if "country_meta" in wide.columns:
        wide["country"] = wide["country_meta"].fillna(wide["country"])
        wide = wide.drop(columns=["country_meta"])
    wide["year"] = pd.to_numeric(wide["year"], errors="coerce").astype(int)
    wide = wide.sort_values(["iso3", "year"]).reset_index(drop=True)
    wide.to_csv(DATA_DIR / "globalpulse_country_year_panel.csv", index=False, encoding="utf-8-sig")

    rows = []
    for key, spec in INDICATORS.items():
        has = key in wide.columns
        vals = wide.loc[wide[key].notna(), "year"] if has else pd.Series(dtype=float)
        rows.append({"indicator_key": key, "indicator_code": spec["code"], "indicator_name": spec["name"],
                     "observations": int(wide[key].notna().sum()) if has else 0,
                     "countries": int(wide.loc[wide[key].notna(), "iso3"].nunique()) if has else 0,
                     "first_year": int(vals.min()) if len(vals) else np.nan, "latest_year": int(vals.max()) if len(vals) else np.nan,
                     "missing_pct": round(float(wide[key].isna().mean() * 100), 2) if has else 100.0})
    coverage = pd.DataFrame(rows)
    coverage.to_csv(TABLE_DIR / "indicator_coverage.csv", index=False, encoding="utf-8-sig")
    info("PANEL", f"{len(wide):,} country-year rows | {wide['iso3'].nunique()} countries | {wide['year'].min()}-{wide['year'].max()}")
    print(coverage[["indicator_key", "observations", "countries", "first_year", "latest_year", "missing_pct"]].to_string(index=False))
    return wide, coverage

# %% 5. EXPLORATORY DATA ANALYSIS (FIGURES) ----------------------------------
def fig_coverage_heatmap(panel):
    keys = [k for k in INDICATORS if k in panel.columns]
    starts = list(range(START_YEAR, END_YEAR + 1, 5))
    n_countries = panel["iso3"].nunique()
    M = np.zeros((len(keys), len(starts)))
    for j, s in enumerate(starts):
        sub = panel[(panel["year"] >= s) & (panel["year"] < s + 5)]
        for i, k in enumerate(keys):
            M[i, j] = sub.loc[sub[k].notna(), "iso3"].nunique() / n_countries * 100
    order = np.argsort(-M.mean(axis=1))
    M = M[order]
    labels = [pretty(keys[i]) for i in order]
    fig, ax = plt.subplots(figsize=(12, 6.2))
    im = ax.imshow(M, aspect="auto", cmap=SEQ, vmin=0, vmax=100)
    ax.set_xticks(range(len(starts)))
    ax.set_xticklabels([f"{s}\u2013{str(s + 4)[2:]}" for s in starts], rotation=0, fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.grid(False)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=7.5, color="white" if v > 55 else C["mid"])
    cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.015)
    cb.set_label("% of countries with \u22651 observation in window", fontsize=8.5)
    cb.outline.set_visible(False)
    finish(fig, "01_data_coverage_heatmap", "Where the data is (and isn't)",
           "Share of countries reporting each indicator, by 5-year window. Sparse series are excluded from the supervised models.")


def fig_global_growth(panel):
    q = panel.groupby("year")["GDP_GROWTH"].quantile([.10, .25, .50, .75, .90]).unstack().dropna()
    cnt = panel.groupby("year")["GDP_GROWTH"].count().reindex(q.index)
    ARTIFACTS["growth_q"] = q
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(12, 6.4), sharex=True, gridspec_kw={"height_ratios": [4.2, 1], "hspace": 0.06})
    ax.fill_between(q.index, q[.10], q[.90], color=C["blue"], alpha=0.10, lw=0, label="10th\u201390th percentile")
    ax.fill_between(q.index, q[.25], q[.75], color=C["blue"], alpha=0.25, lw=0, label="25th\u201375th percentile")
    ax.plot(q.index, q[.50], color=C["navy"], lw=2.2, label="Median country")
    ax.axhline(0, color=C["red"], lw=1, ls="--", alpha=0.7)
    events = {1975: "Oil shocks", 1982: "Debt crisis", 1998: "Asian crisis", 2009: "Global financial crisis", 2020: "COVID-19"}
    ymin = max(q[.10].min() - 1, -16)
    ymax = min(q[.90].max() + 2.5, 20)
    ax.set_ylim(ymin, ymax)
    for yr, lab in events.items():
        if yr in q.index:
            ax.axvline(yr, color=C["grey"], lw=0.8, ls=":")
            ax.text(yr + 0.35, ymax - 0.3, lab, rotation=90, va="top", ha="left", fontsize=8, color=C["mid"])
            ax.scatter([yr], [q.loc[yr, .50]], s=36, color=C["red"], zorder=5)
    ax.set_ylabel("GDP growth (%)")
    ax.legend(loc="lower left", ncol=3)
    style_axis(ax)
    ax2.fill_between(cnt.index, cnt.values, color=C["teal"], alpha=0.35, lw=0)
    ax2.plot(cnt.index, cnt.values, color=C["teal"], lw=1.4)
    ax2.set_ylabel("Countries", fontsize=8.5)
    ax2.set_xlabel("Year")
    style_axis(ax2)
    finish(fig, "02_global_growth_fan", "Global growth: a fan chart of country outcomes",
           "Median and spread of country GDP growth each year (unweighted). Dots mark synchronised global shocks; lower strip shows reporting coverage.")


def fig_growth_distribution(panel):
    vals = panel["GDP_GROWTH"].dropna()
    lo, hi = vals.quantile(0.01), vals.quantile(0.99)
    clipped = vals.clip(lo, hi)
    fig, ax = plt.subplots(figsize=(9.5, 4.8))
    ax.hist(clipped, bins=55, density=True, color=C["sky"], alpha=0.85, edgecolor="white", lw=0.5)
    xs = np.linspace(lo, hi, 400)
    ax.plot(xs, gaussian_kde(clipped)(xs), color=C["navy"], lw=2)
    ax.axvline(0, color=C["red"], ls="--", lw=1.2)
    ax.axvline(vals.median(), color=C["teal"], lw=1.8)
    ax.text(vals.median() + 0.2, ax.get_ylim()[1] * 0.93, f"median {vals.median():.1f}%", color=C["teal"], fontsize=9, fontweight="bold")
    ax.set_xlabel("GDP growth (%), shown between 1st and 99th percentile")
    ax.set_ylabel("Density")
    callout(ax, f"n = {len(vals):,} country-years\nmean {vals.mean():.2f}%  |  sd {vals.std():.2f}\nshare contracting: {(vals < 0).mean() * 100:.1f}%\n5th\u201395th pct: {vals.quantile(.05):.1f}% to {vals.quantile(.95):.1f}%")
    style_axis(ax)
    finish(fig, "03_growth_distribution", "How volatile is annual GDP growth?",
           "Heavy tails matter for forecasting: a few extreme country-years inflate squared-error metrics.")


def fig_income_trends(panel):
    order = ["Low income", "Lower middle income", "Upper middle income", "High income"]
    if "income_level" not in panel.columns:
        return
    sub = panel[panel["income_level"].isin(order)]
    if sub.empty:
        return
    g = sub.groupby(["year", "income_level"])["GDP_GROWTH"].median().unstack().reindex(columns=order)
    r = g.rolling(5, center=True, min_periods=3).mean()
    fig, ax = plt.subplots(figsize=(11, 4.8))
    for col, colr in zip(order, [C["red"], C["orange"], C["teal"], C["navy"]]):
        if col in r.columns:
            ax.plot(r.index, r[col], color=colr, lw=2.2, label=col)
            last = r[col].dropna()
            if len(last):
                ax.text(last.index[-1] + 0.4, last.iloc[-1], col.replace(" income", ""), color=colr, fontsize=8.5, va="center")
    ax.axhline(0, color=C["grey"], lw=0.8)
    ax.set_xlim(r.index.min(), r.index.max() + 6)
    ax.set_ylabel("Median GDP growth, 5-yr centred average (%)")
    ax.set_xlabel("Year")
    style_axis(ax)
    finish(fig, "04_growth_by_income_group", "Growth by World Bank income group",
           "Descriptive only: groups use the current World Bank classification and are not used as model inputs.")


def unit1_eda(panel, coverage):
    banner("EXPLORATORY DATA ANALYSIS")
    numeric_cols = [k for k in INDICATORS if k in panel.columns]
    print("Missing values by indicator (%):")
    print((panel[numeric_cols].isna().mean() * 100).round(1).sort_values(ascending=False).to_string())
    if "GDP_GROWTH" in panel.columns:
        panel.loc[panel["GDP_GROWTH"].notna()].groupby("year")["iso3"].nunique().rename("countries_with_gdp_growth").to_csv(
            TABLE_DIR / "countries_with_gdp_growth_by_year.csv")
    fig_coverage_heatmap(panel)
    fig_global_growth(panel)
    fig_growth_distribution(panel)
    fig_income_trends(panel)

# %% 6. LEAKAGE-SAFE FEATURE ENGINEERING --------------------------------------
def _to_long(grid, name):
    out = grid.rename_axis(columns="iso3").reset_index().melt(id_vars="year", var_name="iso3", value_name=name)
    return out


def create_forecast_frame(panel):
    banner("FEATURE ENGINEERING")
    if "GDP_GROWTH" not in panel.columns:
        raise RuntimeError("GDP growth indicator was unavailable; supervised modelling cannot run.")
    frame = panel.sort_values(["iso3", "year"]).copy()
    for col in ("GDP_GROWTH", "GDP_PER_CAPITA_GROWTH", "INFLATION"):
        if col not in frame.columns:
            frame[col] = np.nan

    for source_col, lag_years, new_col in [("GDP_GROWTH", 1, "GDP_GROWTH_LAG1"), ("GDP_GROWTH", 2, "GDP_GROWTH_LAG2"),
                                           ("GDP_PER_CAPITA_GROWTH", 1, "GDP_PER_CAPITA_GROWTH_LAG1"), ("INFLATION", 1, "INFLATION_LAG1")]:
        lag = frame[["iso3", "year", source_col]].copy()
        lag["year"] = lag["year"] + lag_years
        frame = frame.merge(lag.rename(columns={source_col: new_col}), on=["iso3", "year"], how="left", validate="one_to_one")

    frame["LOG_GDP_PER_CAPITA_REAL"] = np.log1p(pd.to_numeric(frame.get("GDP_PER_CAPITA_REAL_USD"), errors="coerce").clip(lower=0)) \
        if "GDP_PER_CAPITA_REAL_USD" in frame.columns else np.nan
    frame["LOG_POPULATION"] = np.log1p(pd.to_numeric(frame.get("POPULATION"), errors="coerce").clip(lower=0)) \
        if "POPULATION" in frame.columns else np.nan
    frame["GDP_GROWTH_CHANGE_1Y"] = frame["GDP_GROWTH"] - frame["GDP_GROWTH_LAG1"]
    frame["GDP_GROWTH_ROLLING3"] = frame[["GDP_GROWTH", "GDP_GROWTH_LAG1", "GDP_GROWTH_LAG2"]].mean(axis=1, skipna=True)

    # Calendar-aligned rolling features built from a complete year grid.
    g = frame.pivot(index="year", columns="iso3", values="GDP_GROWTH")
    g = g.reindex(range(int(g.index.min()), int(g.index.max()) + 1))
    vol5 = g.rolling(5, min_periods=3).std()
    to_date = g.expanding(min_periods=5).mean()
    world_med = g.median(axis=1)
    frame = frame.merge(_to_long(vol5, "GDP_GROWTH_VOL5"), on=["iso3", "year"], how="left")
    frame = frame.merge(_to_long(to_date, "COUNTRY_MEAN_GROWTH_TO_DATE"), on=["iso3", "year"], how="left")
    frame["GLOBAL_MEDIAN_GROWTH"] = frame["year"].map(world_med)
    frame["GLOBAL_MEDIAN_GROWTH_LAG1"] = frame["year"].map(world_med.shift(1))
    frame["GROWTH_VS_GLOBAL"] = frame["GDP_GROWTH"] - frame["GLOBAL_MEDIAN_GROWTH"]

    nxt = frame[["iso3", "year", "GDP_GROWTH"]].copy()
    nxt["year"] = nxt["year"] - 1
    frame = frame.merge(nxt.rename(columns={"GDP_GROWTH": "target_gdp_growth"}), on=["iso3", "year"], how="left", validate="one_to_one")
    frame["target_year"] = frame["year"] + 1

    region_cols = []
    if "region" in frame.columns:
        dummies = pd.get_dummies(frame["region"].fillna("Unknown"), prefix="REGION", dtype=float)
        frame = pd.concat([frame, dummies], axis=1)
        region_cols = list(dummies.columns)

    features = [c for c in PREDICTOR_KEYS if c in frame.columns] + region_cols
    core = [c for c in ["GDP_GROWTH", "GDP_PER_CAPITA_GROWTH", "GDP_PER_CAPITA_REAL_USD", "LOG_POPULATION", "URBAN_POP_PCT",
                        "INFLATION", "EXPORTS_PCT_GDP", "IMPORTS_PCT_GDP"] if c in frame.columns]
    frame["observed_core_predictor_count"] = frame[core].notna().sum(axis=1)
    frame = frame[frame["observed_core_predictor_count"] >= MIN_FEATURES_PRESENT]
    frame = frame.dropna(subset=["target_gdp_growth"])
    frame = frame[np.isfinite(frame["target_gdp_growth"])].copy()
    frame["target_year"] = frame["target_year"].astype(int)
    frame.to_csv(DATA_DIR / "globalpulse_one_year_ahead_model_dataset.csv", index=False, encoding="utf-8-sig")
    info("MODEL DATA", f"{len(frame):,} one-year-ahead samples | {frame['iso3'].nunique()} countries | target years {frame['target_year'].min()}-{frame['target_year'].max()}")
    info("FEATURES", f"{len(features)} candidate predictors (incl. {len(region_cols)} region dummies); nominal GDP per capita and income label excluded.")
    return frame, features


def chronological_split(frame, features):
    years = np.array(sorted(frame["target_year"].unique()))
    if len(years) < 12:
        raise RuntimeError(f"Only {len(years)} target years available; need at least 12.")
    split_pos = max(3, min(int(np.floor(len(years) * (1 - TEST_YEAR_FRACTION))), len(years) - 3))
    first_test_year = int(years[split_pos])
    train = frame[frame["target_year"] < first_test_year].copy()
    test = frame[frame["target_year"] >= first_test_year].copy()

    train_years = np.array(sorted(train["target_year"].unique()))
    if len(train_years) >= 8:
        val_pos = max(2, min(int(np.floor(len(train_years) * (1 - VALIDATION_YEAR_FRACTION))), len(train_years) - 2))
        feature_fit_start = int(train_years[val_pos])
        feature_fit_rows = train[train["target_year"] < feature_fit_start]
    else:
        feature_fit_rows, feature_fit_start = train, None

    usable, excluded = [], []
    for col in features:
        if col not in train.columns:
            continue
        cov = float(feature_fit_rows[col].notna().mean())
        if cov >= MIN_TRAIN_FEATURE_COVERAGE:
            usable.append(col)
        else:
            excluded.append((col, cov))
    if len(usable) < 4:
        raise RuntimeError("Fewer than four predictors meet the training-period coverage threshold.")
    if feature_fit_start is not None:
        info("FEATURE SELECTION", f"Availability assessed only on model-fit years before {feature_fit_start}.")
    for name, cov in excluded:
        info("EXCLUDED", f"{pretty(name)}: only {cov:.1%} observed in model-fit years")
    info("TIME SPLIT", f"Train target years {train['target_year'].min()}-{train['target_year'].max()} ({len(train):,} rows)")
    info("TIME SPLIT", f"Test  target years {test['target_year'].min()}-{test['target_year'].max()} ({len(test):,} rows) | never shuffled")
    return train, test, usable, first_test_year


def inner_validation_split(train):
    years = np.array(sorted(train["target_year"].unique()))
    if len(years) < 8:
        cut = max(1, int(np.floor(len(train) * (1 - VALIDATION_YEAR_FRACTION))))
        fit, val, val_start = train.iloc[:cut].copy(), train.iloc[cut:].copy(), "row-based fallback"
    else:
        pos = max(2, min(int(np.floor(len(years) * (1 - VALIDATION_YEAR_FRACTION))), len(years) - 2))
        val_start = int(years[pos])
        fit, val = train[train["target_year"] < val_start].copy(), train[train["target_year"] >= val_start].copy()
    if len(fit) < 100 or len(val) < 30:
        raise RuntimeError("Insufficient observations for a chronological validation period.")
    info("VALIDATION SPLIT", f"Fit {fit['target_year'].min()}-{fit['target_year'].max()} ({len(fit):,}) | Validation {val['target_year'].min()}-{val['target_year'].max()} ({len(val):,})")
    return fit, val, val_start


def prepare_feature_matrices(fit_frame, other_frame, features):
    """Winsorise (1st/99th pct of fit period), median-impute with missing flags, standardise. Fit on fit_frame only."""
    fit_raw = fit_frame[features].apply(pd.to_numeric, errors="coerce")
    oth_raw = other_frame[features].apply(pd.to_numeric, errors="coerce")
    for col in features:
        if col.startswith("REGION_"):
            continue
        lo, hi = fit_raw[col].quantile(0.01), fit_raw[col].quantile(0.99)
        if pd.notna(lo) and pd.notna(hi) and lo < hi:
            fit_raw[col] = fit_raw[col].clip(lo, hi)
            oth_raw[col] = oth_raw[col].clip(lo, hi)
    imputer = SimpleImputer(strategy="median", add_indicator=True, keep_empty_features=True)
    fit_imp = imputer.fit_transform(fit_raw)
    other_imp = imputer.transform(oth_raw)
    names = np.asarray(imputer.get_feature_names_out(features), dtype=object)
    keep = np.nanvar(fit_imp, axis=0) > 1e-12
    if int(keep.sum()) < 3:
        raise RuntimeError("Fewer than three non-constant predictors remain after preprocessing.")
    fit_imp, other_imp, names = fit_imp[:, keep], other_imp[:, keep], names[keep]
    scaler = StandardScaler()
    return {"imputer": imputer, "keep": keep, "scaler": scaler, "feature_names": names.tolist(),
            "X_fit": scaler.fit_transform(fit_imp), "X_other": scaler.transform(other_imp)}

# %% 7. REGRESSION: NEXT-YEAR GDP GROWTH --------------------------------------
def rmse_fn(y, p):
    return float(np.sqrt(mean_squared_error(y, p)))


def regression_metrics(name, y_true, y_pred):
    y_pred = np.asarray(y_pred, dtype=float)
    mse = mean_squared_error(y_true, y_pred)
    return {"Model": name, "MAE": mean_absolute_error(y_true, y_pred), "MSE": mse, "RMSE": float(np.sqrt(mse)),
            "R2": r2_score(y_true, y_pred) if len(y_true) > 1 else np.nan}


def _safe_fill(series, fallback):
    values = pd.to_numeric(series, errors="coerce")
    med = values.median()
    return values.fillna(float(med) if pd.notna(med) else float(fallback)).to_numpy(dtype=float).reshape(-1, 1)


def _fit_ols_with_diagnostics(train, y_train):
    """Parsimonious OLS with country-clustered robust errors (explanatory, not the headline forecaster)."""
    cand = [c for c in ["GDP_GROWTH", "GDP_PER_CAPITA_GROWTH", "LOG_GDP_PER_CAPITA_REAL", "URBAN_POP_PCT", "INFLATION", "UNEMPLOYMENT",
                        "EXPORTS_PCT_GDP", "FDI_PCT_GDP", "LIFE_EXPECTANCY", "CO2_PER_CAPITA", "GROWTH_VS_GLOBAL", "GDP_GROWTH_VOL5"] if c in train.columns]
    diagnostics, keep = [], []
    for col in cand:
        cov = float(pd.to_numeric(train[col], errors="coerce").notna().mean())
        if cov >= 0.25:
            keep.append(col)
            diagnostics.append({"feature": col, "coverage": cov, "included": True, "reason": "candidate"})
        else:
            diagnostics.append({"feature": col, "coverage": cov, "included": False, "reason": "coverage below 25%"})
    if len(keep) < 3:
        info("OLS", "Skipped: fewer than three predictors with adequate coverage.")
        return None
    raw = train[keep].apply(pd.to_numeric, errors="coerce")
    X = raw.clip(raw.quantile(0.01), raw.quantile(0.99), axis=1).fillna(raw.median())

    def mark(col, reason):
        for r in diagnostics:
            if r["feature"] == col:
                r["included"], r["reason"] = False, reason

    for col in list(X.columns):
        if float(X[col].var(ddof=0)) <= 1e-12:
            X = X.drop(columns=[col]); mark(col, "constant")
    corr = X.corr().abs()
    cols_now, dropped = list(X.columns), set()
    for i, left in enumerate(cols_now):
        if left in dropped:
            continue
        for right in cols_now[i + 1:]:
            if right not in dropped and pd.notna(corr.loc[left, right]) and corr.loc[left, right] >= 0.97:
                dropped.add(right); mark(right, f"near-duplicate of {left}")
    X = X.drop(columns=list(dropped))
    Xs = pd.DataFrame(StandardScaler().fit_transform(X), columns=X.columns, index=X.index)
    independent, design = [], np.ones((len(Xs), 1))
    rank = np.linalg.matrix_rank(design)
    for col in Xs.columns:
        trial = np.column_stack([design, Xs[independent + [col]].to_numpy()])
        if np.linalg.matrix_rank(trial) > rank:
            independent.append(col); rank += 1
        else:
            mark(col, "exact linear dependency")
    Xs = Xs[independent]
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        while Xs.shape[1] > 3:
            dv = sm.add_constant(Xs, has_constant="add").to_numpy()
            vals = {c: float(variance_inflation_factor(dv, i)) for i, c in enumerate(Xs.columns, start=1)}
            worst, wv = max(vals.items(), key=lambda kv: kv[1])
            if np.isfinite(wv) and wv <= 15:
                break
            Xs = Xs.drop(columns=[worst]); mark(worst, f"high VIF ({wv:.1f})")
    except Exception as exc:
        info("OLS", f"VIF check unavailable: {exc}")
    design_df = sm.add_constant(Xs, has_constant="add")
    try:
        ols = sm.OLS(np.asarray(y_train, dtype=float).clip(*np.quantile(y_train, [0.01, 0.99])), design_df).fit(
            cov_type="cluster", cov_kwds={"groups": train["iso3"].astype(str).to_numpy()})
    except Exception as exc:
        info("OLS", f"Cluster-robust OLS failed: {exc}")
        return None
    (REPORT_DIR / "ols_regression_summary_country_clustered.txt").write_text(ols.summary().as_text(), encoding="utf-8")
    ci = ols.conf_int()
    coef = pd.DataFrame({"feature": ols.params.index, "coef": ols.params.values, "se": ols.bse.values,
                         "p": ols.pvalues.values, "lo": ci[0].values, "hi": ci[1].values})
    coef.to_csv(TABLE_DIR / "ols_coefficients_country_clustered.csv", index=False)
    pd.DataFrame(diagnostics).to_csv(TABLE_DIR / "ols_feature_diagnostics.csv", index=False)
    info("OLS", f"{Xs.shape[1]} standardised predictors | condition number {ols.condition_number:.2f} | clustered SEs")
    c = coef[coef["feature"] != "const"].sort_values("coef")
    fig, ax = plt.subplots(figsize=(9, max(3.6, 0.42 * len(c) + 1.6)))
    colors = [C["teal"] if p < 0.05 else C["grey"] for p in c["p"]]
    ax.hlines(range(len(c)), c["lo"], c["hi"], color=colors, lw=2.2)
    ax.scatter(c["coef"], range(len(c)), color=colors, s=48, zorder=3)
    ax.axvline(0, color=C["ink"], lw=0.8)
    ax.set_yticks(range(len(c)))
    ax.set_yticklabels([pretty(f) for f in c["feature"]])
    ax.set_xlabel("Effect on next-year GDP growth (pp per +1 s.d.), 95% CI")
    ax.legend(handles=[Line2D([0], [0], color=C["teal"], lw=3, label="p < 0.05"), Line2D([0], [0], color=C["grey"], lw=3, label="not significant")], loc="lower right")
    style_axis(ax, xgrid=True, ygrid=False)
    finish(fig, "08_ols_coefficients", "What moves next-year growth? (OLS, associational)",
           "Standardised coefficients with country-clustered 95% intervals, fitted on pre-test years. Not causal.")
    return ols


def fig_regression_comparison(results_df):
    df = results_df.sort_values("Validation_RMSE", ascending=False).reset_index(drop=True)
    y = np.arange(len(df))
    h = 0.38
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 0.62 * len(df) + 2.6), sharey=True, gridspec_kw={"width_ratios": [1.5, 1]})
    is_base = (df["Model_Type"] == "Baseline").to_numpy()
    ax.barh(y + h / 2, df["Validation_RMSE"], h, color=C["sky"], label="Validation (earlier years)")
    ax.barh(y - h / 2, df["RMSE"], h, color=[C["grey"] if b else C["blue"] for b in is_base], label="Final test (untouched)")
    for i, r in df.iterrows():
        ax.text(r["Validation_RMSE"] + 0.05, i + h / 2, f"{r['Validation_RMSE']:.2f}", va="center", fontsize=7.5, color=C["mid"])
        ax.text(r["RMSE"] + 0.05, i - h / 2, f"{r['RMSE']:.2f}", va="center", fontsize=7.5, color=C["ink"], fontweight="bold" if r["Selected_on_Validation"] else "normal")
    ax.set_yticks(y)
    ax.set_yticklabels([("\u2605 " if s else "") + m for m, s in zip(df["Model"], df["Selected_on_Validation"])], fontsize=9)
    ax.set_xlabel("RMSE (percentage points) \u2014 lower is better")
    ax.set_xlim(0, max(df["RMSE"].max(), df["Validation_RMSE"].max()) * 1.12)
    ax.legend(handles=[Patch(color=C["sky"], label="Validation"), Patch(color=C["blue"], label="Test: model"), Patch(color=C["grey"], label="Test: baseline")], loc="upper right")
    ax.set_title("Validation vs final test")
    style_axis(ax, xgrid=True, ygrid=False)
    ax2.barh(y, df["RMSE_ex_COVID"], 0.55, color=[C["grey"] if b else C["teal"] for b in is_base])
    for i, v in enumerate(df["RMSE_ex_COVID"]):
        ax2.text(v + 0.04, i, f"{v:.2f}", va="center", fontsize=7.5)
    ax2.set_xlabel(f"Test RMSE excluding {COVID_YEARS[0]}\u2013{COVID_YEARS[1]}")
    ax2.set_title("COVID sensitivity", fontsize=10.5)
    ax2.set_xlim(0, df["RMSE_ex_COVID"].max() * 1.15)
    style_axis(ax2, xgrid=True, ygrid=False)
    finish(fig, "05_regression_scorecard", "Regression scorecard: can we beat naive forecasts?",
           "\u2605 = best real model on validation. Grey bars are naive baselines. The right panel removes the pandemic shock, which breaks persistence forecasts.")


def fig_regression_diagnostics(selected, yte, pred, test, zero_pred, persist_pred):
    lim = np.nanpercentile(np.r_[yte, pred], [1, 99])
    lo, hi = float(lim[0]) - 1, float(lim[1]) + 1
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5.1), gridspec_kw={"width_ratios": [1, 1, 1.25]})
    hb = ax1.hexbin(pred, yte, gridsize=34, extent=(lo, hi, lo, hi), cmap=SEQ, mincnt=1, bins="log", linewidths=0.2)
    ax1.plot([lo, hi], [lo, hi], color=C["red"], lw=1.4, ls="--", label="perfect forecast")
    ax1.set_xlim(lo, hi); ax1.set_ylim(lo, hi)
    ax1.set_xlabel("Predicted growth (%)"); ax1.set_ylabel("Actual growth (%)")
    ax1.set_title("Predicted vs actual")
    ax1.grid(False)
    corr = np.corrcoef(pred, yte)[0, 1] if np.std(pred) > 0 else np.nan
    callout(ax1, f"corr = {corr:.2f}\nn = {len(yte):,}", "upper left")
    ax1.legend(loc="lower right")
    res = yte - pred
    rl, rh = np.percentile(res, [1, 99])
    ax2.hist(np.clip(res, rl, rh), bins=45, color=C["sky"], edgecolor="white", lw=0.5)
    ax2.axvline(0, color=C["red"], ls="--", lw=1.2)
    ax2.axvline(res.mean(), color=C["navy"], lw=1.8)
    ax2.set_xlabel("Forecast error: actual \u2212 predicted (pp)")
    ax2.set_ylabel("Country-years")
    ax2.set_title("Error distribution")
    callout(ax2, f"bias = {res.mean():+.2f} pp\nMAE = {np.abs(res).mean():.2f}\nRMSE = {np.sqrt((res ** 2).mean()):.2f}")
    style_axis(ax2)
    dfp = pd.DataFrame({"year": test["target_year"].to_numpy(), "sel": (yte - pred) ** 2, "zero": (yte - zero_pred) ** 2, "pers": (yte - persist_pred) ** 2})
    by = dfp.groupby("year")[["sel", "zero", "pers"]].mean().pow(0.5)
    ax3.axvspan(COVID_YEARS[0] - 0.5, COVID_YEARS[1] + 0.5, color=C["red"], alpha=0.08, lw=0)
    ax3.plot(by.index, by["pers"], color=C["orange"], lw=1.6, marker="o", ms=3.5, label="Persistence baseline")
    ax3.plot(by.index, by["zero"], color=C["grey"], lw=1.6, marker="o", ms=3.5, label="Zero-growth baseline")
    ax3.plot(by.index, by["sel"], color=C["navy"], lw=2.4, marker="o", ms=4.5, label=selected)
    ax3.set_xlabel("Target year"); ax3.set_ylabel("RMSE (pp)")
    ax3.set_title("Forecast error by year")
    ax3.legend(loc="upper left", fontsize=8)
    style_axis(ax3)
    finish(fig, "06_regression_diagnostics", f"Diagnostics for the validation-selected model: {selected}",
           "Final chronological test period. Shaded band = pandemic years.")


def fig_risk_curves(y_true, prob, prob_rf, threshold):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.7))
    for p, nm, col in [(prob, "Logistic Regression", C["blue"]), (prob_rf, "Random Forest", C["teal"])]:
        if p is None:
            continue
        fpr, tpr, _ = roc_curve(y_true, p)
        a1.plot(fpr, tpr, lw=2.2, color=col, label=f"{nm} (AUC {roc_auc_score(y_true, p):.2f})")
        pr, rc, _ = precision_recall_curve(y_true, p)
        a2.plot(rc, pr, lw=2.2, color=col, label=f"{nm} (AP {average_precision_score(y_true, p):.2f})")
    a1.plot([0, 1], [0, 1], color=C["grey"], ls="--", lw=1)
    a1.set_xlabel("False positive rate"); a1.set_ylabel("True positive rate"); a1.set_title("ROC curve")
    a2.axhline(float(np.mean(y_true)), color=C["grey"], ls="--", lw=1, label=f"base rate {np.mean(y_true):.2f}")
    a2.set_xlabel("Recall"); a2.set_ylabel("Precision"); a2.set_title("Precision\u2013recall curve")
    for a in (a1, a2):
        a.legend(loc="lower right" if a is a1 else "upper right")
        a.set_xlim(0, 1); a.set_ylim(0, 1.02)
        style_axis(a)
    finish(fig, "09_low_growth_risk_curves", f"Early-warning model: next-year growth \u2264 {threshold:.2f}%",
           "Low-growth threshold = 25th percentile of the model-fit period. Final test years.")


def fig_correlations(train):
    cols = [c for c in ["GDP_GROWTH", "GDP_PER_CAPITA_GROWTH", "GDP_PER_CAPITA_REAL_USD", "INFLATION", "UNEMPLOYMENT", "LIFE_EXPECTANCY",
                        "INTERNET_USERS_PCT", "CO2_PER_CAPITA", "GDP_GROWTH_VOL5", "target_gdp_growth"] if c in train.columns and train[c].notna().sum() >= 20]
    corr = train[cols].corr(method="spearman")
    corr.to_csv(TABLE_DIR / "regression_spearman_correlations.csv", encoding="utf-8-sig")
    train[cols].corr(method="pearson").to_csv(TABLE_DIR / "regression_pearson_correlations.csv", encoding="utf-8-sig")
    names = [("Next-year growth" if c == "target_gdp_growth" else pretty(c)) for c in cols]
    fig, ax = plt.subplots(figsize=(8.6, 7))
    im = ax.imshow(corr.values, cmap=DIV, vmin=-1, vmax=1)
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(names, rotation=45, ha="right", fontsize=8.5)
    ax.set_yticks(range(len(cols))); ax.set_yticklabels(names, fontsize=8.5)
    ax.grid(False); ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for i in range(len(cols)):
        for j in range(len(cols)):
            v = corr.values[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.5, color="white" if abs(v) > 0.6 else C["ink"])
    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02); cb.set_label("Spearman correlation"); cb.outline.set_visible(False)
    finish(fig, "07_correlation_heatmap", "How indicators move together",
           "Rank correlations on pre-test training rows. Correlation \u2260 causation.")


def unit2_regression(train, test, features):
    banner("REGRESSION: PREDICTING NEXT-YEAR GDP GROWTH")
    fit_train, validation, validation_start = inner_validation_split(train)
    y_fit = fit_train["target_gdp_growth"].astype(float).to_numpy()
    y_val = validation["target_gdp_growth"].astype(float).to_numpy()
    ytr = train["target_gdp_growth"].astype(float).to_numpy()
    yte = test["target_gdp_growth"].astype(float).to_numpy()
    inner = prepare_feature_matrices(fit_train, validation, features)
    final = prepare_feature_matrices(train, test, features)
    Xfit, Xval, Xtr, Xte = inner["X_fit"], inner["X_other"], final["X_fit"], final["X_other"]

    baseline_names = {"Zero-Growth Baseline", "Training Mean Baseline", "Persistence Baseline", "Country Historical Mean Baseline"}

    def predict_all(fit_df, eval_df, X_fit, X_eval, y_raw):
        lo, hi = np.quantile(y_raw, [0.01, 0.99])
        y_w = np.clip(y_raw, lo, hi)                       # winsorised training target (evaluation stays raw)
        gmean = float(np.mean(y_raw))
        preds = {"Zero-Growth Baseline": np.zeros(len(eval_df)),
                 "Training Mean Baseline": np.full(len(eval_df), gmean),
                 "Persistence Baseline": pd.to_numeric(eval_df["GDP_GROWTH"], errors="coerce").fillna(gmean).to_numpy(dtype=float)}
        cm = fit_df.groupby("iso3")["target_gdp_growth"].mean()
        preds["Country Historical Mean Baseline"] = eval_df["iso3"].map(cm).fillna(gmean).to_numpy(dtype=float)
        slr_med = float(pd.to_numeric(fit_df["GDP_GROWTH"], errors="coerce").median())
        xs_fit = _safe_fill(fit_df["GDP_GROWTH"], gmean)
        xs_eval = pd.to_numeric(eval_df["GDP_GROWTH"], errors="coerce").fillna(slr_med).to_numpy(dtype=float).reshape(-1, 1)
        cl_lo, cl_hi = np.quantile(xs_fit, [0.01, 0.99])
        xs_fit, xs_eval = np.clip(xs_fit, cl_lo, cl_hi), np.clip(xs_eval, cl_lo, cl_hi)
        preds["Simple Linear Regression"] = LinearRegression().fit(xs_fit, y_w).predict(xs_eval)
        poly = PolynomialFeatures(degree=2, include_bias=False)
        preds["Polynomial Regression (degree 2)"] = LinearRegression().fit(poly.fit_transform(xs_fit), y_w).predict(poly.transform(xs_eval))
        preds["Multiple Linear Regression"] = LinearRegression().fit(X_fit, y_w).predict(X_eval)
        preds["Ridge Regression"] = RidgeCV(alphas=np.logspace(-1, 3, 12)).fit(X_fit, y_w).predict(X_eval)
        preds["Huber Regression (robust)"] = HuberRegressor(max_iter=500, alpha=1e-3).fit(X_fit, y_w).predict(X_eval)
        preds["Random Forest Regressor"] = RandomForestRegressor(n_estimators=250, min_samples_leaf=15, max_features=0.4,
                                                                 random_state=RANDOM_SEED, n_jobs=-1).fit(X_fit, y_w).predict(X_eval)
        preds["Gradient Boosting Regressor (Huber loss)"] = GradientBoostingRegressor(
            loss="huber", n_estimators=220, learning_rate=0.04, max_depth=3, subsample=0.8, min_samples_leaf=20,
            random_state=RANDOM_SEED).fit(X_fit, y_w).predict(X_eval)
        return preds

    info("VALIDATION", "Scoring 4 baselines + 7 regression models on validation years ...")
    val_pred = predict_all(fit_train, validation, Xfit, Xval, y_fit)
    val_rmse = {n: rmse_fn(y_val, p) for n, p in val_pred.items()}
    model_only = {n: v for n, v in val_rmse.items() if n not in baseline_names}
    selected = min(model_only, key=model_only.get)
    info("MODEL SELECTION", f"{selected} (validation RMSE {model_only[selected]:.4f}); baselines are references, not candidates.")

    info("FINAL TEST", "Refitting on all pre-test years; scoring the untouched test period once ...")
    test_pred = predict_all(train, test, Xtr, Xte, ytr)
    ex = ~test["target_year"].isin(COVID_YEARS).to_numpy()
    rows = []
    for name, pred in test_pred.items():
        row = regression_metrics(name, yte, pred)
        row["RMSE_ex_COVID"] = rmse_fn(yte[ex], pred[ex]) if ex.sum() > 30 else np.nan
        row["Validation_RMSE"] = val_rmse[name]
        row["Model_Type"] = "Baseline" if name in baseline_names else "Regression model"
        row["Selected_on_Validation"] = bool(name == selected)
        rows.append(row)
    results_df = pd.DataFrame(rows).sort_values("Validation_RMSE").reset_index(drop=True)
    results_df.to_csv(TABLE_DIR / "regression_model_metrics.csv", index=False, encoding="utf-8-sig")
    print("\n" + results_df.round(4).to_string(index=False))

    sel_row = results_df[results_df["Selected_on_Validation"]].iloc[0]
    best_base = results_df[results_df["Model_Type"] == "Baseline"].sort_values("RMSE").iloc[0]
    gain = float(best_base["RMSE"] - sel_row["RMSE"])
    gain_ex = float(results_df[results_df["Model"] == best_base["Model"]]["RMSE_ex_COVID"].iloc[0] - sel_row["RMSE_ex_COVID"])
    info("BASELINE CHECK", f"Selected RMSE {sel_row['RMSE']:.3f} vs best baseline '{best_base['Model']}' {best_base['RMSE']:.3f} -> gain {gain:+.3f} pp ({gain / best_base['RMSE'] * 100:+.1f}%).")
    info("BASELINE CHECK", f"Excluding {COVID_YEARS[0]}-{COVID_YEARS[1]}: gain vs same baseline {gain_ex:+.3f} pp.")
    if gain <= 0:
        info("INTERPRETATION", "The selected model does not beat the strongest naive baseline on final-test RMSE.")
    if sel_row["R2"] < 0:
        info("INTERPRETATION", "Negative test R\u00b2: it does not beat a constant test-set-mean forecast in squared-error terms.")

    _fit_ols_with_diagnostics(train, ytr)
    fig_correlations(train)
    fig_regression_comparison(results_df)
    fig_regression_diagnostics(selected, yte, test_pred[selected], test, test_pred["Zero-Growth Baseline"], test_pred["Persistence Baseline"])

    # Low-growth early-warning classifier (threshold learned on fit years only).
    threshold = float(np.quantile(y_fit, 0.25))
    y_fit_r, y_val_r, ytr_r, yte_r = [(y <= threshold).astype(int) for y in (y_fit, y_val, ytr, yte)]
    info("EARLY WARNING", f"Low-growth threshold (25th pct of fit years): {threshold:.3f}%")
    risk = {}
    if len(np.unique(y_fit_r)) >= 2 and len(np.unique(yte_r)) == 2:
        lr = LogisticRegression(max_iter=1500, class_weight="balanced", random_state=RANDOM_SEED).fit(Xtr, ytr_r)
        rf = RandomForestClassifier(n_estimators=250, min_samples_leaf=15, class_weight="balanced_subsample", random_state=RANDOM_SEED, n_jobs=-1).fit(Xtr, ytr_r)
        p_lr, p_rf = lr.predict_proba(Xte)[:, 1], rf.predict_proba(Xte)[:, 1]
        info("EARLY WARNING", f"Test AUC: logistic {roc_auc_score(yte_r, p_lr):.3f} | random forest {roc_auc_score(yte_r, p_rf):.3f} | F1 (logistic) {f1_score(yte_r, lr.predict(Xte)):.3f}")
        fig_risk_curves(yte_r, p_lr, p_rf, threshold)
        risk = {"auc_lr": float(roc_auc_score(yte_r, p_lr)), "auc_rf": float(roc_auc_score(yte_r, p_rf)), "threshold": threshold}
    else:
        info("EARLY WARNING", "Skipped: a split contains only one risk class.")

    ARTIFACTS["reg_results"] = results_df
    return {"imputer": final["imputer"], "scaler": final["scaler"], "keep": final["keep"], "Xtr": Xtr, "Xte": Xte,
            "selected_features": final["feature_names"], "raw_features": features, "ytr": ytr, "yte": yte, "train": train, "test": test,
            "fit_train": fit_train, "validation": validation, "Xfit": Xfit, "Xval": Xval, "y_fit": y_fit, "y_val": y_val,
            "validation_start": validation_start, "validation_selected_regression_model": selected, "regression_results": results_df,
            "best_baseline_name": str(best_base["Model"]), "baseline_improvement_rmse": gain, "risk": risk}, results_df

# %% 8. CLASSIFICATION: GROWTH CATEGORIES -------------------------------------
def make_growth_classes(reference_values, values):
    q1, q2 = np.quantile(reference_values, [1 / 3, 2 / 3])
    if not q1 < q2:
        median = float(np.median(reference_values))
        return (np.asarray(values) > median).astype(int), ["Lower growth", "Higher growth"], (median,)
    ids = np.digitize(np.asarray(values), [q1, q2], right=True).astype(int)
    return ids, ["Low growth", "Medium growth", "High growth"], (float(q1), float(q2))


def bagging_classifier(tree):
    kwargs = {"n_estimators": 60, "random_state": RANDOM_SEED, "n_jobs": -1}
    key = "estimator" if "estimator" in inspect.signature(BaggingClassifier).parameters else "base_estimator"
    return BaggingClassifier(**{key: tree}, **kwargs)


def calibrated_linear_svm():
    base = LinearSVC(class_weight="balanced", random_state=RANDOM_SEED, max_iter=6000)
    key = "estimator" if "estimator" in inspect.signature(CalibratedClassifierCV).parameters else "base_estimator"
    return CalibratedClassifierCV(**{key: base}, cv=3)


def tune_hyperparameters(Xfit, yfit, Xval, yval):
    """Pick tree depth and KNN neighbours on the chronological validation years only."""
    depth_rows = []
    for d in range(2, 13):
        m = DecisionTreeClassifier(max_depth=d, class_weight="balanced", random_state=RANDOM_SEED).fit(Xfit, yfit)
        depth_rows.append((d, f1_score(yval, m.predict(Xval), average="macro", zero_division=0)))
    best_depth = max(depth_rows, key=lambda t: t[1])[0]
    knn_rows = []
    for k in (5, 11, 21, 35, 51):
        m = KNeighborsClassifier(n_neighbors=k, weights="distance").fit(Xfit, yfit)
        knn_rows.append((k, f1_score(yval, m.predict(Xval), average="macro", zero_division=0)))
    best_k = max(knn_rows, key=lambda t: t[1])[0]
    info("TUNING", f"Validation-selected tree depth = {best_depth}; KNN neighbours = {best_k}")
    return {"tree_depth": best_depth, "knn_k": best_k}


def _classification_models(params):
    return {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced", random_state=RANDOM_SEED),
        "Decision Tree Classifier": DecisionTreeClassifier(max_depth=params["tree_depth"], class_weight="balanced", random_state=RANDOM_SEED),
        "Support Vector Machine (Linear SVM)": calibrated_linear_svm(),
        "K-Nearest Neighbors (KNN)": KNeighborsClassifier(n_neighbors=params["knn_k"], weights="distance"),
        "Gaussian Naive Bayes": GaussianNB(),
        "Feedforward Neural Network (MLP)": MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam", alpha=0.01,
                                                           early_stopping=True, max_iter=300, random_state=RANDOM_SEED),
        "Bagging Classifier (Decision Trees)": bagging_classifier(DecisionTreeClassifier(max_depth=10, random_state=RANDOM_SEED)),
        "Gradient Boosting Classifier": GradientBoostingClassifier(n_estimators=160, learning_rate=0.05, max_depth=3, subsample=0.8,
                                                                     min_samples_leaf=20, random_state=RANDOM_SEED),
        "Random Forest Classifier": RandomForestClassifier(n_estimators=250, min_samples_leaf=5, class_weight="balanced_subsample",
                                                            random_state=RANDOM_SEED, n_jobs=-1),
    }


def _classifier_metrics(model_name, y_true, pred, proba=None, labels=None):
    row = {"Model": model_name, "Accuracy": accuracy_score(y_true, pred),
           "Macro_Precision": precision_score(y_true, pred, average="macro", zero_division=0),
           "Macro_Recall": recall_score(y_true, pred, average="macro", zero_division=0),
           "Macro_F1": f1_score(y_true, pred, average="macro", zero_division=0),
           "Weighted_F1": f1_score(y_true, pred, average="weighted", zero_division=0),
           "Log_Loss": np.nan, "AUC_macro_OvR": np.nan}
    if proba is not None:
        try:
            row["Log_Loss"] = log_loss(y_true, proba, labels=labels)
        except Exception:
            pass
        try:
            if len(np.unique(y_true)) == len(labels) and len(labels) > 2:
                row["AUC_macro_OvR"] = roc_auc_score(y_true, proba, multi_class="ovr", average="macro", labels=labels)
            elif len(labels) == 2 and len(np.unique(y_true)) == 2:
                row["AUC_macro_OvR"] = roc_auc_score(y_true, proba[:, 1])
        except Exception:
            pass
    return row


def fig_classification_comparison(val_df, test_df, selected):
    models = test_df[test_df["Model_Type"] == "Machine-learning model"].copy()
    models = models.sort_values("Validation_Macro_F1", ascending=True).reset_index(drop=True)
    y = np.arange(len(models))
    h = 0.38
    fig, ax = plt.subplots(figsize=(11, 0.62 * len(models) + 2.6))
    ax.barh(y + h / 2, models["Validation_Macro_F1"], h, color=C["sky"], label="Validation")
    ax.barh(y - h / 2, models["Macro_F1"], h, color=[C["navy"] if m == selected else C["blue"] for m in models["Model"]], label="Final test")
    for i, r in models.iterrows():
        ax.text(r["Validation_Macro_F1"] + 0.006, i + h / 2, f"{r['Validation_Macro_F1']:.3f}", va="center", fontsize=7.5, color=C["mid"])
        ax.text(r["Macro_F1"] + 0.006, i - h / 2, f"{r['Macro_F1']:.3f}", va="center", fontsize=7.5, fontweight="bold" if r["Model"] == selected else "normal")
    base = test_df[test_df["Model_Type"] == "Baseline"].set_index("Model")["Macro_F1"]
    styles = {"Previous-Year Category Baseline": (C["red"], "-"), "Majority-Class Baseline": (C["grey"], ":")}
    for nm, v in base.items():
        col, ls = styles.get(nm, (C["grey"], ":"))
        ax.axvline(v, color=col, ls=ls, lw=1.6)
        ax.text(v, len(models) - 0.35, f" {nm.replace(' Baseline', '')}\n {v:.3f}", color=col, fontsize=8, va="bottom")
    ax.set_yticks(y)
    ax.set_yticklabels([("\u2605 " if m == selected else "") + m for m in models["Model"]], fontsize=9)
    ax.set_xlabel("Macro-F1 \u2014 higher is better")
    ax.set_xlim(0, max(models["Macro_F1"].max(), models["Validation_Macro_F1"].max(), base.max()) * 1.2)
    ax.set_ylim(-0.7, len(models) + 0.5)
    ax.legend(handles=[Patch(color=C["sky"], label="Validation"), Patch(color=C["blue"], label="Final test")], loc="lower right")
    style_axis(ax, xgrid=True, ygrid=False)
    finish(fig, "10_classification_scorecard", "Classification scorecard: low / medium / high next-year growth",
           "\u2605 = best ML model on validation. Vertical lines are naive baselines the models must beat to be useful.")


def fig_confusion(cm_counts, label_names, title_model):
    cm_pct = cm_counts / np.maximum(cm_counts.sum(axis=1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(6.4, 5.4))
    ax.imshow(cm_pct, cmap=SEQ, vmin=0, vmax=1)
    ax.set_xticks(range(len(label_names))); ax.set_xticklabels(label_names, fontsize=9)
    ax.set_yticks(range(len(label_names))); ax.set_yticklabels(label_names, fontsize=9)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.grid(False); ax.tick_params(length=0)
    for i in range(len(label_names)):
        for j in range(len(label_names)):
            ax.text(j, i, f"{cm_pct[i, j] * 100:.0f}%\n({cm_counts[i, j]:,})", ha="center", va="center", fontsize=10,
                    fontweight="bold" if i == j else "normal", color="white" if cm_pct[i, j] > 0.5 else C["ink"])
    finish(fig, "11_confusion_matrix", "Where the classifier gets it right and wrong", f"{title_model} \u2014 % of each actual class (counts in brackets), final test.")


def fig_permutation_importance(imp, title_model):
    top = imp.head(14).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, 5.4))
    colors = [C["blue"] if ("growth" in n.lower() or "world" in n.lower()) else C["teal"] for n in top.index]
    ax.barh(range(len(top)), top["mean"], xerr=top["std"], color=colors, error_kw=dict(ecolor=C["grey"], lw=1))
    ax.set_yticks(range(len(top))); ax.set_yticklabels(top.index, fontsize=9)
    ax.set_xlabel("Drop in macro-F1 when the feature is shuffled")
    ax.legend(handles=[Patch(color=C["blue"], label="Growth momentum / world cycle"), Patch(color=C["teal"], label="Structural indicators")], loc="lower right")
    style_axis(ax, xgrid=True, ygrid=False)
    finish(fig, "12_permutation_importance", "What the classifier relies on",
           f"Permutation importance for {title_model} on the final test years (post-hoc explanation only).")


def unit3_classification(model_data):
    banner("CLASSIFICATION: GROWTH CATEGORIES")
    fit_train, validation = model_data["fit_train"], model_data["validation"]
    test = model_data["test"]
    Xfit, Xval, Xtr, Xte = model_data["Xfit"], model_data["Xval"], model_data["Xtr"], model_data["Xte"]
    y_fit_v, y_val_v, ytr_v, yte_v = model_data["y_fit"], model_data["y_val"], model_data["ytr"], model_data["yte"]

    yfit_raw, class_names, thresholds = make_growth_classes(y_fit_v, y_fit_v)
    yval_raw = make_growth_classes(y_fit_v, y_val_v)[0]
    ytr_raw = make_growth_classes(y_fit_v, ytr_v)[0]
    yte_raw = make_growth_classes(y_fit_v, yte_v)[0]
    info("CLASSES", f"Tertile thresholds (fit years only): {tuple(round(t, 2) for t in thresholds)} -> {class_names}")
    fit_classes = np.unique(yfit_raw)
    if len(fit_classes) < 2:
        info("SKIP", "Fewer than two classes in the model-fit period.")
        return None
    valid_val, valid_test = np.isin(yval_raw, fit_classes), np.isin(yte_raw, fit_classes)
    if not valid_val.any() or not valid_test.any():
        info("SKIP", "Validation or test contains no known classes.")
        return None
    Xval_e, yval_e = Xval[valid_val], yval_raw[valid_val]
    Xte_e, yte_e = Xte[valid_test], yte_raw[valid_test]
    val_eval_df, test_eval_df = validation.loc[valid_val].copy(), test.loc[valid_test].copy()
    class_labels = sorted(fit_classes.tolist())
    label_names = [class_names[i] for i in class_labels]
    majority = int(pd.Series(yfit_raw).mode().iloc[0])

    def prev_category(df):
        prior = pd.to_numeric(df["GDP_GROWTH"], errors="coerce").to_numpy(dtype=float)
        pred = np.full(len(prior), majority, dtype=int)
        ok = np.isfinite(prior)
        pred[ok] = (prior[ok] > thresholds[0]).astype(int) if len(thresholds) == 1 else np.digitize(prior[ok], np.asarray(thresholds), right=True)
        return pred

    params = tune_hyperparameters(Xfit, yfit_raw, Xval_e, yval_e)
    val_rows = []
    for nm, pred in {"Majority-Class Baseline": np.full(len(yval_e), majority), "Previous-Year Category Baseline": prev_category(val_eval_df)}.items():
        row = _classifier_metrics(nm, yval_e, pred, None, class_labels)
        row.update({"Model_Type": "Baseline", "Seconds": 0.0})
        val_rows.append(row)
        info("BASELINE", f"{nm}: validation accuracy {row['Accuracy']:.3f}, macro-F1 {row['Macro_F1']:.3f}")

    model_val_rows = []
    models = _classification_models(params)
    for n, (name, model) in enumerate(models.items(), start=1):
        t0 = time.time()
        try:
            model.fit(Xfit, yfit_raw)
            pred = model.predict(Xval_e)
            proba = model.predict_proba(Xval_e) if hasattr(model, "predict_proba") else None
            row = _classifier_metrics(name, yval_e, pred, proba, class_labels)
            row.update({"Model_Type": "Machine-learning model", "Seconds": round(time.time() - t0, 2)})
            val_rows.append(row); model_val_rows.append(row)
            info(f"VALIDATION {n}/{len(models)}", f"{name:<38} acc {row['Accuracy']:.3f} | macro-F1 {row['Macro_F1']:.3f}")
        except Exception as exc:
            info("FAILED", f"{name}: {exc}")
    if not model_val_rows:
        return None
    validation_df = pd.DataFrame(val_rows).sort_values("Macro_F1", ascending=False)
    model_val_df = pd.DataFrame(model_val_rows).sort_values("Macro_F1", ascending=False)
    selected = str(model_val_df.iloc[0]["Model"])
    info("MODEL SELECTION", f"{selected} (validation macro-F1 {model_val_df.iloc[0]['Macro_F1']:.4f})")

    test_rows, fitted, predictions = [], {}, {}
    for nm, pred in {"Majority-Class Baseline": np.full(len(yte_e), majority), "Previous-Year Category Baseline": prev_category(test_eval_df)}.items():
        row = _classifier_metrics(nm, yte_e, pred, None, class_labels)
        row.update({"Validation_Macro_F1": float(validation_df.loc[validation_df["Model"] == nm, "Macro_F1"].iloc[0]),
                    "Model_Type": "Baseline", "Selected_on_Validation": False, "Seconds": 0.0})
        test_rows.append(row); predictions[nm] = pred
    for n, (name, model) in enumerate(_classification_models(params).items(), start=1):
        t0 = time.time()
        try:
            model.fit(Xtr, ytr_raw)
            pred = model.predict(Xte_e)
            proba = model.predict_proba(Xte_e) if hasattr(model, "predict_proba") else None
            row = _classifier_metrics(name, yte_e, pred, proba, class_labels)
            vm = model_val_df.loc[model_val_df["Model"] == name, "Macro_F1"]
            row.update({"Validation_Macro_F1": float(vm.iloc[0]) if len(vm) else np.nan, "Model_Type": "Machine-learning model",
                        "Selected_on_Validation": bool(name == selected), "Seconds": round(time.time() - t0, 2)})
            test_rows.append(row); fitted[name] = model; predictions[name] = pred
            info(f"FINAL TEST {n}/9", f"{name:<38} acc {row['Accuracy']:.3f} | macro-F1 {row['Macro_F1']:.3f}")
        except Exception as exc:
            info("FAILED", f"{name}: {exc}")
    if selected not in fitted:
        info("SKIP", "Selected model did not complete final testing.")
        return None

    validation_df.to_csv(TABLE_DIR / "classification_validation_metrics.csv", index=False, encoding="utf-8-sig")
    model_val_df.to_csv(TABLE_DIR / "classification_model_validation_metrics.csv", index=False, encoding="utf-8-sig")
    result = pd.DataFrame(test_rows).sort_values(["Model_Type", "Validation_Macro_F1"], ascending=[True, False]).reset_index(drop=True)
    result.to_csv(TABLE_DIR / "classification_model_metrics.csv", index=False, encoding="utf-8-sig")
    print("\n" + result.round(4).to_string(index=False))

    sel_final = result[result["Model"] == selected].iloc[0]
    best_base = result[result["Model_Type"] == "Baseline"].sort_values("Macro_F1", ascending=False).iloc[0]
    delta = float(sel_final["Macro_F1"] - best_base["Macro_F1"])
    info("BASELINE CHECK", f"Selected macro-F1 {sel_final['Macro_F1']:.4f} vs strongest baseline '{best_base['Model']}' {best_base['Macro_F1']:.4f} -> {delta:+.4f}")
    if delta <= 0:
        info("INTERPRETATION", "The selected ML classifier does not beat the strongest simple baseline: growth momentum is hard to improve on.")

    best_pred = predictions[selected]
    pd.DataFrame(classification_report(yte_e, best_pred, labels=class_labels, target_names=label_names, output_dict=True, zero_division=0)).T.to_csv(
        TABLE_DIR / "best_classifier_report.csv", encoding="utf-8-sig")
    cm = confusion_matrix(yte_e, best_pred, labels=class_labels)
    fig_classification_comparison(validation_df, result, selected)
    fig_confusion(cm, label_names, selected)
    try:
        pi = permutation_importance(fitted[selected], Xte_e, yte_e, n_repeats=5, scoring="f1_macro", random_state=RANDOM_SEED, n_jobs=1)
        imp = pd.DataFrame({"mean": pi.importances_mean, "std": pi.importances_std}, index=[pretty(f) for f in model_data["selected_features"]])
        imp = imp.sort_values("mean", ascending=False)
        imp.to_csv(TABLE_DIR / "permutation_importance_selected_classifier.csv", encoding="utf-8-sig")
        fig_permutation_importance(imp, selected)
    except Exception as exc:
        info("PERM. IMPORTANCE", f"Skipped: {exc}")
    try:
        tree = fitted.get("Decision Tree Classifier")
        if tree is not None:
            (REPORT_DIR / "decision_tree_rules.txt").write_text(export_text(tree, feature_names=model_data["selected_features"], max_depth=5), encoding="utf-8")
    except Exception as exc:
        info("TREE RULES", f"Could not export: {exc}")

    ARTIFACTS["class_results"] = result
    return {"results": result, "validation_results": validation_df, "model_validation_results": model_val_df, "fitted": fitted,
            "predictions": predictions, "best_name": selected, "yfit": yfit_raw, "yval": yval_e, "ytr": ytr_raw, "yte": yte_e,
            "Xfit": Xfit, "Xval": Xval_e, "Xtr": Xtr, "Xte": Xte_e, "class_names": label_names, "class_labels": class_labels,
            "selected_by_validation": selected, "baseline_best_name": str(best_base["Model"]), "baseline_macro_f1_delta": delta, "params": params}

# %% 9. CLUSTERING AND PATTERN DISCOVERY --------------------------------------
def make_quantile_category(series, prefix):
    valid = series.dropna()
    if len(valid) < 10:
        return pd.Series(index=series.index, dtype=object)
    q1, q2 = valid.quantile([1 / 3, 2 / 3]).values
    labels = pd.Series(index=series.index, dtype=object)
    if q1 >= q2:
        med = valid.median()
        labels.loc[series <= med] = "Low"
        labels.loc[series > med] = "High"
    else:
        labels.loc[series <= q1] = "Low"
        labels.loc[(series > q1) & (series <= q2)] = "Medium"
        labels.loc[series > q2] = "High"
    return prefix + "=" + labels


def prepare_clustering_features(frame, columns):
    X = frame[columns].apply(pd.to_numeric, errors="coerce").copy()
    for col in X.columns:
        if col == "GDP_PER_CAPITA_REAL_USD":
            X[col] = np.log1p(X[col].clip(lower=0))
        elif col in {"GDP_GROWTH", "GDP_PER_CAPITA_GROWTH", "INFLATION", "FDI_PCT_GDP", "CO2_PER_CAPITA"}:
            X[col] = np.sign(X[col]) * np.log1p(np.abs(X[col]))
    X = X.clip(lower=X.quantile(0.02), upper=X.quantile(0.98), axis=1)
    X_imp = SimpleImputer(strategy="median", keep_empty_features=True).fit_transform(X)
    return RobustScaler(quantile_range=(10, 90)).fit_transform(X_imp)


def latest_snapshot(panel, keys, window=SNAPSHOT_WINDOW_YEARS):
    """Latest available value per country within the last `window` years (GDP growth = window average)."""
    counts = panel[panel["GDP_GROWTH"].notna()].groupby("year")["iso3"].nunique()
    ref_year = int(counts[counts >= 120].index.max())
    w = panel[(panel["year"] <= ref_year) & (panel["year"] > ref_year - window)].sort_values(["iso3", "year"])
    snap = w.groupby("iso3")[keys].last()
    snap["GDP_GROWTH"] = w.groupby("iso3")["GDP_GROWTH"].mean()
    meta_cols = [c for c in ["country", "region", "income_level"] if c in panel.columns]
    meta = panel.drop_duplicates("iso3").set_index("iso3")[meta_cols]
    snap = snap.join(meta).reset_index()
    snap["year"] = ref_year
    return snap, ref_year


def name_clusters(profiles):
    k = len(profiles)
    order = profiles["median_gdp_per_capita_real_usd"].sort_values().index.tolist() if "median_gdp_per_capita_real_usd" in profiles.columns else list(profiles.index)
    labels = {2: ["Lower-income", "Higher-income"], 3: ["Lower-income", "Middle-income", "High-income"],
              4: ["Low-income", "Lower-middle", "Upper-middle", "High-income"]}.get(k, [f"Tier {i + 1}" for i in range(k)])
    return {c: labels[i] for i, c in enumerate(order)}, {c: TIER_COLORS[i % len(TIER_COLORS)] for i, c in enumerate(order)}


def fig_kmeans_diagnostics(validation, best_k, ref_year):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.4))
    a1.plot(validation["k"], validation["inertia"], marker="o", color=C["blue"], lw=2.2)
    a1.set_xlabel("Number of clusters (k)"); a1.set_ylabel("Within-cluster sum of squares"); a1.set_title("Elbow curve")
    a2.plot(validation["k"], validation["silhouette"], marker="s", color=C["teal"], lw=2.2)
    a2.axhline(0.25, color=C["grey"], ls=":", lw=1); a2.text(validation["k"].min(), 0.255, "weak-structure line (0.25)", fontsize=8, color=C["grey"])
    a2.set_xlabel("Number of clusters (k)"); a2.set_ylabel("Silhouette"); a2.set_title("Silhouette by k")
    bad = validation[~validation["valid_cluster_sizes"]]
    for _, r in bad.iterrows():
        a2.scatter([r["k"]], [r["silhouette"]], marker="x", color=C["red"], s=70, zorder=5)
    if best_k is not None:
        for a in (a1, a2):
            a.axvline(best_k, color=C["orange"], lw=1.6, ls="--")
        a2.text(best_k + 0.05, a2.get_ylim()[0] + 0.01, f"chosen k = {best_k}", color=C["orange"], fontsize=9, fontweight="bold")
    if len(bad):
        a2.legend(handles=[Line2D([0], [0], marker="x", color=C["red"], lw=0, label=f"rejected: cluster < {MIN_CLUSTER_SHARE:.0%}")], loc="upper right")
    for a in (a1, a2):
        style_axis(a)
    finish(fig, "13_kmeans_model_selection", "Choosing the number of country groups", f"{ref_year} snapshot. Candidates with a tiny cluster are rejected regardless of silhouette.")


def fig_cluster_map(subset, Xs, labels, names, colors, features, ref_year):
    pca = PCA(n_components=2, random_state=RANDOM_SEED)
    Z = pca.fit_transform(Xs)
    subset["PC1"], subset["PC2"] = Z[:, 0], Z[:, 1]
    fig, ax = plt.subplots(figsize=(11, 7))
    for c in sorted(np.unique(labels)):
        m = labels == c
        pts = Z[m]
        if len(pts) >= 4:
            try:
                hull = ConvexHull(pts)
                poly = pts[hull.vertices]
                ax.fill(poly[:, 0], poly[:, 1], color=colors[c], alpha=0.10, lw=0)
                ax.plot(*np.append(poly, poly[:1], axis=0).T, color=colors[c], lw=1, alpha=0.5)
            except Exception:
                pass
        ax.scatter(pts[:, 0], pts[:, 1], s=46, color=colors[c], alpha=0.85, edgecolor="white", lw=0.6, label=f"{names[c]} (n={m.sum()})")
    for iso, (x, y) in zip(subset["iso3"], Z):
        if iso in LABEL_COUNTRIES:
            ax.annotate(iso, (x, y), xytext=(4, 4), textcoords="offset points", fontsize=7.5, color=C["ink"], fontweight="bold")
    ax.axhline(0, color=C["grey"], lw=0.6); ax.axvline(0, color=C["grey"], lw=0.6)
    ev = pca.explained_variance_ratio_ * 100
    ax.set_xlabel(f"PC1 ({ev[0]:.0f}% of variance)"); ax.set_ylabel(f"PC2 ({ev[1]:.0f}% of variance)")
    ax.legend(loc="best", title="K-Means group")
    ax.grid(False)
    finish(fig, "14_country_cluster_map", "A map of development profiles", f"Countries projected onto the first two principal components of {len(features)} standardised indicators ({ref_year}).")


def fig_cluster_profiles(Xs, labels, profiles, names, features):
    z = pd.DataFrame(Xs, columns=features).groupby(labels).median()
    order = list(names.keys())
    order.sort(key=lambda c: list(names.values()).index(names[c]))
    z = z.loc[order]
    raw_cols = [f"median_{c.lower()}" for c in features]
    raw = profiles.loc[order, raw_cols]
    fig, ax = plt.subplots(figsize=(1.15 * len(features) + 3.2, 1.0 * len(order) + 2.6))
    lim = max(0.5, np.abs(z.values).max())
    ax.imshow(z.values, cmap=DIV, vmin=-lim, vmax=lim, aspect="auto")
    ax.set_xticks(range(len(features))); ax.set_xticklabels([pretty(f).replace(" ", "\n", 1) for f in features], fontsize=8.5)
    ax.set_yticks(range(len(order))); ax.set_yticklabels([f"{names[c]}\n(n={int(profiles.loc[c, 'countries'])})" for c in order], fontsize=9.5)
    ax.xaxis.tick_top(); ax.grid(False); ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)
    for i in range(len(order)):
        for j in range(len(features)):
            v = raw.iloc[i, j]
            txt = f"{v:,.0f}" if abs(v) >= 1000 else f"{v:.1f}"
            ax.text(j, i, txt, ha="center", va="center", fontsize=9, fontweight="bold", color="white" if abs(z.values[i, j]) > 0.65 * lim else C["ink"])
    finish(fig, "15_cluster_profiles", "What defines each group?", "Colour = robust z-score vs all countries (blue above, red below). Numbers = group median in original units.")


def fig_cluster_regions(subset, names, colors):
    if "region" not in subset.columns:
        return
    tab = pd.crosstab(subset["region"].fillna("Unknown"), subset["cluster"])
    if tab.empty:
        return
    share = tab.div(tab.sum(axis=1), axis=0)
    share = share.loc[share.sum(axis=1).sort_values().index]
    order = sorted(share.columns, key=lambda c: list(names.values()).index(names[c]))
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    left = np.zeros(len(share))
    for c in order:
        ax.barh(range(len(share)), share[c].values, left=left, color=colors[c], label=names[c], edgecolor="white", lw=0.8)
        for i, (l, v) in enumerate(zip(left, share[c].values)):
            if v >= 0.12:
                ax.text(l + v / 2, i, f"{v * 100:.0f}%", ha="center", va="center", color="white", fontsize=8.5, fontweight="bold")
        left += share[c].values
    ax.set_yticks(range(len(share)))
    ax.set_yticklabels([f"{r}  (n={int(tab.loc[r].sum())})" for r in share.index], fontsize=9)
    ax.set_xlim(0, 1); ax.set_xlabel("Share of countries in region")
    ax.legend(ncol=len(order), loc="lower center", bbox_to_anchor=(0.5, 1.0))
    style_axis(ax, xgrid=True, ygrid=False)
    finish(fig, "16_cluster_by_region", "Do development groups follow geography?", "Composition of each World Bank region by K-Means group.")


def fig_init_comparison(random_inertia, pp_inertia, best_k):
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    ax.bar(range(len(random_inertia)), random_inertia, color=C["sky"], edgecolor="white", label="Random init (n_init=1)")
    ax.axhline(pp_inertia, color=C["red"], lw=2, ls="--", label="k-means++ (n_init=50)")
    ax.set_ylim(min(min(random_inertia), pp_inertia) * 0.97, max(random_inertia) * 1.02)
    ax.set_xlabel("Random seed"); ax.set_ylabel("Inertia (lower is better)")
    ax.legend(loc="upper right")
    style_axis(ax)
    finish(fig, "17_kmeans_initialisation", "Why initialisation matters", f"Single random starts (k={best_k}) land on different local optima; k-means++ with restarts finds the best one.")


def fig_hierarchical(linkage_scores, Z_best, best_linkage):
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={"width_ratios": [1, 2.4]})
    names = list(linkage_scores.keys())
    vals = [linkage_scores[n] for n in names]
    a1.bar(names, vals, color=[C["navy"] if n == best_linkage else C["sky"] for n in names])
    for i, v in enumerate(vals):
        a1.text(i, v + 0.01, f"{v:.3f}", ha="center", fontsize=8.5)
    a1.set_ylim(0, 1); a1.set_ylabel("Cophenetic correlation"); a1.set_title("Linkage quality")
    style_axis(a1)
    thr = 0.55 * Z_best[:, 2].max()
    dendrogram(Z_best, truncate_mode="lastp", p=30, no_labels=True, color_threshold=thr, above_threshold_color=C["grey"], ax=a2)
    a2.axhline(thr, color=C["red"], ls="--", lw=1)
    a2.set_title(f"Dendrogram ({best_linkage} linkage, country sample)"); a2.set_ylabel("Distance"); a2.set_xlabel("Merged groups")
    a2.grid(False)
    finish(fig, "18_hierarchical_clustering", "Hierarchical clustering", "Cophenetic correlation measures how faithfully the tree preserves the original pairwise distances.")


def fig_rules(rules):
    top = rules.head(10).iloc[::-1]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1, 1.5]})
    sc = a1.scatter(rules["support"], rules["confidence"], s=18 + rules["lift"] * 22, c=rules["lift"], cmap=SEQ, edgecolor=C["navy"], lw=0.4, alpha=0.9)
    a1.set_xlabel("Support"); a1.set_ylabel("Confidence"); a1.set_title("Rule landscape (size & colour = lift)")
    cb = fig.colorbar(sc, ax=a1, fraction=0.045, pad=0.02); cb.set_label("Lift"); cb.outline.set_visible(False)
    style_axis(a1)
    labels = [textwrap.shorten(f"{a}  \u2192  {c}".replace("=", ": "), 78, placeholder="\u2026") for a, c in zip(top["antecedents_text"], top["consequents_text"])]
    a2.barh(range(len(top)), top["lift"], color=C["teal"])
    a2.set_yticks(range(len(top))); a2.set_yticklabels(labels, fontsize=7.8)
    for i, (l, cf) in enumerate(zip(top["lift"], top["confidence"])):
        a2.text(l + 0.04, i, f"lift {l:.2f} \u00b7 conf {cf:.0%}", va="center", fontsize=7.8)
    a2.set_xlim(0, top["lift"].max() * 1.35); a2.set_xlabel("Lift"); a2.set_title("Strongest development patterns")
    style_axis(a2, xgrid=True, ygrid=False)
    finish(fig, "19_association_rules", "Association rules: what tends to go together?", "Apriori on tercile-binned indicators from the country snapshot. Lift > 1 = co-occurs more than chance. Co-occurrence, not causation.")


def unit4_clustering(panel):
    banner("COUNTRY SEGMENTATION AND PATTERN DISCOVERY")
    candidate_keys = ["GDP_GROWTH", "GDP_PER_CAPITA_REAL_USD", "URBAN_POP_PCT", "INFLATION", "UNEMPLOYMENT", "LIFE_EXPECTANCY",
                      "INTERNET_USERS_PCT", "EXPORTS_PCT_GDP", "IMPORTS_PCT_GDP", "FDI_PCT_GDP", "CO2_PER_CAPITA"]
    avail = [c for c in candidate_keys if c in panel.columns]
    if len(avail) < 5 or "GDP_GROWTH" not in avail:
        info("SKIP", "Fewer than five candidate indicators.")
        return None
    snap, ref_year = latest_snapshot(panel, avail)
    cov = snap[avail].notna().mean()
    cols = [c for c in avail if cov[c] >= 0.65]
    if len(cols) < 5:
        info("SKIP", "Fewer than five features reach 65% coverage in the snapshot.")
        return None
    min_obs = max(5, int(np.ceil(len(cols) * 0.70)))
    subset = snap[snap[cols].notna().sum(axis=1) >= min_obs].drop_duplicates("iso3").reset_index(drop=True)
    if len(subset) < 60:
        info("SKIP", "Fewer than 60 reasonably complete country profiles.")
        return None
    info("SNAPSHOT", f"Reference year {ref_year} (latest value within {SNAPSHOT_WINDOW_YEARS} yrs; GDP growth = {SNAPSHOT_WINDOW_YEARS}-yr mean) | {len(subset)} countries")
    info("FEATURES", ", ".join(pretty(c) for c in cols))
    subset[cols].notna().mean().rename("observed_share").to_frame().to_csv(TABLE_DIR / "clustering_feature_coverage.csv", encoding="utf-8-sig")

    Xs = prepare_clustering_features(subset, cols)
    rng = np.random.RandomState(RANDOM_SEED)
    sample_idx = rng.choice(len(Xs), size=min(MAX_CLUSTER_SAMPLE, len(Xs)), replace=False)
    Xsample = Xs[sample_idx]
    rows = []
    for k in range(2, min(7, len(Xsample) - 1)):
        km = KMeans(n_clusters=k, init="k-means++", n_init=50, random_state=RANDOM_SEED)
        lab = km.fit_predict(Xsample)
        counts = np.bincount(lab, minlength=k)
        rows.append({"k": k, "inertia": float(km.inertia_), "silhouette": float(silhouette_score(Xsample, lab)),
                     "smallest_cluster_share": float(counts.min() / len(lab)), "smallest_cluster_n": int(counts.min()),
                     "valid_cluster_sizes": bool(counts.min() / len(lab) >= MIN_CLUSTER_SHARE and counts.min() >= 5)})
    validation = pd.DataFrame(rows)
    validation.to_csv(TABLE_DIR / "kmeans_cluster_validation.csv", index=False, encoding="utf-8-sig")
    print(validation.round(4).to_string(index=False))
    eligible = validation[validation["valid_cluster_sizes"]]
    best_k, best_sil, km_final = None, np.nan, None
    profiles, stability = pd.DataFrame(), pd.DataFrame()
    names, colors = {}, {}

    if eligible.empty:
        info("CLUSTER WARNING", f"No K-Means solution met the minimum {MIN_CLUSTER_SHARE:.0%} cluster-size rule; no segmentation published.")
    else:
        best = eligible.sort_values("silhouette", ascending=False).iloc[0]
        best_k, best_sil = int(best["k"]), float(best["silhouette"])
        info("K-MEANS", f"k = {best_k} | silhouette {best_sil:.4f} | smallest cluster {best['smallest_cluster_share']:.1%}")
        if best_sil < 0.25:
            info("CLUSTER CHECK", "Weak-to-moderate separation: development profiles form a continuum more than sharp groups.")
        ref_lab = KMeans(n_clusters=best_k, init="k-means++", n_init=50, random_state=RANDOM_SEED).fit_predict(Xsample)
        stab = [{"random_seed": s, "adjusted_rand_vs_reference": adjusted_rand_score(
            ref_lab, KMeans(n_clusters=best_k, init="k-means++", n_init=50, random_state=s).fit_predict(Xsample))} for s in [7, 21, 42, 84, 168, 336, 672, 1344]]
        stability = pd.DataFrame(stab)
        stability.to_csv(TABLE_DIR / "kmeans_seed_stability.csv", index=False, encoding="utf-8-sig")
        info("STABILITY", f"Mean adjusted Rand agreement across seeds: {stability['adjusted_rand_vs_reference'].mean():.4f}")
        km_final = KMeans(n_clusters=best_k, init="k-means++", n_init=100, random_state=RANDOM_SEED)
        subset["cluster"] = km_final.fit_predict(Xs)
        shares = subset["cluster"].value_counts(normalize=True)
        if float(shares.min()) < MIN_CLUSTER_SHARE:
            info("CLUSTER SAFETY", "Full-snapshot fit violated the minimum cluster-size rule; labels discarded.")
            subset = subset.drop(columns=["cluster"]); km_final = None; best_k, best_sil = None, np.nan
        else:
            profiles = subset.groupby("cluster").agg(countries=("iso3", "nunique"), snapshot_year=("year", "max"),
                                                    **{f"median_{c.lower()}": (c, "median") for c in cols})
            names, colors = name_clusters(profiles)
            subset["cluster_name"] = subset["cluster"].map(names)
            subset[["iso3", "country", "year", "cluster", "cluster_name"]].to_csv(TABLE_DIR / "kmeans_country_snapshot_assignments.csv", index=False, encoding="utf-8-sig")
            profiles.to_csv(TABLE_DIR / "kmeans_country_snapshot_profiles.csv", encoding="utf-8-sig")
            print(profiles.round(2).rename(index=names).to_string())

    fig_kmeans_diagnostics(validation, best_k, ref_year)
    if km_final is not None:
        labels = subset["cluster"].to_numpy()
        fig_cluster_map(subset, Xs, labels, names, colors, cols, ref_year)
        fig_cluster_profiles(Xs, labels, profiles, names, cols)
        fig_cluster_regions(subset, names, colors)
        rnd = [float(KMeans(n_clusters=best_k, init="random", n_init=1, random_state=s).fit(Xsample).inertia_) for s in range(10)]
        pp = float(KMeans(n_clusters=best_k, init="k-means++", n_init=50, random_state=RANDOM_SEED).fit(Xsample).inertia_)
        pd.DataFrame({"random_state": range(10), "random_init_inertia": rnd}).to_csv(TABLE_DIR / "kmeans_random_initialization_comparison.csv", index=False)
        fig_init_comparison(rnd, pp, best_k)

    h_idx = rng.choice(len(Xs), size=min(MAX_HIERARCHICAL_SAMPLE, len(Xs)), replace=False)
    Xh = Xs[h_idx]
    dist = pdist(Xh)
    scores, mats = {}, {}
    for method in ["single", "complete", "average", "centroid", "ward"]:
        try:
            Z = linkage(Xh, method=method)
            mats[method], scores[method] = Z, float(cophenet(Z, dist)[0])
        except Exception as exc:
            info("LINKAGE FAILED", f"{method}: {exc}")
    if scores:
        pd.Series(scores, name="cophenetic_correlation").to_csv(TABLE_DIR / "hierarchical_linkage_comparison.csv", encoding="utf-8-sig")
        best_link = max(scores, key=scores.get)
        info("HIERARCHICAL", "Cophenetic correlations: " + ", ".join(f"{k} {v:.3f}" for k, v in scores.items()))
        fig_hierarchical(scores, mats[best_link], best_link)

    rules_df = None
    if MLXTEND_AVAILABLE:
        basket_df = pd.DataFrame(index=subset.index)
        for col, label in [("GDP_GROWTH", "GDP growth"), ("INFLATION", "Inflation"), ("UNEMPLOYMENT", "Unemployment"), ("INTERNET_USERS_PCT", "Internet use"),
                           ("LIFE_EXPECTANCY", "Life expectancy"), ("URBAN_POP_PCT", "Urban population"), ("GDP_PER_CAPITA_REAL_USD", "Real GDP per capita")]:
            if col in subset.columns and subset[col].notna().sum() >= 10:
                basket_df[label] = make_quantile_category(subset[col], label)
        if "income_level" in subset.columns:
            basket_df["Income level"] = "Income=" + subset["income_level"].fillna("Unknown").astype(str)
        basket_df = basket_df.dropna(how="all")
        trans = [[str(v) for v in row if pd.notna(v) and str(v) not in ("", "nan")] for row in basket_df.to_numpy()]
        items = sorted({i for t in trans for i in t})
        basket = pd.DataFrame(False, index=range(len(trans)), columns=items)
        for i, t in enumerate(trans):
            if t:
                basket.loc[i, t] = True
        try:
            frequent = apriori(basket, min_support=0.04, use_colnames=True, max_len=3)
            if not frequent.empty:
                try:
                    rules = association_rules(frequent, metric="lift", min_threshold=1.15, num_itemsets=len(basket))
                except TypeError:
                    rules = association_rules(frequent, metric="lift", min_threshold=1.15)
                if not rules.empty:
                    rules["antecedents_text"] = rules["antecedents"].apply(lambda x: ", ".join(sorted(x)))
                    rules["consequents_text"] = rules["consequents"].apply(lambda x: ", ".join(sorted(x)))
                    rules = rules.sort_values("lift", ascending=False).reset_index(drop=True)
                    out_cols = ["antecedents_text", "consequents_text", "support", "confidence", "lift"]
                    rules[out_cols].head(100).to_csv(TABLE_DIR / "association_rules_top100.csv", index=False, encoding="utf-8-sig")
                    print(rules[out_cols].head(8).round(3).to_string(index=False))
                    rules_df = rules
                    fig_rules(rules.head(150))
                else:
                    info("APRIORI", "No rules met the support/lift thresholds.")
            else:
                info("APRIORI", "No frequent itemsets at the chosen support.")
        except Exception as exc:
            info("APRIORI WARNING", f"Pattern mining failed: {exc}")
    else:
        info("APRIORI", "Skipped (mlxtend not installed).")

    result = {"subset": subset, "features": cols, "scaled": Xs, "kmeans": km_final, "validation": validation, "profiles": profiles,
              "stability": stability, "reference_year": ref_year, "silhouette": best_sil, "cluster_count": best_k,
              "cluster_valid": km_final is not None, "names": names, "colors": colors, "rules": rules_df}
    ARTIFACTS["cluster"] = result
    return result

# %% 10. PCA AND NEURAL NETWORK -----------------------------------------------
def unit5_pca_mlp(model_data, classification_data):
    banner("DIMENSIONALITY REDUCTION AND NEURAL NETWORK")
    if model_data is None:
        info("SKIP", "No model data.")
        return
    X_train, features = model_data["Xtr"], model_data["selected_features"]
    n_comp = min(X_train.shape[1], X_train.shape[0] - 1)
    pca = PCA(n_components=n_comp, random_state=RANDOM_SEED).fit(X_train)
    evr = pca.explained_variance_ratio_
    cum = np.cumsum(evr)
    n80, n90 = int(np.searchsorted(cum, 0.80) + 1), int(np.searchsorted(cum, 0.90) + 1)
    info("PCA", f"PC1 {evr[0] * 100:.1f}% | PC2 {evr[1] * 100:.1f}% | 80% variance needs {n80} comps, 90% needs {n90} (pre-test data only)")
    pd.DataFrame({"component": np.arange(1, len(evr) + 1), "explained_variance_ratio": evr, "cumulative_variance": cum}).to_csv(TABLE_DIR / "pca_explained_variance.csv", index=False)
    load = pd.DataFrame(pca.components_.T, index=[pretty(f) for f in features], columns=[f"PC{i + 1}" for i in range(pca.components_.shape[0])])
    load.to_csv(TABLE_DIR / "pca_feature_loadings.csv", encoding="utf-8-sig")

    show_n = min(25, len(evr))
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 6), gridspec_kw={"width_ratios": [1.15, 1]})
    a1.bar(range(1, show_n + 1), evr[:show_n] * 100, color=C["sky"], label="Per component")
    a1.set_ylabel("Variance explained (%)"); a1.set_xlabel("Principal component")
    a1b = a1.twinx()
    a1b.plot(range(1, show_n + 1), cum[:show_n] * 100, color=C["navy"], lw=2.2, marker="o", ms=3.5)
    a1b.axhline(80, color=C["orange"], ls="--", lw=1); a1b.axhline(90, color=C["red"], ls=":", lw=1)
    a1b.text(show_n, 81, "80%", color=C["orange"], ha="right", fontsize=8.5); a1b.text(show_n, 91, "90%", color=C["red"], ha="right", fontsize=8.5)
    a1b.set_ylabel("Cumulative (%)"); a1b.set_ylim(0, 102); a1b.spines["right"].set_visible(True); a1b.grid(False)
    a1.set_title("Scree plot")
    style_axis(a1)
    top_feats = load.iloc[:, :4].abs().max(axis=1).sort_values(ascending=False).head(14).index
    sub = load.loc[top_feats, [c for c in load.columns[:4]]]
    lim = np.abs(sub.values).max()
    im = a2.imshow(sub.values, cmap=DIV, vmin=-lim, vmax=lim, aspect="auto")
    a2.set_xticks(range(sub.shape[1])); a2.set_xticklabels(sub.columns)
    a2.set_yticks(range(sub.shape[0])); a2.set_yticklabels(sub.index, fontsize=8.5)
    a2.grid(False); a2.tick_params(length=0); a2.set_title("What each component measures")
    for sp in a2.spines.values():
        sp.set_visible(False)
    for i in range(sub.shape[0]):
        for j in range(sub.shape[1]):
            a2.text(j, i, f"{sub.values[i, j]:.2f}", ha="center", va="center", fontsize=7.5, color="white" if abs(sub.values[i, j]) > 0.6 * lim else C["ink"])
    finish(fig, "20_pca_scree_and_loadings", "Dimensionality reduction (PCA)", f"{n80} components capture 80% of variance and {n90} capture 90%: the indicators are only moderately redundant.")

    if classification_data is None:
        info("MLP", "Skipped: classification stage did not complete.")
        return
    Xfit, Xval, yfit, yval = classification_data["Xfit"], classification_data["Xval"], classification_data["yfit"], classification_data["yval"]
    Xtr, Xte, ytr, yte = classification_data["Xtr"], classification_data["Xte"], classification_data["ytr"], classification_data["yte"]
    rows = []
    for alpha in [1e-5, 1e-3, 1e-1, 1.0, 3.0]:
        try:
            m = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam", alpha=alpha, early_stopping=True,
                              max_iter=300, random_state=RANDOM_SEED).fit(Xfit, yfit)
            pred = m.predict(Xval)
            rows.append({"alpha": alpha, "validation_accuracy": accuracy_score(yval, pred),
                         "validation_macro_F1": f1_score(yval, pred, average="macro", zero_division=0), "iterations": getattr(m, "n_iter_", np.nan)})
        except Exception as exc:
            info("MLP", f"alpha={fmt_alpha(alpha)} failed: {exc}")
    if not rows:
        return
    mlp_df = pd.DataFrame(rows)
    sel_alpha = float(mlp_df.sort_values("validation_macro_F1", ascending=False).iloc[0]["alpha"])
    final_mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation="relu", solver="adam", alpha=sel_alpha, early_stopping=True,
                              max_iter=300, random_state=RANDOM_SEED).fit(Xtr, ytr)
    tp = final_mlp.predict(Xte)
    test_row = {"selected_alpha": sel_alpha, "test_accuracy": accuracy_score(yte, tp), "test_macro_F1": f1_score(yte, tp, average="macro", zero_division=0),
                "test_weighted_F1": f1_score(yte, tp, average="weighted", zero_division=0)}
    mlp_df.to_csv(TABLE_DIR / "mlp_regularization_validation_comparison.csv", index=False)
    pd.DataFrame([test_row]).to_csv(TABLE_DIR / "mlp_selected_alpha_final_test.csv", index=False)
    info("MLP", f"Validation-selected alpha = {fmt_alpha(sel_alpha)} | test accuracy {test_row['test_accuracy']:.3f} | macro-F1 {test_row['test_macro_F1']:.3f}")
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.semilogx(mlp_df["alpha"], mlp_df["validation_macro_F1"], marker="o", color=C["purple"], lw=2.3, label="Validation macro-F1")
    ax.semilogx(mlp_df["alpha"], mlp_df["validation_accuracy"], marker="s", color=C["sky"], lw=1.6, ls="--", label="Validation accuracy")
    ax.scatter([sel_alpha], [mlp_df.loc[mlp_df["alpha"] == sel_alpha, "validation_macro_F1"].iloc[0]], s=190, facecolor="none", edgecolor=C["orange"], lw=2.2, zorder=5)
    ax.set_xticks(mlp_df["alpha"]); ax.set_xticklabels([fmt_alpha(a) for a in mlp_df["alpha"]])
    ax.set_xlabel("L2 regularisation strength (alpha)"); ax.set_ylabel("Score"); ax.legend(loc="best")
    style_axis(ax)
    finish(fig, "21_mlp_regularisation", "Neural network: how much regularisation?", f"Circled = validation choice (alpha = {fmt_alpha(sel_alpha)}). Stronger regularisation helps when features are noisy.")


# %% 11. GENERALISATION AND ENSEMBLES -----------------------------------------
def unit6_ensembles(classification_data, model_data):
    banner("GENERALISATION, BIAS-VARIANCE AND ENSEMBLES")
    if classification_data is None:
        info("SKIP", "Classification stage did not complete.")
        return
    Xfit, Xval, yfit, yval = classification_data["Xfit"], classification_data["Xval"], classification_data["yfit"], classification_data["yval"]
    if len(Xfit) < 30 or len(Xval) < 20:
        info("SKIP", "Too few chronological observations.")
        return
    depths = list(range(1, 16))
    tr_err, va_err = [], []
    for d in depths:
        m = DecisionTreeClassifier(max_depth=d, random_state=RANDOM_SEED, class_weight="balanced").fit(Xfit, yfit)
        tr_err.append(1 - accuracy_score(yfit, m.predict(Xfit)))
        va_err.append(1 - accuracy_score(yval, m.predict(Xval)))
    pd.DataFrame({"max_depth": depths, "training_error": tr_err, "validation_error": va_err}).to_csv(TABLE_DIR / "bias_variance_tree_depth.csv", index=False)
    best_depth = depths[int(np.argmin(va_err))]
    info("BIAS-VARIANCE", f"Lowest validation error at max_depth = {best_depth}")

    rf = classification_data["fitted"].get("Random Forest Classifier")
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(14, 5.2), gridspec_kw={"width_ratios": [1.1, 1]})
    a1.plot(depths, np.array(tr_err) * 100, marker="o", color=C["blue"], lw=2.2, label="Model-fit error")
    a1.plot(depths, np.array(va_err) * 100, marker="s", color=C["red"], lw=2.2, label="Validation error")
    a1.fill_between(depths, np.array(tr_err) * 100, np.array(va_err) * 100, where=np.array(va_err) > np.array(tr_err), color=C["red"], alpha=0.07, label="generalisation gap")
    a1.axvline(best_depth, color=C["orange"], ls="--", lw=1.6)
    ymax = max(max(tr_err), max(va_err)) * 100
    a1.text(best_depth + 0.15, ymax * 0.97, f"best depth = {best_depth}", color=C["orange"], fontweight="bold", fontsize=9)
    a1.text(1.1, ymax * 0.5, "high bias\n(underfit)", color=C["mid"], fontsize=8.5)
    a1.text(depths[-1] - 0.2, ymax * 0.55, "high variance\n(overfit)", color=C["mid"], fontsize=8.5, ha="right")
    a1.set_xlabel("Decision tree max_depth"); a1.set_ylabel("Classification error (%)"); a1.set_title("Bias\u2013variance trade-off")
    a1.set_ylim(0, ymax * 1.12)
    a1.legend(loc="lower left")
    style_axis(a1)
    if rf is not None and model_data is not None:
        imp = pd.Series(rf.feature_importances_, index=[pretty(f) for f in model_data["selected_features"]]).sort_values(ascending=False)
        imp.to_csv(TABLE_DIR / "random_forest_feature_importance.csv", header=["importance"], encoding="utf-8-sig")
        ARTIFACTS["rf_importance"] = imp
        top = imp.head(14).iloc[::-1]
        cols = [C["blue"] if ("growth" in n.lower() or "world" in n.lower()) else C["teal"] for n in top.index]
        a2.barh(range(len(top)), top.values, color=cols)
        a2.set_yticks(range(len(top))); a2.set_yticklabels(top.index, fontsize=8.8)
        a2.set_xlabel("Impurity-based importance"); a2.set_title("Random Forest: top predictors")
        a2.legend(handles=[Patch(color=C["blue"], label="Growth momentum / world cycle"), Patch(color=C["teal"], label="Structural")], loc="lower right")
        style_axis(a2, xgrid=True, ygrid=False)
        info("RANDOM FOREST", "Top features: " + ", ".join(f"{n} ({v:.3f})" for n, v in imp.head(5).items()))
    else:
        a2.axis("off")
    finish(fig, "22_bias_variance_and_forest", "Generalisation and ensembles", "Left: why shallow-but-not-trivial trees generalise best. Right: what the forest uses to separate growth regimes.")

# %% 12. EXECUTIVE DASHBOARD AND REPORT ---------------------------------------
def kpi_tile(ax, value, label, sub=None, color=None):
    ax.axis("off")
    ax.add_patch(FancyBboxPatch((0.02, 0.05), 0.96, 0.9, boxstyle="round,pad=0.0,rounding_size=0.06", transform=ax.transAxes,
                                fc="#F6F8FC", ec="#D5DCE8", lw=1))
    ax.text(0.5, 0.64, value, ha="center", va="center", fontsize=21, fontweight="bold", color=color or C["navy"], transform=ax.transAxes)
    ax.text(0.5, 0.33, label, ha="center", va="center", fontsize=9, color=C["ink"], transform=ax.transAxes)
    if sub:
        ax.text(0.5, 0.15, sub, ha="center", va="center", fontsize=7.8, color=C["mid"], transform=ax.transAxes)


def executive_dashboard(panel, forecast_frame, reg_results, class_results, cluster_result):
    fig = plt.figure(figsize=(17, 10.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[0.8, 2.3, 2.3], hspace=0.38)
    g_top = gs[0].subgridspec(1, 6, wspace=0.12)
    g_mid = gs[1].subgridspec(1, 3, wspace=0.85, width_ratios=[1.0, 1.15, 1.15])
    g_bot = gs[2].subgridspec(1, 2, wspace=0.55, width_ratios=[1.15, 1.0])
    reg_sel = reg_base = cls_sel = cls_base = None
    if reg_results is not None and not reg_results.empty:
        reg_sel = reg_results[reg_results["Selected_on_Validation"]].iloc[0]
        reg_base = reg_results[reg_results["Model_Type"] == "Baseline"].sort_values("RMSE").iloc[0]
    if class_results is not None:
        r = class_results["results"]
        cls_sel = r[r["Model"] == class_results["selected_by_validation"]].iloc[0]
        cls_base = r[r["Model_Type"] == "Baseline"].sort_values("Macro_F1", ascending=False).iloc[0]
    valid = bool(cluster_result and cluster_result.get("cluster_valid"))
    tiles = [
        (f"{panel['iso3'].nunique()}", "countries & economies", f"{panel['year'].min()}\u2013{panel['year'].max()}", None),
        (f"{len(forecast_frame):,}", "one-year-ahead samples", f"{panel.shape[0]:,} country-years in panel", None),
        (f"{reg_sel['RMSE']:.2f} pp" if reg_sel is not None else "n/a", "best regression RMSE",
         (f"{(reg_base['RMSE'] - reg_sel['RMSE']) / reg_base['RMSE'] * 100:+.1f}% vs best naive baseline" if reg_sel is not None else None),
         C["teal"] if reg_sel is not None and reg_sel["RMSE"] < reg_base["RMSE"] else C["red"]),
        (f"{cls_sel['Macro_F1']:.3f}" if cls_sel is not None else "n/a", "best classifier macro-F1",
         (f"{cls_sel['Macro_F1'] - cls_base['Macro_F1']:+.3f} vs strongest baseline" if cls_sel is not None else None),
         C["teal"] if cls_sel is not None and cls_sel["Macro_F1"] > cls_base["Macro_F1"] else C["red"]),
        (f"{cluster_result['cluster_count']}" if valid else "n/a", "country development groups",
         (f"silhouette {cluster_result['silhouette']:.2f}" if valid else None), None),
        (f"{reg_sel['RMSE_ex_COVID']:.2f} pp" if reg_sel is not None and pd.notna(reg_sel["RMSE_ex_COVID"]) else "n/a", "RMSE excluding 2020\u201321", "pandemic sensitivity check", None),
    ]
    for i, t in enumerate(tiles):
        kpi_tile(fig.add_subplot(g_top[0, i]), t[0], t[1], t[2], t[3])

    q = ARTIFACTS.get("growth_q")
    ax = fig.add_subplot(g_mid[0, 0])
    if q is not None:
        ax.fill_between(q.index, q[.10], q[.90], color=C["blue"], alpha=0.10, lw=0)
        ax.fill_between(q.index, q[.25], q[.75], color=C["blue"], alpha=0.25, lw=0)
        ax.plot(q.index, q[.50], color=C["navy"], lw=2)
        ax.axhline(0, color=C["red"], lw=0.9, ls="--")
        ax.set_ylim(max(q[.10].min() - 1, -16), min(q[.90].max() + 1, 20))
    ax.set_title("Global growth fan chart"); ax.set_ylabel("GDP growth (%)")
    style_axis(ax)

    ax = fig.add_subplot(g_mid[0, 1])
    if reg_results is not None:
        d = reg_results.sort_values("RMSE", ascending=False).reset_index(drop=True)
        cols = [C["grey"] if t == "Baseline" else (C["navy"] if s else C["blue"]) for t, s in zip(d["Model_Type"], d["Selected_on_Validation"])]
        lo = d["RMSE"].min() * 0.97
        ax.hlines(range(len(d)), lo, d["RMSE"], color=cols, lw=2.2)
        ax.scatter(d["RMSE"], range(len(d)), color=cols, s=52, zorder=3)
        ax.set_yticks(range(len(d))); ax.set_yticklabels([m.replace(" (Huber loss)", "").replace(" Regression", "").replace("Regressor", "") for m in d["Model"]], fontsize=8)
        ax.set_xlim(lo, d["RMSE"].max() * 1.08)
        for i, v in enumerate(d["RMSE"]):
            ax.text(v + (d["RMSE"].max() - lo) * 0.02, i, f"{v:.2f}", ha="left", va="center", fontsize=7.5)
    ax.set_title("Regression test RMSE (grey = naive baseline)")
    ax.set_xlabel("RMSE, pp (lower is better)")
    style_axis(ax, xgrid=True, ygrid=False)

    ax = fig.add_subplot(g_mid[0, 2])
    if class_results is not None:
        d = class_results["results"].sort_values("Macro_F1")
        ax.barh(range(len(d)), d["Macro_F1"], color=[C["grey"] if t == "Baseline" else (C["navy"] if m == class_results["selected_by_validation"] else C["blue"]) for t, m in zip(d["Model_Type"], d["Model"])])
        ax.set_yticks(range(len(d))); ax.set_yticklabels([m.replace(" Classifier", "").replace(" (Decision Trees)", "").replace("Support Vector Machine (Linear SVM)", "Linear SVM").replace("Feedforward Neural Network (MLP)", "MLP").replace("K-Nearest Neighbors (KNN)", "KNN") for m in d["Model"]], fontsize=7.8)
        for i, v in enumerate(d["Macro_F1"]):
            ax.text(v, i, f" {v:.2f}", va="center", fontsize=7.5)
        ax.set_xlim(0, d["Macro_F1"].max() * 1.15)
    ax.set_title("Classification test macro-F1")
    style_axis(ax, xgrid=True, ygrid=False)

    ax = fig.add_subplot(g_bot[0, 0])
    if valid and "PC1" in cluster_result["subset"].columns:
        sub = cluster_result["subset"]
        for c in sorted(sub["cluster"].unique()):
            m = sub["cluster"] == c
            ax.scatter(sub.loc[m, "PC1"], sub.loc[m, "PC2"], s=26, color=cluster_result["colors"][c], alpha=0.85, edgecolor="white", lw=0.4,
                       label=f"{cluster_result['names'][c]} ({m.sum()})")
        for iso, x, y in zip(sub["iso3"], sub["PC1"], sub["PC2"]):
            if iso in LABEL_COUNTRIES[:8]:
                ax.annotate(iso, (x, y), xytext=(3, 3), textcoords="offset points", fontsize=7)
        ax.legend(loc="best", fontsize=8)
        ax.grid(False)
    else:
        ax.text(0.5, 0.5, "No valid cluster solution", ha="center", va="center", transform=ax.transAxes, color=C["mid"])
    ax.set_title("Country development map (PCA of snapshot)")

    ax = fig.add_subplot(g_bot[0, 1])
    imp = ARTIFACTS.get("rf_importance")
    if imp is not None:
        top = imp.head(10).iloc[::-1]
        ax.barh(range(len(top)), top.values, color=[C["blue"] if ("growth" in n.lower() or "world" in n.lower()) else C["teal"] for n in top.index])
        ax.set_yticks(range(len(top))); ax.set_yticklabels(top.index, fontsize=8.3)
        ax.legend(handles=[Patch(color=C["blue"], label="Growth momentum / world cycle"), Patch(color=C["teal"], label="Structural")], loc="lower right", fontsize=8)
    ax.set_title("What drives the growth regime? (Random Forest)")
    style_axis(ax, xgrid=True, ygrid=False)
    finish(fig, "00_executive_dashboard", "GlobalPulse \u2014 Executive Dashboard",
           "World Bank WDI \u00b7 chronological train/validation/test \u00b7 every model is judged against naive baselines.", tight=False)


def write_headline_tables(reg_results, class_results, cluster_result):
    """Stable, paper-friendly copies of the three headline result tables (tables/)."""
    if reg_results is not None:
        reg_results.to_csv(TABLE_DIR / "regression_results.csv", index=False, encoding="utf-8-sig")
    if class_results is not None:
        class_results["results"].to_csv(TABLE_DIR / "classification_results.csv", index=False, encoding="utf-8-sig")
    if cluster_result is not None and cluster_result.get("validation") is not None:
        v = cluster_result["validation"].copy()
        v["selected"] = v["k"] == cluster_result.get("cluster_count")
        v.to_csv(TABLE_DIR / "clustering_results.csv", index=False, encoding="utf-8-sig")


def final_summary(panel, coverage, forecast_frame, reg_results, class_results, cluster_result):
    banner("GLOBALPULSE - ANALYSIS COMPLETE")
    write_headline_tables(reg_results, class_results, cluster_result)
    print(f"  Country-year observations : {len(panel):,}")
    print(f"  Countries/economies       : {panel['iso3'].nunique():,}")
    print(f"  Panel year range          : {panel['year'].min()}-{panel['year'].max()}")
    print(f"  One-year-ahead samples    : {len(forecast_frame):,}")
    best_reg = reg_base = best_cls = None
    verdicts = []
    if reg_results is not None and not reg_results.empty:
        best_reg = reg_results[reg_results["Selected_on_Validation"]].iloc[0]
        reg_base = reg_results[reg_results["Model_Type"] == "Baseline"].sort_values("RMSE").iloc[0]
        d = reg_base["RMSE"] - best_reg["RMSE"]
        print(f"  Regression  : {best_reg['Model']} | test RMSE {best_reg['RMSE']:.4f}, R2 {best_reg['R2']:.4f} | best baseline {reg_base['Model']} {reg_base['RMSE']:.4f} | gain {d:+.4f}")
        verdicts.append(f"Regression: the validation-selected **{best_reg['Model']}** reaches test RMSE {best_reg['RMSE']:.3f} pp versus {reg_base['RMSE']:.3f} pp for the strongest naive baseline (**{reg_base['Model']}**), a change of {d:+.3f} pp "
                        f"({'better' if d > 0 else 'not better'}). Test R\u00b2 = {best_reg['R2']:.3f}. Next-year growth is largely unpredictable from macro indicators; the honest benchmark is the naive forecast.")
    if class_results is not None:
        r = class_results["results"]
        best_cls = r[r["Model"] == class_results["selected_by_validation"]].iloc[0]
        print(f"  Classifier  : {best_cls['Model']} | test macro-F1 {best_cls['Macro_F1']:.4f} | vs strongest baseline ({class_results['baseline_best_name']}) {class_results['baseline_macro_f1_delta']:+.4f}")
        verdicts.append(f"Classification: **{best_cls['Model']}** reaches test macro-F1 {best_cls['Macro_F1']:.3f}; difference versus the strongest baseline (**{class_results['baseline_best_name']}**) is {class_results['baseline_macro_f1_delta']:+.3f}.")
    valid = bool(cluster_result and cluster_result.get("cluster_valid"))
    if valid:
        print(f"  Clusters    : k={cluster_result['cluster_count']} | silhouette {cluster_result['silhouette']:.4f} | snapshot {cluster_result['reference_year']}")
        verdicts.append(f"Segmentation: k = {cluster_result['cluster_count']} development groups (silhouette {cluster_result['silhouette']:.3f}, snapshot {cluster_result['reference_year']}).")

    try:
        executive_dashboard(panel, forecast_frame, reg_results, class_results, cluster_result)
    except Exception as exc:
        info("DASHBOARD", f"Could not build dashboard: {exc}")

    run_time = datetime.now().astimezone().isoformat(timespec="seconds")
    figs = sorted(p.name for p in FIG_DIR.glob("*.png"))
    lines = ["# GlobalPulse v6 \u2013 Global Economic Development & Risk Intelligence", "", f"**Generated:** {run_time}",
             "**Data source:** World Bank WDI API v2", f"**Run type:** {RUN_LABEL}",
             f"**Data provenance:** {DATA_PROVENANCE.get('mode', 'unknown')} (status: {DATA_PROVENANCE.get('status', 'unknown')})",
             f"**Output directory:** {PROJECT_DIR.resolve()}",
             f"**Requested period:** {START_YEAR}\u2013{END_YEAR}", "",
             "## Headline findings", *[f"- {v}" for v in verdicts], "", "## Dataset overview",
             f"- Country-year rows: {len(panel):,}", f"- Countries/economies: {panel['iso3'].nunique():,}",
             f"- One-year-ahead target rows: {len(forecast_frame):,}", "", "## Regression results"]
    lines.append(("```text\n" + reg_results.round(4).to_string(index=False) + "\n```") if reg_results is not None else "Not produced.")
    lines += ["", "## Classification results"]
    lines.append(("```text\n" + class_results["results"].round(4).to_string(index=False) + "\n```") if class_results is not None else "Not produced.")
    lines += ["", "## Country segmentation"]
    if valid:
        lines += [f"- Snapshot: {cluster_result['reference_year']} (latest value within {SNAPSHOT_WINDOW_YEARS} years)", f"- Groups: {cluster_result['cluster_count']}",
                  f"- Silhouette: {cluster_result['silhouette']:.4f}", "- Features: " + ", ".join(pretty(c) for c in cluster_result["features"])]
    else:
        lines.append("No valid K-Means solution; see tables/kmeans_cluster_validation.csv.")
    lines += ["", "## Reproducibility",
              f"- Download mode: {DATA_PROVENANCE.get('mode', 'unknown')}; status: {DATA_PROVENANCE.get('status', 'unknown')}",
              f"- Download finished: {DATA_PROVENANCE.get('finished_at', 'n/a')}",
              f"- SHA-256 of data/world_bank_wdi_long.csv: `{DATA_PROVENANCE.get('long_csv_sha256', 'n/a')}`",
              "- Full provenance: `data/download_manifest.json`; console log: `logs/execution_log.txt`",
              f"- Random seed: {RANDOM_SEED}"]
    lines += ["", "## Data coverage", "```text\n" + coverage.round(2).to_string(index=False) + "\n```", "", "## Figures", *[f"- `figures/{f}`" for f in figs],
              "", "## Methodological limitations",
              "- Country-year rows are not independent; countries recur over time.",
              "- Final test years are never used for model selection; feature coverage, winsorisation, imputation and scaling are fit on earlier years only.",
              "- The test period contains the 2020 pandemic shock; RMSE excluding 2020-21 is reported alongside.",
              "- Region dummies are static; the World Bank income label is excluded from predictors to avoid look-ahead.",
              "- Relationships are observational, not causal; clusters describe one snapshot; association rules show co-occurrence only.",
              "- Not an official sovereign-risk rating or policy forecast."]
    report_path = REPORT_DIR / "globalpulse_research_report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    summary = {"generated_at": run_time, "run_label": RUN_LABEL, "data_mode": DATA_PROVENANCE.get("mode"), "data_status": DATA_PROVENANCE.get("status"),
               "start_year": START_YEAR, "end_year": END_YEAR, "country_year_rows": int(len(panel)),
               "countries": int(panel["iso3"].nunique()), "forecast_rows": int(len(forecast_frame)),
               "regression_selected": str(best_reg["Model"]) if best_reg is not None else None,
               "regression_test_rmse": float(best_reg["RMSE"]) if best_reg is not None else None,
               "best_baseline_rmse": float(reg_base["RMSE"]) if reg_base is not None else None,
               "classification_selected": str(best_cls["Model"]) if best_cls is not None else None,
               "classification_test_macro_f1": float(best_cls["Macro_F1"]) if best_cls is not None else None,
               "cluster_valid": valid, "cluster_count": int(cluster_result["cluster_count"]) if valid else None,
               "cluster_silhouette": float(cluster_result["silhouette"]) if valid else None, "runtime_seconds": round(time.time() - RUN_STARTED, 1)}
    (REPORT_DIR / "globalpulse_run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n  Figures : {FIG_DIR}\n  Tables  : {TABLE_DIR}\n  Data    : {DATA_DIR}\n  Report  : {report_path}")
    print(f"  Runtime : {time.time() - RUN_STARTED:.0f}s")
    print("=" * 78)


# %% 13. MAIN ------------------------------------------------------------------
def main():
    log_path = start_logging()
    run_ok = False
    try:
        banner("GLOBALPULSE IEEE FRESH RUN - GLOBAL ECONOMIC DEVELOPMENT & RISK INTELLIGENCE")
        print(f"  World Bank WDI | {START_YEAR}-{END_YEAR}")
        print(f"  OUTPUT DIRECTORY (absolute): {PROJECT_DIR.resolve()}")
        print(f"    data/    {DATA_DIR.resolve()}")
        print(f"    figures/ {FIG_DIR.resolve()}")
        print(f"    tables/  {TABLE_DIR.resolve()}")
        print(f"    reports/ {REPORT_DIR.resolve()}")
        print(f"    logs/    {LOG_DIR.resolve()}  (this run: {log_path.name})")
        print(f"  Fresh download: {FRESH_DOWNLOAD} | strict: {STRICT_FRESH_DOWNLOAD} | seed: {RANDOM_SEED}")
        print("  The original globalpulse_out folder is not read or modified by this script.")
        long_df, countries = download_fresh_data()
        panel, coverage = build_panel(long_df, countries)
        unit1_eda(panel, coverage)
        forecast_frame, features = create_forecast_frame(panel)
        if forecast_frame.empty:
            raise RuntimeError("No consecutive country-year pairs available for next-year GDP growth modelling.")
        train, test, usable, _ = chronological_split(forecast_frame, features)
        model_data, reg_results = unit2_regression(train, test, usable)
        class_results = unit3_classification(model_data) if model_data is not None else None
        cluster_result = unit4_clustering(panel)
        unit5_pca_mlp(model_data, class_results)
        unit6_ensembles(class_results, model_data)
        final_summary(panel, coverage, forecast_frame, reg_results, class_results, cluster_result)
        run_ok = True
    except BaseException as exc:
        print("\n" + "!" * 78)
        print(f"  RUN FAILED: {type(exc).__name__}: {exc}")
        print("!" * 78)
        try:
            _LOG_STATE["handle"].write("\n" + traceback.format_exc() + "\n")   # full traceback goes to the log file
            _LOG_STATE["handle"].flush()
        except Exception:
            pass
        raise
    finally:
        list_generated_files()
        print(f"\n  RUN STATUS: {'COMPLETED' if run_ok else 'FAILED / INCOMPLETE'}  |  data provenance: "
              f"{DATA_PROVENANCE.get('mode', 'n/a')} / {DATA_PROVENANCE.get('status', 'n/a')}")
        stop_logging()


if __name__ == "__main__":
    main()
