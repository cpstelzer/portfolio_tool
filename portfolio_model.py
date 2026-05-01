"""
portfolio_model.py  (v2)

Reconstructs historical portfolio performance from transaction data and
compares it against a configurable benchmark ETF.

Returns are computed as Time-Weighted Returns (TWR), which isolate
price-driven growth from the mechanical effect of capital injections.

Assets can be identified in two ways:
  • ISIN   — a 12-character identifier mapped to a Yahoo Finance ticker
             via the lookup file (e.g. IE00B5BMR087 → SXR8.DE). Used for
             most equities, ETFs and bonds.
  • ticker — a Yahoo Finance symbol used directly, for assets that have
             no ISIN. Examples: BTC-USD (Bitcoin), ETH-USD (Ethereum),
             GC=F (gold futures), SI=F (silver futures).

Cash / sell model assumptions
------------------------------
There are two cash models, selected via config.yml:

  Legacy mode (clearing_account.use_clearing_account = false):
    • Buys are modelled as external capital injection: money enters the
      portfolio and is immediately converted to shares. Net cash effect
      of a buy = 0; cumulative_invested increases by the cost.
    • Sells convert shares to cash that remains inside the portfolio.
      Cash only grows (from sell proceeds) and is never withdrawn.
    • No deposits or withdrawals are modelled.

  Clearing-account mode (default, use_clearing_account = true):
    • The broker's cashflow_detailed.csv ledger drives a real running
      cash balance. Deposits add to cash, withdrawals subtract from it,
      and (optionally) quarterly Zinsabschluss interest is applied.
    • Buys consume cash; the cash to fund a buy must already have been
      deposited. cumulative_invested still tracks gross buy volume.
    • A new series cumulative_contributed = deposits − withdrawals
      reflects net capital actually put in, and is used for the
      money-weighted return (XIRR) and the "net" total-return figure.
    • The "mirror flows" benchmark mirrors real external flows.

  Common to both modes:
    • Ausschüttung (cash distributions) are added to portfolio cash and
      do NOT reduce cumulative_invested; they represent income, not a
      return of capital.
    • TWR uses Yahoo Finance adjusted-close prices (dividends reinvested
      in the price series). Cash-distribution tracking and tax simulation
      use broker-reported amounts. These are intentionally different views
      of the same underlying economic events.

Usage:
    python portfolio_model.py transactions_detailed.csv isin_to_yahoo.csv \
        --benchmark VWCE.DE

    # Legacy simplified format also accepted:
    python portfolio_model.py transaction.csv isin_to_yahoo.csv

    # With manual off-broker transactions:
    python portfolio_model.py transactions_detailed.csv isin_to_yahoo.csv \
        --manual-transactions transactions_manual.csv

    # With Austrian tax simulation:
    python portfolio_model.py transactions_detailed.csv isin_to_yahoo.csv \
        --tax

    # Verbose diagnostics:
    python portfolio_model.py transactions_detailed.csv isin_to_yahoo.csv \
        --verbose
"""

from __future__ import annotations

import argparse
import shutil
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yaml
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


# ---------------------------------------------------------------------------
# 0. Configuration (config.yml)
# ---------------------------------------------------------------------------

@dataclass
class CacheConfig:
    gap_warn_business_days: int = 5


@dataclass
class DataSourcesConfig:
    stooq_fallback: bool = True


@dataclass
class ClearingAccountConfig:
    use_clearing_account: bool = True
    path: str = "cashflow_detailed.csv"
    reference_iban_suffix: str = "4480"
    include_clearing_interest: bool = True
    warn_on_negative_cash: bool = True


@dataclass
class AppConfig:
    cache: CacheConfig = field(default_factory=CacheConfig)
    data_sources: DataSourcesConfig = field(default_factory=DataSourcesConfig)
    clearing_account: ClearingAccountConfig = field(default_factory=ClearingAccountConfig)


def _load_config(path: Path | None = None) -> AppConfig:
    """
    Load configuration from config.yml next to this module.
    Missing file → defaults. Malformed file → fail loudly.
    """
    cfg_path = path or (Path(__file__).resolve().parent / "config.yml")
    if not cfg_path.exists():
        return AppConfig()
    try:
        with cfg_path.open("r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except yaml.YAMLError as e:
        sys.exit(f"[ERROR] Failed to parse {cfg_path}: {e}")
    if not isinstance(data, dict):
        sys.exit(f"[ERROR] {cfg_path}: top-level must be a mapping.")
    cache_data = data.get("cache") or {}
    if not isinstance(cache_data, dict):
        sys.exit(f"[ERROR] {cfg_path}: 'cache' must be a mapping.")
    ds_data = data.get("data_sources") or {}
    if not isinstance(ds_data, dict):
        sys.exit(f"[ERROR] {cfg_path}: 'data_sources' must be a mapping.")
    ca_data = data.get("clearing_account") or {}
    if not isinstance(ca_data, dict):
        sys.exit(f"[ERROR] {cfg_path}: 'clearing_account' must be a mapping.")
    return AppConfig(
        cache=CacheConfig(
            gap_warn_business_days=int(cache_data.get(
                "gap_warn_business_days", CacheConfig.gap_warn_business_days)),
        ),
        data_sources=DataSourcesConfig(
            stooq_fallback=bool(ds_data.get(
                "stooq_fallback", DataSourcesConfig.stooq_fallback)),
        ),
        clearing_account=ClearingAccountConfig(
            use_clearing_account=bool(ca_data.get(
                "use_clearing_account", ClearingAccountConfig.use_clearing_account)),
            path=str(ca_data.get(
                "path", ClearingAccountConfig.path)),
            reference_iban_suffix=str(ca_data.get(
                "reference_iban_suffix", ClearingAccountConfig.reference_iban_suffix)),
            include_clearing_interest=bool(ca_data.get(
                "include_clearing_interest", ClearingAccountConfig.include_clearing_interest)),
            warn_on_negative_cash=bool(ca_data.get(
                "warn_on_negative_cash", ClearingAccountConfig.warn_on_negative_cash)),
        ),
    )


# ---------------------------------------------------------------------------
# 1. CLI
# ---------------------------------------------------------------------------

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct portfolio performance and compare to a benchmark."
    )
    parser.add_argument("transaction_file", type=Path,
                        help="Path to transaction CSV (broker format or legacy format; auto-detected)")
    parser.add_argument(
        "lookup_file",
        type=Path,
        nargs="?",
        default=None,
        help=(
            "Path to isin_to_yahoo.csv (optional). Only required when "
            "transactions identify assets by ISIN."
        ),
    )
    parser.add_argument(
        "--manual-transactions",
        type=Path,
        default=None,
        dest="manual_transactions",
        help="Path to transactions_manual.csv for off-broker transactions (optional)",
    )
    parser.add_argument(
        "--benchmark",
        default="VWCE.DE",
        help="Yahoo Finance ticker for the benchmark ETF (default: VWCE.DE)",
    )
    parser.add_argument(
        "--buy_and_hold",
        type=lambda x: x.lower() in ("true", "1", "yes"),
        default=True,
        help="Benchmark mode: true = buy-and-hold, false = mirror cash flows (default: true)",
    )
    parser.add_argument("--start_date", default=None, help="Override start date (YYYY-MM-DD)")
    parser.add_argument("--end_date", default=None, help="Override end date (YYYY-MM-DD)")
    parser.add_argument(
        "--export_csv",
        default=None,
        type=Path,
        help="Optional path to export TWR and value time series as CSV",
    )
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        default=False,
        help="Print detailed diagnostics: ticker map, tx classifications, spike filter actions, FX rates",
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        default=False,
        dest="no_plot",
        help="Suppress chart display (useful for scripting / CI)",
    )
    parser.add_argument(
        "--holdings",
        action="store_true",
        default=False,
        help="Print current holdings breakdown table without running full analysis",
    )
    parser.add_argument(
        "--tax",
        action="store_true",
        default=False,
        help="Run Austrian KESt (27.5%%) tax simulation and print yearly summary",
    )
    parser.add_argument(
        "--spike-threshold",
        type=float,
        default=0.15,
        dest="spike_threshold",
        help="Candidate spike threshold for equity tickers (default: 0.15 = 15%%)",
    )
    parser.add_argument(
        "--base-currency",
        default="EUR",
        dest="base_currency",
        help="Base reporting currency (default: EUR; only EUR supported now)",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        default=False,
        dest="no_cache",
        help="Force full re-download of price data; ignore local cache",
    )
    parser.add_argument(
        "--clear-cache",
        action="store_true",
        default=False,
        dest="clear_cache",
        help="Delete all cached price files and exit",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "data",
        dest="cache_dir",
        help="Directory for local price cache (default: <repo>/data/)",
    )
    parser.add_argument(
        "--no-cache-gap-refetch",
        action="store_true",
        default=False,
        dest="no_cache_gap_refetch",
        help="Detect and warn on interior cache gaps but do not refetch them",
    )
    parser.add_argument(
        "--no-stooq-fallback",
        action="store_true",
        default=False,
        dest="no_stooq_fallback",
        help="Disable the Stooq fallback used when yfinance returns gappy or stale data",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# 2. Data loading & validation
# ---------------------------------------------------------------------------

REQUIRED_TX_COLS = {"date", "shares"}
REQUIRED_LOOKUP_COLS = {"ISIN", "YahooTicker"}


def _parse_dates(raw_dates: pd.Series) -> pd.Series:
    """Accept DD.MM.YYYY (European) and YYYY-MM-DD (ISO) formats."""
    raw = raw_dates.astype(str).str.strip()
    iso = pd.to_datetime(raw, format="%Y-%m-%d", errors="coerce")
    eu = pd.to_datetime(raw, format="%d.%m.%Y", errors="coerce")
    return iso.fillna(eu)


def _normalise_id_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure ISIN and ticker columns exist and are clean strings."""
    for col in ("ISIN", "ticker"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str).str.strip()
    return df


def load_broker_transactions(path: Path) -> pd.DataFrame:
    """
    Load a broker-format CSV (Flatex/DADAT export).

    Expected columns (by position — unit columns are interleaved):
        0  Buchungstag      → date
        1  Valuta           (skipped)
        2  Bezeichnung      → name
        3  ISIN             → ISIN
        4  Nominal (Stk.)   → shares
        5  (unit, e.g. Stück) (skipped)
        6  Betrag           → gross_amount
        7  (unit, e.g. €)   (skipped)
        8  Kurs             → price
        9  (unit)           (skipped)
        10 Devisenkurs      → fx_rate
        11 TA.-Nr.          → tx_id
        12 Buchungsinformation → booking_info

    Encoding: Latin-1.
    """
    raw = pd.read_csv(path, encoding="latin-1", header=0)

    # Rename by position to handle the unnamed unit columns
    col_map = {
        raw.columns[0]:  "date",
        raw.columns[2]:  "name",
        raw.columns[3]:  "ISIN",
        raw.columns[4]:  "shares",
        raw.columns[6]:  "gross_amount",
        raw.columns[8]:  "price",
        raw.columns[10]: "fx_rate",
        raw.columns[11]: "tx_id",
        raw.columns[12]: "booking_info",
    }
    df = raw.rename(columns=col_map)
    # Keep only the mapped columns
    keep = list(col_map.values())
    df = df[[c for c in keep if c in df.columns]].copy()

    df["date"] = _parse_dates(df["date"])
    if df["date"].isna().any():
        bad = (df.index[df["date"].isna()] + 2).tolist()
        sys.exit(f"[ERROR] Unparseable dates in rows {bad} of broker file.")

    for num_col in ("shares", "gross_amount", "price", "fx_rate"):
        if num_col in df.columns:
            df[num_col] = pd.to_numeric(df[num_col], errors="coerce")

    df["ticker"] = ""
    df["currency"] = "EUR"   # broker file is always EUR in this implementation
    df["is_off_broker"] = False
    df = _normalise_id_columns(df)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_legacy_transactions(path: Path) -> pd.DataFrame:
    """Load the simplified legacy transaction.csv format."""
    df = pd.read_csv(path)
    missing = REQUIRED_TX_COLS - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] transaction file missing columns: {missing}")

    if "ISIN" not in df.columns and "ticker" not in df.columns:
        sys.exit(
            "[ERROR] transaction file must contain either an 'ISIN' column "
            "(resolved via the lookup file) or a 'ticker' column (used as a "
            "Yahoo Finance symbol directly for assets without an ISIN, e.g. "
            "BTC-USD, GC=F)."
        )

    df = _normalise_id_columns(df)

    empty_ids = (df["ISIN"] == "") & (df["ticker"] == "")
    if empty_ids.any():
        bad = (df.index[empty_ids] + 2).tolist()
        sys.exit(
            f"[ERROR] transaction rows {bad} have neither 'ISIN' nor 'ticker' "
            "set. Every row must provide one of them."
        )

    df["date"] = _parse_dates(df["date"])
    if df["date"].isna().any():
        bad = (df.index[df["date"].isna()] + 2).tolist()
        sys.exit(
            f"[ERROR] Unparseable dates in rows {bad}. "
            "Use YYYY-MM-DD (ISO) or DD.MM.YYYY (European)."
        )

    df["shares"] = pd.to_numeric(df["shares"], errors="coerce")

    for col in ("price", "gross_amount"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        else:
            df[col] = np.nan

    for col in ("booking_info", "tx_id", "name", "fx_rate"):
        if col not in df.columns:
            df[col] = "" if col in ("booking_info", "tx_id", "name") else np.nan

    if "currency" not in df.columns:
        df["currency"] = "EUR"

    # Heuristic: a row with no ISIN in the legacy file is an off-broker
    # holding (e.g. physical gold, crypto wallet) — it does not move money
    # through the broker's clearing account.
    df["is_off_broker"] = (df["ISIN"] == "")

    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_manual_transactions(path: Path) -> pd.DataFrame:
    """
    Load transactions_manual.csv (off-broker trades).

    Required columns: date, name, ticker, shares, price, currency
    Optional:         fx_rate, notes
    """
    df = pd.read_csv(path)
    required = {"date", "name", "ticker", "shares", "price", "currency"}
    missing = required - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] manual transactions file missing columns: {missing}")

    df["date"] = _parse_dates(df["date"])
    if df["date"].isna().any():
        bad = (df.index[df["date"].isna()] + 2).tolist()
        sys.exit(f"[ERROR] Unparseable dates in manual transactions rows {bad}.")

    df["shares"] = pd.to_numeric(df["shares"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")

    if "fx_rate" not in df.columns:
        df["fx_rate"] = np.nan
    else:
        df["fx_rate"] = pd.to_numeric(df["fx_rate"], errors="coerce")

    if "gross_amount" not in df.columns:
        df["gross_amount"] = df["shares"] * df["price"]

    df["ISIN"] = ""
    df["booking_info"] = ""
    df["tx_id"] = ""
    df["is_off_broker"] = True

    df = _normalise_id_columns(df)
    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_transactions(path: Path) -> pd.DataFrame:
    """
    Unified entry point — auto-detects format and routes to the
    appropriate loader.

    Detection:
      • If the file contains a 'Buchungstag' column → broker format
        (load_broker_transactions).
      • Otherwise → legacy simplified format (load_legacy_transactions).
    """
    # Peek at the header only
    try:
        header_df = pd.read_csv(path, encoding="latin-1", nrows=0)
        if "Buchungstag" in header_df.columns:
            return load_broker_transactions(path)
    except Exception:
        pass
    return load_legacy_transactions(path)


CASHFLOW_CATEGORIES = (
    "deposit", "withdrawal", "trade_leg", "distribution",
    "thesaurierung", "clearing_interest", "unclassified",
)


def _classify_cashflow_row(
    info: str,
    amount: float,
    counterparty_in: str,
    counterparty_out: str,
    reference_iban_suffix: str,
) -> tuple[str, bool]:
    """
    Classify one ledger row by its 'Buchungsinformationen' text and amount.

    Returns (category, was_iban_fallback). The fallback flag is True only
    when we had to rely on the reference IBAN suffix to recognise an
    own-account deposit or withdrawal.
    """
    text = (info or "").strip()
    upper = text.upper()

    # Direct text-based classification.
    if "ZINSABSCHLUSS" in upper:
        return "clearing_interest", False
    if "AUSF" in upper and "KAUF" in upper:
        return "trade_leg", False
    if "AUSF" in upper and "VERKAUF" in upper:
        return "trade_leg", False
    if "THESAURIERUNG" in upper:
        return "thesaurierung", False
    if "ERTR" in upper and ("AUSSCH" in upper or "GNIS" in upper):
        return "distribution", False
    if "CODE: SCOR" in upper:
        return "deposit" if amount > 0 else "withdrawal", False
    if "BEKANNT" in upper or "RUECKUEBERWEISUNG" in upper or "RÜCKÜBERWEISUNG" in upper:
        return "deposit" if amount > 0 else "withdrawal", False

    # IBAN-suffix fallback for empty/unrecognised info.
    # The broker records the external bank account in the same column
    # regardless of direction, so check both Empfänger and Zahlungspfl.
    # The amount sign tells us which side the money is moving.
    suffix = (reference_iban_suffix or "").strip()
    if suffix and (suffix in counterparty_in or suffix in counterparty_out):
        return ("deposit" if amount > 0 else "withdrawal"), True

    return "unclassified", False


def load_cashflow_ledger(
    path: Path,
    reference_iban_suffix: str,
    verbose: bool = False,
) -> pd.DataFrame:
    """
    Load the broker's clearing-account ledger (cashflow_detailed.csv).

    Expected columns (Latin-1 encoded German export):
        Buchungstag, Valuta, Empfänger, Zahlungspfl., TA.Nr.,
        Buchungsinformationen, Betrag, Currency

    Returns a DataFrame indexed by row number with columns:
        valuta            — booking value-date (pd.Timestamp)
        buchungstag       — booking date (pd.Timestamp)
        category          — one of CASHFLOW_CATEGORIES
        amount_eur        — float (positive = credit, negative = debit)
        ta_nr             — broker TA-number string
        info              — original Buchungsinformationen string
        counterparty      — Empfänger (credit side) or Zahlungspfl. (debit side)
        iban_fallback     — True if classification used the IBAN-suffix fallback
    """
    raw = pd.read_csv(path, encoding="latin-1", header=0)
    raw.columns = [c.strip() for c in raw.columns]

    # Tolerate slightly different column names from different exports.
    def _col(*candidates: str) -> str:
        for c in candidates:
            if c in raw.columns:
                return c
        sys.exit(
            f"[ERROR] {path}: cashflow ledger missing one of columns {candidates}; "
            f"got {list(raw.columns)}"
        )

    col_buchung = _col("Buchungstag")
    col_valuta = _col("Valuta")
    col_recv = _col("Empfänger", "Empfaenger")
    col_pay = _col("Zahlungspfl.", "Zahlungspfl", "Zahlungspflichtiger")
    col_tanr = _col("TA.Nr.", "TA-Nr.", "TA.-Nr.", "TA.Nr")
    col_info = _col("Buchungsinformationen", "Buchungsinformation")
    col_amt = _col("Betrag")

    df = pd.DataFrame({
        "buchungstag": _parse_dates(raw[col_buchung]),
        "valuta": _parse_dates(raw[col_valuta]),
        "counterparty_in": raw[col_recv].fillna("").astype(str).str.strip(),
        "counterparty_out": raw[col_pay].fillna("").astype(str).str.strip(),
        "ta_nr": raw[col_tanr].fillna("").astype(str).str.strip(),
        "info": raw[col_info].fillna("").astype(str).str.strip(),
        "amount_eur": pd.to_numeric(raw[col_amt], errors="coerce"),
    })

    bad_dates = df["valuta"].isna()
    if bad_dates.any():
        rows = (df.index[bad_dates] + 2).tolist()
        sys.exit(f"[ERROR] {path}: unparseable Valuta dates in rows {rows}.")

    bad_amt = df["amount_eur"].isna()
    if bad_amt.any():
        rows = (df.index[bad_amt] + 2).tolist()
        sys.exit(f"[ERROR] {path}: unparseable Betrag values in rows {rows}.")

    cats: list[str] = []
    fallbacks: list[bool] = []
    for _, r in df.iterrows():
        cat, was_fb = _classify_cashflow_row(
            r["info"], float(r["amount_eur"]),
            r["counterparty_in"], r["counterparty_out"],
            reference_iban_suffix,
        )
        cats.append(cat)
        fallbacks.append(was_fb)
    df["category"] = cats
    df["iban_fallback"] = fallbacks
    df["counterparty"] = np.where(
        df["amount_eur"] > 0, df["counterparty_in"], df["counterparty_out"]
    )

    # Inform the user about IBAN-fallback resolutions and remaining
    # unclassified rows (the latter is a real warning).
    fb_rows = df[df["iban_fallback"]]
    for _, r in fb_rows.iterrows():
        warnings.warn(
            f"[INFO] Cashflow row on {r['valuta'].date()} (TA {r['ta_nr']}, "
            f"EUR {r['amount_eur']:.2f}): empty info field; classified as "
            f"{r['category']} via reference-IBAN match.",
            stacklevel=2,
        )
    unc = df[df["category"] == "unclassified"]
    for _, r in unc.iterrows():
        warnings.warn(
            f"[WARN] Cashflow row on {r['valuta'].date()} (TA {r['ta_nr']}, "
            f"EUR {r['amount_eur']:.2f}, info='{r['info']}') could not be "
            f"classified and will be ignored.",
            stacklevel=2,
        )

    if verbose:
        counts = df["category"].value_counts().to_dict()
        print("[VERBOSE] Cashflow ledger category counts:")
        for cat in CASHFLOW_CATEGORIES:
            if cat in counts:
                print(f"    {cat:18s} {counts[cat]:4d}")

    df = df.drop(columns=["counterparty_in", "counterparty_out"])
    df = df.sort_values("valuta").reset_index(drop=True)
    return df


def reconcile_cashflow_with_transactions(
    cashflow: pd.DataFrame,
    transactions: pd.DataFrame,
    verbose: bool = False,
) -> None:
    """
    Print a one-line-per-category reconciliation report comparing the
    clearing-account ledger to the transactions file. Strictly
    informational — never aborts.
    """
    if cashflow.empty:
        return

    by_cat = cashflow.groupby("category")["amount_eur"].agg(["count", "sum"])

    # Count buy/sell rows in the transactions file (cashflow trade-legs
    # should match this 1:1). Distribution and thesaurierung rows in the
    # transactions file come as paired negative+positive entries, so a
    # row-count comparison would be misleading and is omitted here.
    # The legacy simplified format has no booking_info column; in that
    # case we skip the comparison entirely.
    info_series = transactions.get("booking_info")
    has_info = info_series is not None and info_series.fillna("").astype(str).str.strip().ne("").any()
    if has_info:
        info_upper = info_series.fillna("").astype(str).str.upper()
        n_buy_sell = int(info_upper.str.contains("AUSF").sum())
    else:
        n_buy_sell = None

    def _fmt(cat: str) -> str:
        if cat in by_cat.index:
            n = int(by_cat.loc[cat, "count"])
            s = float(by_cat.loc[cat, "sum"])
            return f"{n:4d} entries, EUR {s:>14,.2f}"
        return "   0 entries"

    print("\n" + "=" * 66)
    print("  CLEARING-ACCOUNT RECONCILIATION")
    print("=" * 66)
    print(f"  Deposits         : {_fmt('deposit')}")
    print(f"  Withdrawals      : {_fmt('withdrawal')}")
    trade_line = f"  Trade legs       : {_fmt('trade_leg')}"
    if n_buy_sell is not None:
        trade_line += f"   (vs {n_buy_sell} buy/sell tx)"
    print(trade_line)
    print(f"  Distributions    : {_fmt('distribution')}")
    print(f"  Thesaurierungen  : {_fmt('thesaurierung')}")
    print(f"  Clearing interest: {_fmt('clearing_interest')}")
    print(f"  Unclassified     : {_fmt('unclassified')}")
    n_trade_legs = int(by_cat.loc["trade_leg", "count"]) if "trade_leg" in by_cat.index else 0
    if n_buy_sell is not None and n_trade_legs != n_buy_sell:
        print(f"  [WARN] Trade-leg count ({n_trade_legs}) differs from buy/sell tx count ({n_buy_sell}).")
    print("=" * 66)

    if verbose:
        unc = cashflow[cashflow["category"] == "unclassified"]
        if not unc.empty:
            print("\n  Unclassified rows:")
            for _, r in unc.iterrows():
                print(f"    {r['valuta'].date()} TA={r['ta_nr']} "
                      f"EUR {r['amount_eur']:>10,.2f}  info='{r['info']}'")
            print()


def load_lookup(path: Path) -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """
    Load isin_to_yahoo.csv.

    Returns:
        isin_to_ticker  : dict ISIN → YahooTicker
        isin_to_currency: dict ISIN → Currency (default 'EUR')
        isin_to_class   : dict ISIN → asset_class (default 'other')
    """
    df = pd.read_csv(path)
    missing = REQUIRED_LOOKUP_COLS - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] lookup file missing columns: {missing}")

    isin_to_ticker: dict[str, str] = dict(zip(df["ISIN"], df["YahooTicker"]))

    if "Currency" in df.columns:
        isin_to_currency: dict[str, str] = dict(zip(df["ISIN"], df["Currency"].fillna("EUR")))
    else:
        isin_to_currency = {isin: "EUR" for isin in df["ISIN"]}

    if "asset_class" in df.columns:
        isin_to_class: dict[str, str] = dict(zip(df["ISIN"], df["asset_class"].fillna("other")))
    else:
        isin_to_class = {isin: "other" for isin in df["ISIN"]}

    return isin_to_ticker, isin_to_currency, isin_to_class


def resolve_tickers(
    transactions: pd.DataFrame,
    isin_to_ticker: dict[str, str],
) -> pd.Series:
    """
    Return a Series aligned to transactions.index giving the Yahoo Finance
    symbol for each row, or None if unresolvable.

    Precedence:
      1. Non-empty 'ticker' column → used verbatim.
      2. Non-empty 'ISIN' column  → looked up in isin_to_ticker.
      3. Otherwise → None.
    """
    def _resolve(row: pd.Series) -> str | None:
        direct = str(row.get("ticker", "") or "").strip()
        if direct:
            return direct
        isin = str(row.get("ISIN", "") or "").strip()
        if isin:
            return isin_to_ticker.get(isin)
        return None

    return transactions.apply(_resolve, axis=1)


def resolve_currencies(
    transactions: pd.DataFrame,
    isin_to_currency: dict[str, str],
) -> pd.Series:
    """
    Return a Series with the price currency for each transaction row.

    Precedence:
      1. 'currency' column in transaction row (for manual/direct-ticker rows).
      2. ISIN lookup via isin_to_currency.
      3. Default 'EUR'.
    """
    def _cur(row: pd.Series) -> str:
        explicit = str(row.get("currency", "") or "").strip().upper()
        if explicit and explicit != "NAN":
            return explicit
        isin = str(row.get("ISIN", "") or "").strip()
        if isin:
            return isin_to_currency.get(isin, "EUR")
        return "EUR"

    return transactions.apply(_cur, axis=1)


# ---------------------------------------------------------------------------
# 3. Auto-classification of transaction types  (§9)
# ---------------------------------------------------------------------------

def classify_tx_type(booking_info: str, shares: float) -> str:
    """
    Classify a transaction row based on its booking_info string.

    Patterns (checked in order):
      'Kauf'                                      → buy
      'Verkauf'                                   → sell
      'Thesaurierung'                             → thesaurierung
      'Fusion'                                    → fusion
      'Ausschüttung' or 'Erträgnisausschüttung'  → ausschuettung
      'Spin-off' or 'Kapitalmaßnahme'             → corporate_action
      fallback: shares > 0 → buy, shares < 0 → sell
    """
    info = str(booking_info or "")
    # Normalise German special characters that may appear as encoding artefacts
    # or raw Unicode equivalently.
    if "Kauf" in info:
        return "buy"
    if "Verkauf" in info:
        return "sell"
    if "Thesaurierung" in info:
        return "thesaurierung"
    if "Fusion" in info:
        return "fusion"
    # 'Ausschüttung' covers both 'Erträgnisausschüttung' and plain 'Ausschüttung'
    if "ttung" in info and ("Aussch" in info or "Ertr" in info):
        return "ausschuettung"
    if "Spin-off" in info or "Kapitalma" in info:
        return "corporate_action"
    # Sign-based fallback
    if float(shares or 0) >= 0:
        return "buy"
    return "sell"


def classify_transactions(df: pd.DataFrame, verbose: bool = False) -> pd.DataFrame:
    """Add a 'tx_type' column to the transactions DataFrame."""
    df = df.copy()
    booking_col = df["booking_info"] if "booking_info" in df.columns else pd.Series("", index=df.index)
    df["tx_type"] = [
        classify_tx_type(bi, sh)
        for bi, sh in zip(booking_col, df["shares"])
    ]
    if verbose:
        print("\n[VERBOSE] Transaction type classifications:")
        for _, row in df.iterrows():
            print(f"  {row['date'].date()}  {row.get('ISIN',''):<16}  "
                  f"shares={row['shares']:>10.4f}  type={row['tx_type']:<20}  "
                  f"info={str(row.get('booking_info',''))[:60]}")
    return df


# ---------------------------------------------------------------------------
# 4. Price retrieval & cache  (§10a, §4)
# ---------------------------------------------------------------------------

_CRYPTO_SUFFIXES = ("-USD", "-EUR", "-GBP", "-USDT", "-BTC", "-ETH", "-JPY")


def _is_crypto_ticker(ticker: str) -> bool:
    upper = ticker.upper()
    return any(upper.endswith(suf) for suf in _CRYPTO_SUFFIXES)


def _cache_path(cache_dir: Path, ticker: str) -> Path:
    safe = ticker.replace("/", "_").replace("\\", "_").replace(":", "_")
    return cache_dir / f"{safe}.csv"


def _load_cache(cache_dir: Path, ticker: str) -> pd.DataFrame | None:
    p = _cache_path(cache_dir, ticker)
    if not p.exists():
        return None
    try:
        df = pd.read_csv(p, parse_dates=["date"], index_col="date")
        df.index = pd.to_datetime(df.index, utc=False)
        df.index = df.index.tz_localize(None)
        return df
    except Exception:
        return None


def _save_cache(cache_dir: Path, ticker: str, df: pd.DataFrame) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = _cache_path(cache_dir, ticker)
    df.to_csv(p, index_label="date")


def _detect_cache_gaps(
    series: pd.Series,
    threshold_bdays: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp]]:
    """
    Find interior gaps in a daily-ish price series.

    A gap is a span of consecutive *missing* business days strictly
    between two existing rows. yfinance never returns weekends or
    exchange holidays, so the threshold is in business days to avoid
    false positives over normal weekends/holiday weeks.

    Returns a list of (gap_start, gap_end) inclusive ranges, where
    gap_start = (prev_row + 1 calendar day) and gap_end = (next_row -
    1 calendar day). Empty list when the series has no gaps exceeding
    the threshold.
    """
    if series is None or series.empty or threshold_bdays < 0:
        return []
    idx = series.dropna().index.sort_values()
    if len(idx) < 2:
        return []
    gaps: list[tuple[pd.Timestamp, pd.Timestamp]] = []
    one_day = pd.Timedelta(days=1)
    for prev, curr in zip(idx[:-1], idx[1:]):
        gap_start = prev + one_day
        gap_end = curr - one_day
        if gap_start > gap_end:
            continue
        n_bdays = len(pd.bdate_range(gap_start, gap_end))
        if n_bdays > threshold_bdays:
            gaps.append((gap_start, gap_end))
    return gaps


def _detect_stale_runs(
    series: pd.Series,
    threshold_bdays: int,
) -> list[tuple[pd.Timestamp, pd.Timestamp, float]]:
    """
    Find runs of consecutive identical values in a daily-ish price series.

    Detects the failure mode where yfinance returns a daily row for
    every business day but the value is the same number repeated for
    weeks/months/years (illiquid or halted instruments). This passes
    the index-gap check undetected.

    Returns (run_start, run_end, value) for runs whose business-day
    span strictly exceeds threshold_bdays.
    """
    if series is None or series.empty or threshold_bdays < 0:
        return []
    s = series.dropna().sort_index()
    if len(s) < 2:
        return []

    runs: list[tuple[pd.Timestamp, pd.Timestamp, float]] = []
    run_start_pos = 0
    for i in range(1, len(s)):
        if s.iloc[i] != s.iloc[i - 1]:
            if i - 1 > run_start_pos:
                start_ts = s.index[run_start_pos]
                end_ts = s.index[i - 1]
                n_bdays = len(pd.bdate_range(start_ts, end_ts))
                if n_bdays > threshold_bdays:
                    runs.append((start_ts, end_ts, float(s.iloc[run_start_pos])))
            run_start_pos = i
    # Trailing run
    if len(s) - 1 > run_start_pos:
        start_ts = s.index[run_start_pos]
        end_ts = s.index[-1]
        n_bdays = len(pd.bdate_range(start_ts, end_ts))
        if n_bdays > threshold_bdays:
            runs.append((start_ts, end_ts, float(s.iloc[run_start_pos])))
    return runs


def _download_prices_raw(
    tickers: list[str],
    start: str,
    end: str,
) -> pd.DataFrame:
    """Download adjusted-close prices via yfinance; returns wide DataFrame."""
    raw = yf.download(
        tickers,
        start=start,
        end=end,
        auto_adjust=True,
        progress=False,
        threads=True,
    )
    if raw.empty:
        warnings.warn(
            f"[WARN] yfinance returned no data for tickers {tickers} "
            f"({start} → {end}). They may be delisted or the range is outside their trading history.",
            stacklevel=2,
        )
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        if len(tickers) == 1:
            prices.columns = tickers

    # Drop timezone info for consistency
    if hasattr(prices.index, "tz") and prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)

    return prices


# ── Stooq fallback ─────────────────────────────────────────────────────────
# Stooq (https://stooq.com) serves daily OHLCV CSV without auth or rate
# limits and often has cleaner data for European ETCs/ETFs than Yahoo.
# Used as a fallback when yfinance returns gappy or stale data.

# Yahoo exchange suffix → Stooq exchange suffix.
# Stooq omits the suffix for US tickers (or accepts ".us"); for others
# it uses ISO-style country codes. List is conservative — extend as
# needed when new exchanges show up.
_YAHOO_TO_STOOQ_SUFFIX: dict[str, str] = {
    "AS": "nl",   # Euronext Amsterdam
    "DE": "de",   # Xetra
    "F":  "de",   # Frankfurt
    "L":  "uk",   # London
    "PA": "fr",   # Euronext Paris
    "MI": "it",   # Borsa Italiana
    "SW": "ch",   # SIX Swiss
    "MC": "es",   # Madrid
    "VI": "at",   # Vienna
    "OL": "no",   # Oslo
    "ST": "se",   # Stockholm
    "CO": "dk",   # Copenhagen
    "HE": "fi",   # Helsinki
    "BR": "be",   # Brussels
    "LS": "pt",   # Lisbon
    "IR": "ie",   # Ireland
    "WA": "pl",   # Warsaw
    "PR": "cz",   # Prague
    "BD": "hu",   # Budapest
}


def _yahoo_to_stooq(yahoo_symbol: str) -> str | None:
    """
    Translate a Yahoo Finance symbol to its Stooq equivalent.
    Returns None when no mapping is known (caller should skip).
    Examples:
        PHPD.AS  → phpd.nl
        VWCE.DE  → vwce.de
        AAPL     → aapl.us
        BTC-USD  → None (crypto / non-equity, not handled)
    """
    if not yahoo_symbol or "-" in yahoo_symbol or "=" in yahoo_symbol:
        return None
    if "." not in yahoo_symbol:
        # Assume US listing
        return f"{yahoo_symbol.lower()}.us"
    base, _, suffix = yahoo_symbol.rpartition(".")
    stooq_suffix = _YAHOO_TO_STOOQ_SUFFIX.get(suffix.upper())
    if stooq_suffix is None:
        return None
    return f"{base.lower()}.{stooq_suffix}"


def _download_stooq(
    yahoo_symbol: str,
    start: str,
    end: str,
) -> pd.Series:
    """
    Fetch daily Close prices for `yahoo_symbol` from Stooq for
    [start, end] (both inclusive in Stooq's API). Returns an empty
    Series on any failure (unknown symbol, network error, no data).
    """
    stooq_sym = _yahoo_to_stooq(yahoo_symbol)
    if stooq_sym is None:
        return pd.Series(dtype="float64")
    d1 = pd.Timestamp(start).strftime("%Y%m%d")
    # Stooq's d2 is inclusive; yfinance end is exclusive. Subtract 1
    # day so the request covers the same set of trading days as the
    # equivalent yfinance call.
    d2 = (pd.Timestamp(end) - pd.Timedelta(days=1)).strftime("%Y%m%d")
    url = f"https://stooq.com/q/d/l/?s={stooq_sym}&d1={d1}&d2={d2}&i=d"
    try:
        df = pd.read_csv(url)
    except Exception:
        return pd.Series(dtype="float64")
    if df.empty or "Date" not in df.columns or "Close" not in df.columns:
        return pd.Series(dtype="float64")
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"]).set_index("Date").sort_index()
    if df.empty:
        return pd.Series(dtype="float64")
    s = df["Close"].astype("float64").dropna()
    s.name = yahoo_symbol
    return s


def fetch_prices(
    tickers: list[str],
    start: str,
    end: str,
    cache_dir: Path | None = None,
    no_cache: bool = False,
    verbose: bool = False,
    spike_threshold: float = 0.15,
    gap_warn_bdays: int = 5,
    refetch_gaps: bool = True,
    use_stooq_fallback: bool = True,
) -> pd.DataFrame:
    """
    Download (or load from cache) adjusted-close prices for all tickers.
    Returns a wide DataFrame indexed by trading date.

    Spike filter (§4 — reversal-based detection):
      A candidate spike is flagged when |r[t]| > threshold.
      It is confirmed as a data error only if r[t+1] has opposite sign
      AND |r[t+1]| > 0.5 × |r[t]| (i.e. more than half reverses next day).
      Confirmed spikes are replaced with linearly interpolated values.
      Real crashes / rallies that do NOT reverse are left untouched.
    """
    SPIKE_THRESHOLD_CRYPTO = 0.50

    # ── Per-ticker cache logic ───────────────────────────────────────────────
    all_frames: dict[str, pd.Series] = {}
    # ticker → list of (dl_start, dl_end) ranges to fetch (dl_end exclusive,
    # matching yfinance semantics).
    need_download: dict[str, list[tuple[str, str]]] = {}

    for ticker in tickers:
        if cache_dir is not None and not no_cache:
            cached = _load_cache(cache_dir, ticker)
            if cached is not None and not cached.empty and "close" in cached.columns:
                cached_series = cached["close"]
                last_cached = cached.index.max()
                target_end = pd.Timestamp(end)
                ranges_for_ticker: list[tuple[str, str]] = []

                # Tail refetch when the cache hasn't been updated to the
                # requested end date.
                if last_cached < target_end - pd.Timedelta(days=2):
                    dl_start = (last_cached + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                    ranges_for_ticker.append((dl_start, end))

                # Interior-gap refetch: any span of consecutive missing
                # business days inside the cache that exceeds the threshold.
                if refetch_gaps:
                    gaps = _detect_cache_gaps(cached_series, gap_warn_bdays)
                    for gap_s, gap_e in gaps:
                        n_bd = len(pd.bdate_range(gap_s, gap_e))
                        print(f"[INFO] {ticker}: detected interior cache gap "
                              f"{gap_s.date()} → {gap_e.date()} ({n_bd} business "
                              f"days); scheduling refetch.")
                        ranges_for_ticker.append((
                            gap_s.strftime("%Y-%m-%d"),
                            (gap_e + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                        ))

                all_frames[ticker] = cached_series
                if ranges_for_ticker:
                    need_download[ticker] = ranges_for_ticker
                    if verbose:
                        print(f"[VERBOSE] {ticker}: {len(ranges_for_ticker)} "
                              f"refetch range(s) scheduled")
                else:
                    if verbose:
                        print(f"[VERBOSE] Cache hit for {ticker}: "
                              f"{len(cached_series)} rows (last={last_cached.date()})")
                continue
        need_download[ticker] = [(start, end)]

    # Batch download needed tickers grouped by (start, end) so multiple
    # tickers sharing a range hit yfinance in a single call.
    if need_download:
        ranges: dict[tuple[str, str], list[str]] = {}
        for ticker, dl_ranges in need_download.items():
            for (dl_s, dl_e) in dl_ranges:
                ranges.setdefault((dl_s, dl_e), []).append(ticker)

        for (dl_s, dl_e), batch in ranges.items():
            print(f"[INFO] Downloading prices for {len(batch)} ticker(s) "
                  f"from {dl_s} to {dl_e} …")
            raw_prices = _download_prices_raw(batch, dl_s, dl_e)

            for ticker in batch:
                if ticker not in raw_prices.columns:
                    warnings.warn(
                        f"[WARN] Ticker '{ticker}' not returned by yfinance; skipping.",
                        stacklevel=2,
                    )
                    continue
                new_series = raw_prices[ticker].dropna()

                existing = all_frames.get(ticker)
                if existing is not None and not existing.empty:
                    combined = pd.concat([existing, new_series])
                    combined = combined[~combined.index.duplicated(keep="last")]
                    combined = combined.sort_index()
                    all_frames[ticker] = combined
                else:
                    all_frames[ticker] = new_series

                if cache_dir is not None and not no_cache:
                    _save_cache(cache_dir, ticker,
                                pd.DataFrame({"close": all_frames[ticker]}))

    if not all_frames:
        sys.exit("[ERROR] No price data available after cache/download.")

    # ── Stooq fallback ──────────────────────────────────────────────────────
    # For each ticker, if yfinance left index gaps or stale-value runs in
    # the analysis window, ask Stooq for the affected range and splice in
    # the result if it's healthier (no gaps and no stale runs of its own).
    # The cache is updated so subsequent runs benefit from the repair.
    if use_stooq_fallback:
        for ticker in list(all_frames.keys()):
            series = all_frames[ticker].loc[start:end]
            bad_ranges: list[tuple[pd.Timestamp, pd.Timestamp]] = []
            bad_ranges.extend(_detect_cache_gaps(series, gap_warn_bdays))
            for run_s, run_e, _val in _detect_stale_runs(series, gap_warn_bdays):
                bad_ranges.append((run_s, run_e))
            if not bad_ranges:
                continue

            stooq_sym = _yahoo_to_stooq(ticker)
            if stooq_sym is None:
                if verbose:
                    print(f"[VERBOSE] Stooq fallback: no symbol mapping for "
                          f"{ticker}; skipping.")
                continue

            spliced_any = False
            for r_start, r_end in bad_ranges:
                # Pad each range by a few days on either side so Stooq
                # data overlaps neighbouring known-good rows.
                pad = pd.Timedelta(days=7)
                fetch_start = (r_start - pad).strftime("%Y-%m-%d")
                fetch_end = (r_end + pad + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                print(f"[INFO] {ticker}: yfinance data unhealthy for "
                      f"{r_start.date()} → {r_end.date()}; trying Stooq "
                      f"({stooq_sym}) …")
                stooq_series = _download_stooq(ticker, fetch_start, fetch_end)
                if stooq_series.empty:
                    print(f"[INFO] {ticker}: Stooq returned no data for this range.")
                    continue

                # Restrict to the bad range itself so we don't overwrite
                # known-good neighbours with potentially-different Stooq values.
                window = stooq_series.loc[r_start:r_end]
                if window.empty:
                    continue

                # Verify Stooq's data is actually healthier than what we have.
                if (_detect_cache_gaps(window, gap_warn_bdays)
                        or _detect_stale_runs(window, gap_warn_bdays)):
                    print(f"[INFO] {ticker}: Stooq data also has gaps or "
                          f"stale runs in this range; not splicing.")
                    continue

                existing = all_frames[ticker]
                combined = pd.concat([existing, window])
                combined = combined[~combined.index.duplicated(keep="last")]
                combined = combined.sort_index()
                all_frames[ticker] = combined
                spliced_any = True
                print(f"[INFO] {ticker}: spliced {len(window)} row(s) from "
                      f"Stooq covering {window.index.min().date()} → "
                      f"{window.index.max().date()}.")

            if spliced_any and cache_dir is not None and not no_cache:
                _save_cache(cache_dir, ticker,
                            pd.DataFrame({"close": all_frames[ticker]}))

    # ── Residual-gap and stale-run warnings ─────────────────────────────────
    # Two failure modes are checked here, both restricted to the analysis
    # window so noise outside [start, end] doesn't surface:
    #
    #  1. *Index gaps*: missing business-day rows between two cached rows.
    #     If they remain after refetch, yfinance has no data for the range.
    #
    #  2. *Stale-value runs*: rows present for every business day but the
    #     value is the same number repeated. yfinance does this for halted
    #     or illiquid instruments. The index-gap check cannot see this; the
    #     value-based check catches it. No remediation is possible (refetch
    #     returns the same stale data) — just warn so the user knows the
    #     TWR is being computed across artificial flat data.
    for ticker, series in all_frames.items():
        windowed = series.loc[start:end]

        residual_gaps = _detect_cache_gaps(windowed, gap_warn_bdays)
        for gap_s, gap_e in residual_gaps:
            n_bd = len(pd.bdate_range(gap_s, gap_e))
            warnings.warn(
                f"[WARN] {ticker}: residual price gap {gap_s.date()} → "
                f"{gap_e.date()} ({n_bd} business days) after refetch. "
                f"Forward-fill will hold the last known price across this "
                f"range; portfolio value and TWR will appear flat here.",
                stacklevel=2,
            )

        stale_runs = _detect_stale_runs(windowed, gap_warn_bdays)
        for run_s, run_e, val in stale_runs:
            n_bd = len(pd.bdate_range(run_s, run_e))
            warnings.warn(
                f"[WARN] {ticker}: stale price run {run_s.date()} → "
                f"{run_e.date()} ({n_bd} business days at {val:.4f}). "
                f"yfinance returned the same value for every day in this "
                f"range (likely halted/illiquid instrument). Portfolio "
                f"value and TWR will appear flat here; treat reported "
                f"performance over this range as unreliable.",
                stacklevel=2,
            )

    # Build wide DataFrame from start to end
    prices = pd.DataFrame(all_frames)
    prices = prices.loc[start:end]

    # Warn and drop tickers with >85% missing values
    frac_missing = prices.isna().mean()
    bad_tickers = frac_missing[frac_missing > 0.85].index.tolist()
    if bad_tickers:
        warnings.warn(
            f"[WARN] Dropping tickers with insufficient data: {bad_tickers}",
            stacklevel=2,
        )
        prices = prices.drop(columns=bad_tickers)

    prices = prices.ffill()

    # ── Reversal-based spike filter ──────────────────────────────────────────
    for _pass in range(2):
        returns = prices.pct_change()
        for col in prices.columns:
            threshold = (
                SPIKE_THRESHOLD_CRYPTO if _is_crypto_ticker(col)
                else spike_threshold
            )
            r = returns[col]
            confirmed_spikes: list[pd.Timestamp] = []
            for i in range(len(r) - 1):
                t = r.index[i]
                t1 = r.index[i + 1]
                rt = r.iloc[i]
                rt1 = r.iloc[i + 1]
                if pd.isna(rt) or pd.isna(rt1):
                    continue
                if abs(rt) > threshold:
                    # Confirmed spike: next day reverses > 50% of the move
                    if (np.sign(rt1) != np.sign(rt)) and (abs(rt1) > 0.5 * abs(rt)):
                        confirmed_spikes.append(t)
            if confirmed_spikes:
                if verbose:
                    print(
                        f"[VERBOSE] {col}: reversal-confirmed spike(s) on "
                        f"{[d.date() for d in confirmed_spikes]}; interpolating."
                    )
                else:
                    warnings.warn(
                        f"[WARN] {col}: reversal-confirmed spike(s) on "
                        f"{[d.date() for d in confirmed_spikes]}; interpolating.",
                        stacklevel=2,
                    )
                prices.loc[confirmed_spikes, col] = np.nan
        prices = prices.interpolate(method="time").ffill().bfill()

    # Drop the final bar (possible partial intraday data)
    if len(prices) > 1:
        prices = prices.iloc[:-1]

    return prices


# ---------------------------------------------------------------------------
# 5. FX rates  (§1)
# ---------------------------------------------------------------------------

def fetch_fx_rates(
    currencies: set[str],
    start: str,
    end: str,
    cache_dir: Path | None = None,
    no_cache: bool = False,
    verbose: bool = False,
    spike_threshold: float = 0.15,
    gap_warn_bdays: int = 5,
    refetch_gaps: bool = True,
    use_stooq_fallback: bool = True,
) -> pd.DataFrame:
    """
    Download daily EUR/foreign FX rates for any non-EUR currencies.

    Yahoo Finance convention for EUR/USD: EURUSD=X gives USD per 1 EUR.
    To convert a USD price to EUR:  price_eur = price_usd / EURUSD=X.
    This function returns a DataFrame with columns named by currency code
    (e.g. "USD") where the value is units of that currency per 1 EUR.

    EUR/EUR is trivially 1.0 everywhere and is NOT downloaded.
    """
    foreign = {c.upper() for c in currencies if c.upper() != "EUR"}
    if not foreign:
        return pd.DataFrame()

    fx_tickers = [f"EUR{c}=X" for c in sorted(foreign)]
    if verbose:
        print(f"[VERBOSE] Fetching FX rates for: {fx_tickers}")

    fx_prices = fetch_prices(
        fx_tickers,
        start=start,
        end=end,
        cache_dir=cache_dir,
        no_cache=no_cache,
        verbose=verbose,
        spike_threshold=spike_threshold,
        gap_warn_bdays=gap_warn_bdays,
        refetch_gaps=refetch_gaps,
        use_stooq_fallback=use_stooq_fallback,
    )

    # Rename columns from EURUSD=X → USD
    rename = {f"EUR{c}=X": c for c in foreign}
    fx_prices = fx_prices.rename(columns=rename)
    return fx_prices


def get_fx_rate(
    fx_rates: pd.DataFrame,
    currency: str,
    day: pd.Timestamp,
) -> float:
    """
    Return units of `currency` per 1 EUR on `day`.
    For EUR, always returns 1.0.
    Falls back to the nearest available date if exact date is missing.
    """
    cur = currency.upper()
    if cur == "EUR":
        return 1.0
    if fx_rates.empty or cur not in fx_rates.columns:
        return 1.0  # safe fallback; will generate a warning upstream
    if day in fx_rates.index:
        v = fx_rates.at[day, cur]
        if pd.notna(v) and v > 0:
            return float(v)
    # Use the nearest earlier date
    earlier = fx_rates.index[fx_rates.index <= day]
    if len(earlier) > 0:
        v = fx_rates.at[earlier[-1], cur]
        if pd.notna(v) and v > 0:
            return float(v)
    return 1.0


def price_to_eur(
    price_foreign: float,
    currency: str,
    fx_rate_local: float,
) -> float:
    """
    Convert a price in `currency` to EUR using `fx_rate_local`
    (units of currency per 1 EUR).

    price_eur = price_foreign / fx_rate_local
    """
    if currency.upper() == "EUR":
        return price_foreign
    if fx_rate_local > 0:
        return price_foreign / fx_rate_local
    return price_foreign


# ---------------------------------------------------------------------------
# 6. Per-lot cost basis ledger  (§3a)
# ---------------------------------------------------------------------------

@dataclass
class Lot:
    ticker: str
    isin: str
    date: pd.Timestamp
    shares: float
    cost_price_eur: float           # per-share cost in EUR at acquisition
    thesaurierung_adj: float = 0.0  # cumulative per-share thesaurierung adjustment


@dataclass
class LotLedger:
    """FIFO lot-based cost basis tracker."""
    lots: list[Lot] = field(default_factory=list)

    def add_lot(self, lot: Lot) -> None:
        self.lots.append(lot)

    def get_lots_for(self, ticker: str) -> list[Lot]:
        return [l for l in self.lots if l.ticker == ticker and l.shares > 1e-9]

    def consume_fifo(
        self,
        ticker: str,
        shares_to_sell: float,
        sell_price_eur: float,
    ) -> list[dict]:
        """
        Consume lots FIFO for a sell of `shares_to_sell` units of `ticker`.
        Returns a list of gain records: {shares, cost_eur, gain_eur}.
        """
        remaining = shares_to_sell
        gains = []
        for lot in self.lots:
            if lot.ticker != ticker or lot.shares < 1e-9:
                continue
            if remaining <= 1e-9:
                break
            take = min(lot.shares, remaining)
            adj_cost = lot.cost_price_eur + lot.thesaurierung_adj
            gain = (sell_price_eur - adj_cost) * take
            gains.append({
                "shares": take,
                "cost_eur": adj_cost,
                "gain_eur": gain,
                "lot_date": lot.date,
            })
            lot.shares -= take
            remaining -= take

        if remaining > 1e-9:
            warnings.warn(
                f"[WARN] Sell of {shares_to_sell} {ticker} exceeds tracked lots "
                f"by {remaining:.4f} shares. Check transaction history.",
                stacklevel=2,
            )
        return gains

    def update_thesaurierung(self, ticker: str, isin: str, date: pd.Timestamp,
                              delta_eur_per_share: float) -> None:
        """Apply a per-share thesaurierung adjustment to all matching lots."""
        for lot in self.lots:
            if lot.ticker == ticker and lot.shares > 1e-9:
                lot.thesaurierung_adj += delta_eur_per_share

    def rename_ticker(self, old_ticker: str, new_ticker: str, new_isin: str) -> None:
        """Rename lots for a fusion event."""
        for lot in self.lots:
            if lot.ticker == old_ticker:
                lot.ticker = new_ticker
                lot.isin = new_isin

    def total_shares(self, ticker: str) -> float:
        return sum(l.shares for l in self.lots if l.ticker == ticker)

    def total_cost_basis_eur(self, ticker: str) -> float:
        return sum(
            l.shares * (l.cost_price_eur + l.thesaurierung_adj)
            for l in self.lots
            if l.ticker == ticker and l.shares > 1e-9
        )


# ---------------------------------------------------------------------------
# 7. Portfolio reconstruction
# ---------------------------------------------------------------------------

def _snap_to_trading_day(dates: pd.Series, trading_days: pd.DatetimeIndex) -> pd.Series:
    """Map each date forward to the nearest following trading day."""
    def snap(d: pd.Timestamp) -> pd.Timestamp | None:
        idx = trading_days.searchsorted(d)
        if idx >= len(trading_days):
            return None
        return trading_days[idx]
    return dates.map(snap)


def _mark_to_market(
    positions: dict[str, float],
    prices: pd.DataFrame,
    day: pd.Timestamp,
    fx_rates: pd.DataFrame,
    ticker_currency: dict[str, str],
) -> float:
    """
    Compute total EUR value of positions at `day`.
    Converts foreign-currency positions to EUR using fx_rates.
    """
    total = 0.0
    for ticker, shares in positions.items():
        if ticker not in prices.columns:
            continue
        if day not in prices.index:
            continue
        p = prices.at[day, ticker]
        if pd.isna(p):
            continue
        cur = ticker_currency.get(ticker, "EUR").upper()
        fx = get_fx_rate(fx_rates, cur, day)
        total += shares * price_to_eur(float(p), cur, fx)
    return total


def _pair_thesaurierung_rows(rows: list[pd.Series]) -> list[tuple[pd.Series, pd.Series]]:
    """
    Given a list of raw transaction rows for a single ISIN on a single date,
    pair negative rows with matching positive rows by absolute share count.

    Returns list of (negative_row, positive_row) pairs.
    Unpaired rows are logged as warnings.
    """
    negatives = [r for r in rows if float(r["shares"]) < 0]
    positives = [r for r in rows if float(r["shares"]) > 0]

    pairs = []
    used_pos = [False] * len(positives)

    for neg in negatives:
        abs_neg = abs(float(neg["shares"]))
        matched = False
        for j, pos in enumerate(positives):
            if used_pos[j]:
                continue
            abs_pos = abs(float(pos["shares"]))
            if abs(abs_neg - abs_pos) < 1e-6:
                pairs.append((neg, pos))
                used_pos[j] = True
                matched = True
                break
        if not matched:
            warnings.warn(
                f"[WARN] Unpaired thesaurierung negative row: "
                f"ISIN={neg.get('ISIN','')} shares={neg['shares']}; skipping.",
                stacklevel=2,
            )

    unpaired_pos = [positives[j] for j, used in enumerate(used_pos) if not used]
    for pos in unpaired_pos:
        warnings.warn(
            f"[WARN] Unpaired thesaurierung positive row: "
            f"ISIN={pos.get('ISIN','')} shares={pos['shares']}; skipping.",
            stacklevel=2,
        )

    return pairs


def reconstruct_portfolio(
    transactions: pd.DataFrame,
    isin_to_ticker: dict[str, str],
    isin_to_currency: dict[str, str],
    prices: pd.DataFrame,
    fx_rates: pd.DataFrame,
    run_tax: bool = False,
    verbose: bool = False,
    cashflow_ledger: Optional[pd.DataFrame] = None,
    include_clearing_interest: bool = True,
    warn_on_negative_cash: bool = True,
) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by trading date with columns:
        value_before_tx        — portfolio EUR value before today's transactions
        portfolio_value        — portfolio EUR value after today's transactions
        cash                   — EUR cash held
        cumulative_invested    — running sum of gross buy cost (EUR)
        cumulative_contributed — net deposits − withdrawals (EUR);
                                 only meaningful when cashflow_ledger is given,
                                 otherwise mirrors cumulative_invested.
        cumulative_gain        — portfolio_value − cumulative_invested

    Attributes on the returned DataFrame:
        .cash_flows     — pd.Series of net capital deployed per day; when a
                          cashflow_ledger is provided this reflects real
                          deposits/withdrawals, otherwise trade-implied flows.
        .external_flows — pd.Series of (deposit − withdrawal) per day, only
                          when cashflow_ledger is provided; empty otherwise.
        .lot_ledger     — LotLedger (for §3 tax simulation)
        .tax_timeline   — pd.DataFrame (filled only when run_tax=True)
        .final_positions — dict ticker → shares
        .ticker_currency — dict ticker → currency

    When `cashflow_ledger` is provided the cash model switches from the
    legacy "buys are external injections" view to a real running cash
    balance fed by deposits, withdrawals and (optionally) clearing-account
    interest. Buys then consume cash; sells/distributions add to it.
    """
    transactions = transactions.copy()
    transactions["_yf_ticker"] = resolve_tickers(transactions, isin_to_ticker)
    transactions["_currency"] = resolve_currencies(transactions, isin_to_currency)

    no_ticker = transactions["_yf_ticker"].isna()
    if no_ticker.any():
        unresolved = sorted({
            (str(r.get("ticker") or "").strip()
             or str(r.get("ISIN") or "").strip()
             or "<empty>")
            for _, r in transactions[no_ticker].iterrows()
        })
        for ident in unresolved:
            warnings.warn(
                f"[WARN] No ticker resolved for '{ident}'; skipping.",
                stacklevel=2,
            )
    transactions = transactions.dropna(subset=["_yf_ticker"])

    # Classify transaction types
    transactions = classify_transactions(transactions, verbose=verbose)

    # Snap transaction dates to the next available trading day
    original_dates = transactions["date"].copy()
    transactions["date"] = _snap_to_trading_day(transactions["date"], prices.index)
    snapped = transactions["date"] != original_dates
    for _, row in transactions[snapped].iterrows():
        orig = original_dates.at[row.name]
        warnings.warn(
            f"[INFO] Transaction on non-trading day {orig.date()} booked on {row['date'].date()}.",
            stacklevel=2,
        )
    out_of_range = transactions["date"].isna()
    if out_of_range.any():
        warnings.warn(
            f"[WARN] {out_of_range.sum()} transaction(s) fall after the last "
            "available price date and are skipped.",
            stacklevel=2,
        )
    transactions = transactions.dropna(subset=["date"])

    # Build ticker_currency map
    ticker_currency: dict[str, str] = {}
    for _, row in transactions.iterrows():
        t = row["_yf_ticker"]
        c = row["_currency"]
        if t and c:
            ticker_currency[t] = c.upper()

    # Group thesaurierung/ausschuettung rows before day-loop
    # Key: (tx_type, ISIN, date_snapped)
    paired_thesaurierung: dict[tuple, list[tuple]] = {}
    paired_ausschuettung: dict[tuple, list[tuple]] = {}

    thesaurierung_rows = transactions[transactions["tx_type"] == "thesaurierung"]
    ausschuettung_rows = transactions[transactions["tx_type"] == "ausschuettung"]

    for date_val in thesaurierung_rows["date"].unique():
        for isin_val in thesaurierung_rows.loc[
            thesaurierung_rows["date"] == date_val, "ISIN"
        ].unique():
            mask = (
                (thesaurierung_rows["date"] == date_val)
                & (thesaurierung_rows["ISIN"] == isin_val)
            )
            rows_here = [r for _, r in thesaurierung_rows[mask].iterrows()]
            pairs = _pair_thesaurierung_rows(rows_here)
            if pairs:
                paired_thesaurierung[(isin_val, date_val)] = pairs

    for date_val in ausschuettung_rows["date"].unique():
        for isin_val in ausschuettung_rows.loc[
            ausschuettung_rows["date"] == date_val, "ISIN"
        ].unique():
            mask = (
                (ausschuettung_rows["date"] == date_val)
                & (ausschuettung_rows["ISIN"] == isin_val)
            )
            rows_here = [r for _, r in ausschuettung_rows[mask].iterrows()]
            # Pair negative + positive rows (same structure as thesaurierung)
            pairs = _pair_thesaurierung_rows(rows_here)
            if pairs:
                paired_ausschuettung[(isin_val, date_val)] = pairs

    # Keep only buy/sell/fusion/corporate_action rows for position changes
    trade_types = {"buy", "sell", "fusion", "corporate_action"}
    trade_txs = transactions[transactions["tx_type"].isin(trade_types)]

    tx_by_date: dict[pd.Timestamp, list] = {}
    for _, row in trade_txs.iterrows():
        tx_by_date.setdefault(row["date"], []).append(row)

    # Collect thesaurierung/ausschuettung events by date for the day loop
    thesSaur_by_date: dict[pd.Timestamp, list[tuple[str, list[tuple]]]] = {}
    for (isin_val, date_val), pairs in paired_thesaurierung.items():
        thesSaur_by_date.setdefault(date_val, []).append((isin_val, pairs))

    ausSch_by_date: dict[pd.Timestamp, list[tuple[str, list[tuple]]]] = {}
    for (isin_val, date_val), pairs in paired_ausschuettung.items():
        ausSch_by_date.setdefault(date_val, []).append((isin_val, pairs))

    # ── External cashflows (deposits / withdrawals / clearing interest) ─────
    use_clearing = cashflow_ledger is not None and not cashflow_ledger.empty
    deposits_by_day: dict[pd.Timestamp, float] = {}
    withdrawals_by_day: dict[pd.Timestamp, float] = {}
    interest_by_day: dict[pd.Timestamp, float] = {}
    if use_clearing:
        ledger = cashflow_ledger.copy()
        ledger["snap"] = _snap_to_trading_day(ledger["valuta"], prices.index)
        n_dropped = int(ledger["snap"].isna().sum())
        if n_dropped:
            warnings.warn(
                f"[WARN] {n_dropped} cashflow ledger row(s) fall after the last "
                "available price date and are skipped.",
                stacklevel=2,
            )
        ledger = ledger.dropna(subset=["snap"])
        for _, r in ledger.iterrows():
            day = r["snap"]
            cat = r["category"]
            amt = float(r["amount_eur"])
            if cat == "deposit":
                deposits_by_day[day] = deposits_by_day.get(day, 0.0) + amt
            elif cat == "withdrawal":
                withdrawals_by_day[day] = withdrawals_by_day.get(day, 0.0) + amt
            elif cat == "clearing_interest" and include_clearing_interest:
                interest_by_day[day] = interest_by_day.get(day, 0.0) + amt
            # trade_leg / distribution / thesaurierung / unclassified: no
            # effect on cash here — already accounted for via the
            # transactions file.

    # ── Main day loop ────────────────────────────────────────────────────────
    positions: dict[str, float] = {}   # ticker → shares (aggregate)
    cash = 0.0
    cumulative_invested = 0.0
    cumulative_contributed = 0.0
    cash_flows: dict[pd.Timestamp, float] = {}
    external_flows: dict[pd.Timestamp, float] = {}
    lot_ledger = LotLedger()
    tax_events: list[dict] = []
    cumulative_tax = 0.0
    negative_cash_warned = False
    rows_out = []

    for day in prices.index:
        value_before_tx = cash + _mark_to_market(positions, prices, day, fx_rates, ticker_currency)

        day_invested = 0.0

        # ── External cashflows (deposits, withdrawals, clearing interest) ───
        if use_clearing:
            dep = deposits_by_day.get(day, 0.0)
            wdr = withdrawals_by_day.get(day, 0.0)
            intr = interest_by_day.get(day, 0.0)
            if dep:
                cash += dep
                cumulative_contributed += dep
            if wdr:
                cash += wdr  # withdrawals are stored as negative amounts
                cumulative_contributed += wdr
            if intr:
                cash += intr
            net_external = dep + wdr  # deposits − |withdrawals|
            if net_external != 0:
                external_flows[day] = external_flows.get(day, 0.0) + net_external

        # ── Thesaurierung events ────────────────────────────────────────────
        if day in thesSaur_by_date:
            for isin_val, pairs in thesSaur_by_date[day]:
                ticker = isin_to_ticker.get(isin_val, "")
                if not ticker:
                    continue
                cur = ticker_currency.get(ticker, "EUR")
                fx = get_fx_rate(fx_rates, cur, day)

                for neg_row, pos_row in pairs:
                    shares_abs = abs(float(neg_row["shares"]))
                    old_price_raw = abs(float(neg_row.get("gross_amount", 0) or 0)) / shares_abs if shares_abs else 0
                    new_price_raw = abs(float(pos_row.get("gross_amount", 0) or 0)) / shares_abs if shares_abs else 0

                    # Use broker gross_amount for taxable delta
                    old_amount_eur = price_to_eur(
                        abs(float(neg_row.get("gross_amount", 0) or 0)), cur, fx
                    )
                    new_amount_eur = price_to_eur(
                        abs(float(pos_row.get("gross_amount", 0) or 0)), cur, fx
                    )
                    taxable_delta_eur = new_amount_eur - old_amount_eur

                    if run_tax and taxable_delta_eur > 0:
                        tax = taxable_delta_eur * 0.275
                        cumulative_tax += tax
                        tax_events.append({
                            "date": day,
                            "event_type": "thesaurierung",
                            "ticker": ticker,
                            "isin": isin_val,
                            "taxable_amount": round(taxable_delta_eur, 4),
                            "tax": round(tax, 4),
                            "cumulative_tax": round(cumulative_tax, 4),
                        })
                        # Increase cost basis by delta per share
                        delta_per_share = taxable_delta_eur / shares_abs if shares_abs else 0
                        lot_ledger.update_thesaurierung(
                            ticker, isin_val, day, delta_per_share
                        )

        # ── Ausschüttung events ─────────────────────────────────────────────
        if day in ausSch_by_date:
            for isin_val, pairs in ausSch_by_date[day]:
                ticker = isin_to_ticker.get(isin_val, "")
                if not ticker:
                    continue
                cur = ticker_currency.get(ticker, "EUR")
                fx = get_fx_rate(fx_rates, cur, day)

                for neg_row, pos_row in pairs:
                    shares_abs = abs(float(neg_row["shares"]))
                    # Distribution amount = difference in gross amounts
                    old_amount_eur = price_to_eur(
                        abs(float(neg_row.get("gross_amount", 0) or 0)), cur, fx
                    )
                    new_amount_eur = price_to_eur(
                        abs(float(pos_row.get("gross_amount", 0) or 0)), cur, fx
                    )
                    distribution_eur = new_amount_eur - old_amount_eur

                    # Cash received from distribution
                    if distribution_eur > 0:
                        cash += distribution_eur

                    if run_tax:
                        tax_base = new_amount_eur - old_amount_eur
                        if tax_base > 0:
                            tax = tax_base * 0.275
                            cumulative_tax += tax
                            tax_events.append({
                                "date": day,
                                "event_type": "ausschuettung",
                                "ticker": ticker,
                                "isin": isin_val,
                                "taxable_amount": round(tax_base, 4),
                                "tax": round(tax, 4),
                                "cumulative_tax": round(cumulative_tax, 4),
                            })

        # ── Buy / Sell / Fusion transactions ────────────────────────────────
        if day in tx_by_date:
            for tx in tx_by_date[day]:
                ticker = tx["_yf_ticker"]
                shares = float(tx["shares"])
                cur = tx["_currency"]
                tx_type = tx["tx_type"]

                # ── Fusion handling ──────────────────────────────────────────
                if tx_type == "fusion":
                    # In a fusion, old shares are removed (negative row) and
                    # new shares added (positive row). The lot_ledger handles
                    # the rename separately. Here we just update positions.
                    positions[ticker] = positions.get(ticker, 0.0) + shares
                    if positions[ticker] < 1e-9:
                        del positions[ticker]
                    continue

                # ── Resolve execution price in EUR ───────────────────────────
                fx_broker = float(tx.get("fx_rate") or 1.0)
                if pd.isna(fx_broker) or fx_broker <= 0:
                    fx_broker = 1.0

                if pd.notna(tx.get("price")) and float(tx["price"]) > 0:
                    px_native = float(tx["price"])
                elif pd.notna(tx.get("gross_amount")) and shares != 0:
                    px_native = abs(float(tx["gross_amount"])) / abs(shares)
                elif ticker in prices.columns and not pd.isna(prices.at[day, ticker]):
                    px_native = float(prices.at[day, ticker])
                else:
                    warnings.warn(
                        f"[WARN] Cannot determine price for {ticker} on {day.date()}; skipping tx.",
                        stacklevel=2,
                    )
                    continue

                # Use broker FX rate first; fallback to model FX
                if cur.upper() != "EUR":
                    model_fx = get_fx_rate(fx_rates, cur, day)
                    fx_used = fx_broker if abs(fx_broker - 1.0) > 1e-6 else model_fx
                else:
                    fx_used = 1.0

                px_eur = price_to_eur(px_native, cur, fx_used)
                cost_eur = shares * px_eur  # positive = buy, negative = sell proceeds
                # Off-broker assets (e.g. physical gold, crypto wallet) do
                # not move money through the broker's clearing account.
                # In clearing-account mode we treat them as in-kind
                # contributions/withdrawals: they update positions and
                # cumulative_contributed but never touch cash.
                is_off = bool(tx.get("is_off_broker", False)) and use_clearing

                if tx_type == "buy" or (tx_type not in ("sell",) and shares > 0):
                    # Buy: gross invested volume always increases. Cash is
                    # credited as a synthetic "external injection" only in
                    # the legacy mode (no clearing-account ledger). When
                    # the ledger is used the cash to fund this buy must
                    # already have been deposited.
                    if not use_clearing:
                        cash += cost_eur
                    cumulative_invested += cost_eur
                    day_invested += cost_eur
                    if is_off:
                        cumulative_contributed += cost_eur
                        external_flows[day] = external_flows.get(day, 0.0) + cost_eur
                    if run_tax:
                        lot_ledger.add_lot(Lot(
                            ticker=ticker,
                            isin=str(tx.get("ISIN", "") or ""),
                            date=day,
                            shares=shares,
                            cost_price_eur=px_eur,
                        ))

                elif tx_type == "sell" or (tx_type not in ("buy",) and shares < 0):
                    shares_sold = abs(shares)
                    if run_tax:
                        gains = lot_ledger.consume_fifo(ticker, shares_sold, px_eur)
                        for g in gains:
                            gain = g["gain_eur"]
                            if gain > 0:
                                tax = gain * 0.275
                                cumulative_tax += tax
                                tax_events.append({
                                    "date": day,
                                    "event_type": "sell",
                                    "ticker": ticker,
                                    "isin": str(tx.get("ISIN", "") or ""),
                                    "taxable_amount": round(gain, 4),
                                    "tax": round(tax, 4),
                                    "cumulative_tax": round(cumulative_tax, 4),
                                })
                            elif gain < 0:
                                # Record loss (cannot carry forward in Austria)
                                tax_events.append({
                                    "date": day,
                                    "event_type": "sell_loss",
                                    "ticker": ticker,
                                    "isin": str(tx.get("ISIN", "") or ""),
                                    "taxable_amount": round(gain, 4),
                                    "tax": 0.0,
                                    "cumulative_tax": round(cumulative_tax, 4),
                                })
                    day_invested += cost_eur   # negative
                    if is_off:
                        # In-kind withdrawal: cost_eur is negative, so this
                        # decreases cumulative_contributed and external_flows.
                        cumulative_contributed += cost_eur
                        external_flows[day] = external_flows.get(day, 0.0) + cost_eur

                if not is_off:
                    cash -= cost_eur   # buy: nets to 0; sell: cash increases
                positions[ticker] = positions.get(ticker, 0.0) + shares

        if use_clearing:
            # Benchmark / MWR consume real external flows (already
            # accumulated above into external_flows[day]).
            pass
        elif day_invested != 0:
            cash_flows[day] = cash_flows.get(day, 0.0) + day_invested

        if use_clearing and warn_on_negative_cash and cash < -1e-6 and not negative_cash_warned:
            warnings.warn(
                f"[WARN] Cash balance went negative on {day.date()} "
                f"(EUR {cash:,.2f}); likely a settlement-date quirk where a "
                "buy booked before its funding deposit. Continuing.",
                stacklevel=2,
            )
            negative_cash_warned = True

        value_after_tx = cash + _mark_to_market(positions, prices, day, fx_rates, ticker_currency)
        cumulative_gain = value_after_tx - cumulative_invested

        rows_out.append({
            "date": day,
            "value_before_tx": value_before_tx,
            "portfolio_value": value_after_tx,
            "cash": cash,
            "cumulative_invested": cumulative_invested,
            "cumulative_contributed": (
                cumulative_contributed if use_clearing else cumulative_invested
            ),
            "cumulative_gain": cumulative_gain,
        })

    result = pd.DataFrame(rows_out).set_index("date")

    # When the clearing account drives flows, the benchmark mirrors real
    # external deposits/withdrawals instead of trade-implied injections.
    if use_clearing:
        cash_flows = dict(external_flows)

    # Attach extra data as attributes
    result.cash_flows = pd.Series(cash_flows, name="cash_flow").sort_index()  # type: ignore[attr-defined]
    result.external_flows = pd.Series(  # type: ignore[attr-defined]
        external_flows, name="external_flow", dtype=float,
    ).sort_index()
    result.lot_ledger = lot_ledger  # type: ignore[attr-defined]
    result.tax_timeline = (  # type: ignore[attr-defined]
        pd.DataFrame(tax_events).set_index("date") if tax_events else pd.DataFrame()
    )
    result.final_positions = dict(positions)  # type: ignore[attr-defined]
    result.ticker_currency = ticker_currency  # type: ignore[attr-defined]
    return result


# ---------------------------------------------------------------------------
# 8. Benchmark simulation
# ---------------------------------------------------------------------------

def simulate_benchmark(
    cash_flows: pd.Series,
    benchmark_prices: pd.Series,
    buy_and_hold: bool,
) -> pd.DataFrame:
    """
    Simulate a benchmark investment strategy.

    buy_and_hold=True  → inflows buy benchmark shares; outflows ignored.
    buy_and_hold=False → inflows buy shares; outflows sell shares, park cash.
    """
    bm_shares = 0.0
    bm_cash = 0.0
    rows = []

    for day in benchmark_prices.index:
        px = benchmark_prices.at[day]
        if pd.isna(px) or px <= 0:
            continue

        equity = bm_shares * px
        idle_cash = bm_cash if not buy_and_hold else 0.0
        value_before = equity + idle_cash

        flow = cash_flows.get(day, 0.0)

        if buy_and_hold:
            if flow > 0:
                bm_shares += flow / px
        else:
            if flow > 0:
                bm_shares += flow / px
            elif flow < 0:
                bm_shares += flow / px
                bm_cash -= flow

        equity_after = bm_shares * px
        idle_cash_after = bm_cash if not buy_and_hold else 0.0
        value_after = equity_after + idle_cash_after

        rows.append({
            "date": day,
            "value_before_tx": value_before,
            "benchmark_value": value_after,
        })

    return pd.DataFrame(rows).set_index("date")


# ---------------------------------------------------------------------------
# 9. Time-Weighted Return (TWR)
# ---------------------------------------------------------------------------

def compute_twr(value_after: pd.Series, value_before: pd.Series) -> pd.Series:
    """
    Compute a daily TWR index starting at 1.0 on the first invested day.

    For each day t:
        sub-period factor = value_before[t] / value_after[t-1]

    Strips the mechanical effect of capital injections and withdrawals.
    """
    twr = pd.Series(np.nan, index=value_after.index, dtype=float)
    twr_val: float | None = None
    prev_after: float | None = None

    for day in value_after.index:
        v_before = float(value_before.at[day])
        v_after = float(value_after.at[day])

        if twr_val is None:
            if v_after > 0:
                twr_val = 1.0
                prev_after = v_after
                twr.at[day] = twr_val
        else:
            if prev_after is not None and prev_after > 0 and not np.isnan(v_before):
                twr_val *= v_before / prev_after
            twr.at[day] = twr_val
            prev_after = v_after if v_after > 0 else prev_after

    return twr


# ---------------------------------------------------------------------------
# 9b. Money-Weighted Return (XIRR)
# ---------------------------------------------------------------------------

def compute_xirr(
    cash_flows: pd.Series,
    final_value: float,
    final_date: pd.Timestamp,
) -> float | None:
    """
    Compute the money-weighted return (annualised XIRR) given external
    cash flows and the terminal portfolio value.

    Sign convention (from the investor's perspective):
        deposit (money in)   → negative
        withdrawal (money out) → positive
        terminal value       → positive

    The input `cash_flows` follows the model's convention
    (deposit = +, withdrawal = −), so signs are flipped internally.

    Returns the annualised IRR as a fraction (e.g. 0.07 = 7%/yr) or
    None if the iteration fails to converge or the input is degenerate.
    """
    cf = cash_flows.dropna()
    cf = cf[cf != 0.0]
    if cf.empty or final_value <= 0:
        return None

    # Build (date, amount) list in investor sign convention.
    items: list[tuple[pd.Timestamp, float]] = [
        (d, -float(a)) for d, a in cf.items()
    ]
    items.append((final_date, float(final_value)))
    items.sort(key=lambda x: x[0])

    t0 = items[0][0]
    years = np.array([(d - t0).days / 365.25 for d, _ in items], dtype=float)
    amts = np.array([a for _, a in items], dtype=float)

    # Newton-Raphson on f(r) = sum(a_i / (1+r)^t_i)
    r = 0.05
    for _ in range(120):
        denom = (1.0 + r) ** years
        f = float(np.sum(amts / denom))
        df = float(np.sum(-years * amts / (denom * (1.0 + r))))
        if not np.isfinite(f) or not np.isfinite(df) or abs(df) < 1e-14:
            break
        step = f / df
        r_new = r - step
        if not np.isfinite(r_new) or r_new <= -0.999:
            r_new = (r - 0.999) / 2.0  # damp toward the lower bound
        if abs(r_new - r) < 1e-9:
            r = r_new
            break
        r = r_new

    # Verify the residual is small; otherwise treat as non-convergence.
    denom = (1.0 + r) ** years
    if not np.all(np.isfinite(denom)):
        return None
    residual = float(np.sum(amts / denom))
    scale = float(np.sum(np.abs(amts))) or 1.0
    if abs(residual) / scale > 1e-4:
        return None
    return r


# ---------------------------------------------------------------------------
# 10. Metrics  (§7)
# ---------------------------------------------------------------------------

def compute_metrics(twr: pd.Series, label: str) -> dict:
    """
    Compute performance metrics from a TWR index series.

    Metrics:
      total_return_pct    — total return over the full period
      cagr_pct            — compound annual growth rate
      annualised_vol_pct  — annualised daily return volatility
      sharpe_ratio        — Sharpe ratio (rf = 0)
      sortino_ratio       — Sortino ratio (downside std, rf = 0)
      calmar_ratio        — CAGR / |max_drawdown|
      max_drawdown_pct    — maximum peak-to-trough drawdown
      rolling_1yr         — Series of trailing-252-day returns
      benchmark_correlation — filled in by main() after both series computed
    """
    clean = twr.dropna()
    if clean.empty:
        return {}

    total_return = (clean.iloc[-1] / clean.iloc[0] - 1) * 100
    daily_ret = clean.pct_change().dropna()
    vol = daily_ret.std() * np.sqrt(252) * 100
    sharpe = (
        (daily_ret.mean() / daily_ret.std() * np.sqrt(252))
        if daily_ret.std() > 0 else np.nan
    )
    cummax = clean.cummax()
    max_dd = ((clean - cummax) / cummax).min() * 100

    # CAGR
    days = (clean.index[-1] - clean.index[0]).days
    cagr = ((clean.iloc[-1] / clean.iloc[0]) ** (365 / days) - 1) * 100 if days > 0 else np.nan

    # Sortino
    downside = daily_ret[daily_ret < 0]
    sortino = (
        (daily_ret.mean() / downside.std() * np.sqrt(252))
        if len(downside) > 1 and downside.std() > 0 else np.nan
    )

    # Calmar
    calmar = (
        (cagr / abs(max_dd)) if (not np.isnan(cagr) and max_dd < 0) else np.nan
    )

    # Rolling 1-year return (trailing 252 trading days)
    rolling_1yr = clean.pct_change(periods=252).mul(100)

    return {
        "label": label,
        "total_return_pct": round(total_return, 2),
        "cagr_pct": round(cagr, 2) if not np.isnan(cagr) else "n/a",
        "annualised_vol_pct": round(vol, 2),
        "sharpe_ratio": round(sharpe, 3) if not np.isnan(sharpe) else "n/a",
        "sortino_ratio": round(sortino, 3) if not np.isnan(sortino) else "n/a",
        "calmar_ratio": round(calmar, 3) if not np.isnan(calmar) else "n/a",
        "max_drawdown_pct": round(max_dd, 2),
        "rolling_1yr": rolling_1yr,
        "start_date": clean.index[0].date(),
        "end_date": clean.index[-1].date(),
        "trading_days": len(clean),
        "benchmark_correlation": None,  # filled in by caller
    }


# ---------------------------------------------------------------------------
# 11. Holdings table  (§2, §6)
# ---------------------------------------------------------------------------

def build_holdings_table(
    positions: dict[str, float],
    prices: pd.DataFrame,
    fx_rates: pd.DataFrame,
    ticker_currency: dict[str, str],
    isin_to_ticker: dict[str, str],
    isin_to_class: dict[str, str],
    lot_ledger: LotLedger,
) -> pd.DataFrame:
    """
    Build a per-holding breakdown table.

    Columns: ticker, shares, currency, price_native, price_eur,
             value_eur, weight_pct, cost_basis_eur, unrealised_gain_eur,
             asset_class
    """
    last_day = prices.index[-1] if not prices.empty else None
    rows = []
    total_value = 0.0

    ticker_to_isin = {v: k for k, v in isin_to_ticker.items()}

    for ticker, shares in positions.items():
        if shares < 1e-9:
            continue
        if last_day is None or ticker not in prices.columns:
            continue

        px_native = float(prices.at[last_day, ticker]) if last_day in prices.index else np.nan
        cur = ticker_currency.get(ticker, "EUR").upper()
        fx = get_fx_rate(fx_rates, cur, last_day) if last_day else 1.0
        px_eur = price_to_eur(px_native, cur, fx) if pd.notna(px_native) else np.nan
        value_eur = shares * px_eur if pd.notna(px_eur) else np.nan

        if pd.notna(value_eur):
            total_value += value_eur

        isin = ticker_to_isin.get(ticker, "")
        asset_class = isin_to_class.get(isin, "other") if isin else "other"
        cost_basis = lot_ledger.total_cost_basis_eur(ticker)
        unrealised_gain = (value_eur - cost_basis) if pd.notna(value_eur) else np.nan

        rows.append({
            "ticker": ticker,
            "shares": round(shares, 4),
            "currency": cur,
            "price_native": round(px_native, 4) if pd.notna(px_native) else np.nan,
            "price_eur": round(px_eur, 4) if pd.notna(px_eur) else np.nan,
            "value_eur": round(value_eur, 2) if pd.notna(value_eur) else np.nan,
            "cost_basis_eur": round(cost_basis, 2),
            "unrealised_gain_eur": round(unrealised_gain, 2) if pd.notna(unrealised_gain) else np.nan,
            "asset_class": asset_class,
        })

    df = pd.DataFrame(rows)
    if not df.empty and total_value > 0:
        df["weight_pct"] = (df["value_eur"] / total_value * 100).round(2)
    else:
        df["weight_pct"] = np.nan

    return df.sort_values("value_eur", ascending=False).reset_index(drop=True)


def print_holdings_table(holdings: pd.DataFrame) -> None:
    if holdings.empty:
        print("  (no holdings)")
        return
    total_value = holdings["value_eur"].sum()
    total_cost = holdings["cost_basis_eur"].sum()
    total_gain = holdings["unrealised_gain_eur"].sum()

    print(f"\n{'Ticker':<14} {'Shares':>10} {'Cur':>4} {'Px(native)':>12} "
          f"{'Px(EUR)':>10} {'Value(EUR)':>12} {'Weight%':>8} "
          f"{'CostBasis':>12} {'Gain(EUR)':>12} {'AssetClass'}")
    print("-" * 120)
    for _, row in holdings.iterrows():
        print(
            f"{row['ticker']:<14} "
            f"{row['shares']:>10.4f} "
            f"{row['currency']:>4} "
            f"{row['price_native']:>12.4f} "
            f"{row['price_eur']:>10.4f} "
            f"{row['value_eur']:>12.2f} "
            f"{row['weight_pct']:>8.2f} "
            f"{row['cost_basis_eur']:>12.2f} "
            f"{row['unrealised_gain_eur']:>12.2f} "
            f"{row['asset_class']}"
        )
    print("-" * 120)
    print(f"{'TOTAL':<14} {'':>10} {'':>4} {'':>12} {'':>10} "
          f"{total_value:>12.2f} {'100.00':>8} "
          f"{total_cost:>12.2f} {total_gain:>12.2f}")

    # Asset class allocation summary
    alloc = holdings.groupby("asset_class")["value_eur"].sum()
    alloc_pct = (alloc / total_value * 100).round(2) if total_value > 0 else alloc
    print("\n  Asset class allocation:")
    for ac, pct in alloc_pct.sort_values(ascending=False).items():
        print(f"    {ac:<15} {pct:>6.2f}%")


# ---------------------------------------------------------------------------
# 12. Contribution table  (§6)
# ---------------------------------------------------------------------------

def print_contribution_table(
    holdings: pd.DataFrame,
) -> None:
    """
    Print per-holding contribution to total portfolio gain.
    Simplified: contribution = unrealised_gain / total_cost_basis.
    """
    if holdings.empty:
        return
    total_cost = holdings["cost_basis_eur"].sum()
    if total_cost <= 0:
        return

    print("\n  Per-holding contribution to total gain:")
    print(f"  {'Ticker':<14} {'Gain(EUR)':>12} {'Contribution%':>14}")
    print("  " + "-" * 42)
    for _, row in holdings.sort_values("unrealised_gain_eur", ascending=False).iterrows():
        contrib = row["unrealised_gain_eur"] / total_cost * 100
        print(f"  {row['ticker']:<14} {row['unrealised_gain_eur']:>12.2f} {contrib:>14.2f}%")


# ---------------------------------------------------------------------------
# 13. Tax output  (§3f)
# ---------------------------------------------------------------------------

def print_tax_summary(tax_timeline: pd.DataFrame) -> None:
    """Print Austrian KESt yearly summary."""
    if tax_timeline.empty:
        print("\n  (no tax events recorded)")
        return

    # Note: Austrian private investors cannot carry tax losses forward
    # to subsequent years. Losses within the same year CAN offset gains,
    # but that offsetting is not modelled here — each event is reported
    # independently. Cross-year loss carry-forward is out of scope.

    tt = tax_timeline.copy()
    tt["year"] = tt.index.year  # type: ignore[attr-defined]

    print("\n" + "=" * 70)
    print("  AUSTRIAN KESt (27.5%) TAX SIMULATION")
    print("=" * 70)
    print("  Note: Austrian private investors cannot carry losses forward")
    print("  across tax years. Within-year loss offsets are not modelled.")
    print()

    print(f"  {'Year':<6} {'Type':<20} {'Taxable(EUR)':>14} {'KESt(EUR)':>12}")
    print("  " + "-" * 56)

    for year, grp in tt.groupby("year"):
        for etype, eg in grp.groupby("event_type"):
            taxable = eg["taxable_amount"].sum()
            tax = eg["tax"].sum()
            print(f"  {year:<6} {etype:<20} {taxable:>14.2f} {tax:>12.2f}")

    total_tax = tt[tt["event_type"] != "sell_loss"]["tax"].sum()
    print("  " + "=" * 56)
    print(f"  {'TOTAL KESt due':30} {total_tax:>26.2f}")
    print("=" * 70)


# ---------------------------------------------------------------------------
# 14. Visualisation
# ---------------------------------------------------------------------------

def plot_performance(
    portfolio_twr: pd.Series,
    benchmark_twr: pd.Series,
    benchmark_ticker: str,
    buy_and_hold: bool,
    portfolio_df: pd.DataFrame,
    pm_metrics: dict,
    bm_metrics: dict,
    tax_timeline: Optional[pd.DataFrame] = None,
    show_tax: bool = False,
    portfolio_twr_broker: Optional[pd.Series] = None,
) -> None:
    """
    Multi-panel chart:
      Panel 1: TWR comparison (portfolio vs benchmark, plus optional
               broker-only TWR when off-broker assets exist)
      Panel 2: Stacked area — cumulative_invested vs cumulative_gain
      Panel 3: Rolling 1-year return
      Panel 4 (optional, if show_tax): cumulative tax paid
    """
    n_panels = 3 + (1 if show_tax and tax_timeline is not None and not tax_timeline.empty else 0)
    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 4 * n_panels), sharex=True)
    if n_panels == 1:
        axes = [axes]

    ax_twr = axes[0]
    ax_value = axes[1]
    ax_rolling = axes[2]

    # ── Panel 1: TWR ─────────────────────────────────────────────────────────
    ax_twr.plot(portfolio_twr.index, portfolio_twr.values,
                label="Portfolio (incl. off-broker)" if portfolio_twr_broker is not None else "Portfolio",
                linewidth=2)
    if portfolio_twr_broker is not None:
        clean = portfolio_twr_broker.dropna()
        if not clean.empty:
            ax_twr.plot(clean.index, clean.values,
                        label="Portfolio (broker-only)",
                        linewidth=1.8, linestyle="-.", color="#7f3fbf")
    bm_label = f"Benchmark ({benchmark_ticker})"
    bm_label += " [buy-and-hold]" if buy_and_hold else " [mirror flows]"
    ax_twr.plot(benchmark_twr.index, benchmark_twr.values,
                label=bm_label, linewidth=2, linestyle="--")
    ax_twr.axhline(1.0, color="grey", linewidth=0.8, linestyle=":")
    title = "Portfolio vs Benchmark — Time-Weighted Return (TWR)"
    if portfolio_twr_broker is not None:
        title += "\nBroker-only excludes assets held outside the broker (e.g. physical metals, crypto wallets)"
    ax_twr.set_title(title, fontsize=13)
    ax_twr.set_ylabel("TWR Index (start = 1.0)")
    ax_twr.legend()

    # ── Panel 2: Stacked area — invested vs gain ─────────────────────────────
    inv = portfolio_df["cumulative_invested"]
    gain = portfolio_df["cumulative_gain"]
    ax_value.stackplot(
        inv.index,
        [inv.values, np.where(gain.values > 0, gain.values, 0)],
        labels=["Capital Invested (EUR)", "Unrealised Gain (EUR)"],
        colors=["#4878d0", "#6acc65"],
        alpha=0.85,
    )
    # Show loss (negative gain) as a separate fill
    loss = np.where(gain.values < 0, gain.values, 0)
    if loss.min() < 0:
        ax_value.fill_between(inv.index, loss, 0,
                              color="#d65f5f", alpha=0.75, label="Unrealised Loss (EUR)")
    ax_value.set_title("Portfolio Value Decomposition", fontsize=13)
    ax_value.set_ylabel("EUR")
    ax_value.legend(loc="upper left")
    ax_value.axhline(0, color="grey", linewidth=0.5)

    # ── Panel 3: Rolling 1-year return ───────────────────────────────────────
    rolling_pf = pm_metrics.get("rolling_1yr", pd.Series(dtype=float))
    rolling_bm = bm_metrics.get("rolling_1yr", pd.Series(dtype=float))
    if not rolling_pf.empty:
        ax_rolling.plot(rolling_pf.dropna().index, rolling_pf.dropna().values,
                        label="Portfolio rolling 1yr %", linewidth=1.5)
    if not rolling_bm.empty:
        ax_rolling.plot(rolling_bm.dropna().index, rolling_bm.dropna().values,
                        label="Benchmark rolling 1yr %", linewidth=1.5, linestyle="--")
    ax_rolling.axhline(0, color="grey", linewidth=0.8, linestyle=":")
    ax_rolling.set_title("Rolling 1-Year Return", fontsize=13)
    ax_rolling.set_ylabel("Return (%)")
    ax_rolling.legend()

    # ── Panel 4: Cumulative tax (optional) ───────────────────────────────────
    if show_tax and tax_timeline is not None and not tax_timeline.empty and n_panels >= 4:
        ax_tax = axes[3]
        tt = tax_timeline[tax_timeline["event_type"] != "sell_loss"].copy()
        if not tt.empty:
            cum_tax = tt["tax"].cumsum()
            ax_tax.step(cum_tax.index, cum_tax.values, where="post",
                        color="crimson", linewidth=1.5, label="Cumulative KESt (EUR)")
            ax_tax.fill_between(cum_tax.index, cum_tax.values,
                                step="post", alpha=0.25, color="crimson")
        ax_tax.set_title("Cumulative Austrian KESt Paid", fontsize=13)
        ax_tax.set_ylabel("EUR")
        ax_tax.legend()

    for ax in axes:
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
        ax.xaxis.set_major_locator(mdates.AutoDateLocator())

    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 15. Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    app_config = _load_config()
    gap_warn_bdays = app_config.cache.gap_warn_business_days
    refetch_gaps = not args.no_cache_gap_refetch
    use_stooq_fallback = (
        app_config.data_sources.stooq_fallback and not args.no_stooq_fallback
    )

    # ── Clear cache and exit if requested ───────────────────────────────────
    if args.clear_cache:
        if args.cache_dir.exists():
            shutil.rmtree(args.cache_dir)
            print(f"[INFO] Cache cleared: {args.cache_dir}")
        else:
            print(f"[INFO] Cache directory does not exist: {args.cache_dir}")
        return

    cache_dir: Path | None = None if args.no_cache else args.cache_dir

    # ── Load transactions ────────────────────────────────────────────────────
    transactions = load_transactions(args.transaction_file)

    if args.manual_transactions is not None:
        manual_txs = load_manual_transactions(args.manual_transactions)
        transactions = pd.concat([transactions, manual_txs], ignore_index=True)
        transactions = transactions.sort_values("date").reset_index(drop=True)

    # ── Load clearing-account ledger (optional) ──────────────────────────────
    cashflow_ledger: Optional[pd.DataFrame] = None
    ca_cfg = app_config.clearing_account
    if ca_cfg.use_clearing_account:
        ledger_path = Path(ca_cfg.path)
        if not ledger_path.is_absolute():
            ledger_path = (Path(__file__).resolve().parent / ledger_path)
        if ledger_path.exists():
            cashflow_ledger = load_cashflow_ledger(
                ledger_path,
                reference_iban_suffix=ca_cfg.reference_iban_suffix,
                verbose=args.verbose,
            )
            reconcile_cashflow_with_transactions(
                cashflow_ledger, transactions, verbose=args.verbose,
            )
        else:
            warnings.warn(
                f"[WARN] clearing_account.use_clearing_account is true but "
                f"{ledger_path} was not found; falling back to legacy "
                "cash model (buys treated as external injections, no "
                "withdrawals).",
                stacklevel=2,
            )

    # ── Load lookup file ─────────────────────────────────────────────────────
    if args.lookup_file is not None:
        isin_to_ticker, isin_to_currency, isin_to_class = load_lookup(args.lookup_file)
    else:
        isin_to_ticker, isin_to_currency, isin_to_class = {}, {}, {}
        if (transactions.get("ISIN", pd.Series("")) != "").any():
            sys.exit(
                "[ERROR] transactions use ISINs but no lookup file was provided. "
                "Pass the lookup file as the second positional argument."
            )

    # Resolve tickers
    resolved = resolve_tickers(transactions, isin_to_ticker)
    unresolved_mask = resolved.isna()
    if unresolved_mask.any():
        unresolved_ids = sorted({
            (str(row.get("ticker") or "").strip()
             or str(row.get("ISIN") or "").strip()
             or "<empty>")
            for _, row in transactions[unresolved_mask].iterrows()
        })
        sys.exit(
            f"[ERROR] Could not resolve ticker for: {unresolved_ids}. "
            "Either add the ISIN to the lookup file, or provide a direct "
            "Yahoo Finance symbol in the 'ticker' column."
        )

    if args.verbose:
        print("\n[VERBOSE] Ticker resolution map:")
        for isin, ticker in isin_to_ticker.items():
            print(f"  {isin} → {ticker}")

    portfolio_tickers = sorted(set(resolved.tolist()))
    all_tickers = sorted(set(portfolio_tickers + [args.benchmark]))

    # ── Date range ───────────────────────────────────────────────────────────
    start = args.start_date or transactions["date"].min().strftime("%Y-%m-%d")
    end = args.end_date or pd.Timestamp.today().strftime("%Y-%m-%d")

    # ── FX rates ─────────────────────────────────────────────────────────────
    currencies_needed: set[str] = set()
    currencies_needed.update(isin_to_currency.values())
    if "currency" in transactions.columns:
        currencies_needed.update(transactions["currency"].dropna().str.upper().unique())
    currencies_needed.discard("EUR")
    currencies_needed.discard("")
    currencies_needed.discard("NAN")

    if currencies_needed:
        print(f"[INFO] Fetching FX rates for: {sorted(currencies_needed)}")
        fx_rates = fetch_fx_rates(
            currencies_needed,
            start=start,
            end=end,
            cache_dir=cache_dir,
            no_cache=args.no_cache,
            verbose=args.verbose,
            spike_threshold=args.spike_threshold,
            gap_warn_bdays=gap_warn_bdays,
            refetch_gaps=refetch_gaps,
            use_stooq_fallback=use_stooq_fallback,
        )
        if args.verbose and not fx_rates.empty:
            print(f"[VERBOSE] FX rates tail:\n{fx_rates.tail(5)}")
    else:
        fx_rates = pd.DataFrame()

    # ── Price download ───────────────────────────────────────────────────────
    prices = fetch_prices(
        all_tickers,
        start=start,
        end=end,
        cache_dir=cache_dir,
        no_cache=args.no_cache,
        verbose=args.verbose,
        spike_threshold=args.spike_threshold,
        gap_warn_bdays=gap_warn_bdays,
        refetch_gaps=refetch_gaps,
        use_stooq_fallback=use_stooq_fallback,
    )

    if args.benchmark not in prices.columns:
        sys.exit(f"[ERROR] Benchmark ticker '{args.benchmark}' not found in downloaded data.")

    missing_tickers = [t for t in portfolio_tickers if t not in prices.columns]
    if missing_tickers:
        warnings.warn(
            f"[WARN] Missing price data for portfolio ticker(s): {missing_tickers}. "
            "These positions will be excluded from valuation.",
            stacklevel=2,
        )

    # ── Holdings-only mode ───────────────────────────────────────────────────
    if args.holdings:
        # Quick reconstruction just to get positions
        portfolio_df = reconstruct_portfolio(
            transactions, isin_to_ticker, isin_to_currency,
            prices, fx_rates, run_tax=False, verbose=args.verbose,
            cashflow_ledger=cashflow_ledger,
            include_clearing_interest=ca_cfg.include_clearing_interest,
            warn_on_negative_cash=ca_cfg.warn_on_negative_cash,
        )
        holdings = build_holdings_table(
            portfolio_df.final_positions,  # type: ignore[attr-defined]
            prices,
            fx_rates,
            portfolio_df.ticker_currency,  # type: ignore[attr-defined]
            isin_to_ticker,
            isin_to_class,
            portfolio_df.lot_ledger,  # type: ignore[attr-defined]
        )
        print("\n" + "=" * 70)
        print("  CURRENT HOLDINGS")
        print("=" * 70)
        print_holdings_table(holdings)
        return

    # ── Portfolio reconstruction ─────────────────────────────────────────────
    portfolio_df = reconstruct_portfolio(
        transactions, isin_to_ticker, isin_to_currency,
        prices, fx_rates,
        run_tax=args.tax,
        verbose=args.verbose,
        cashflow_ledger=cashflow_ledger,
        include_clearing_interest=ca_cfg.include_clearing_interest,
        warn_on_negative_cash=ca_cfg.warn_on_negative_cash,
    )
    cash_flows: pd.Series = portfolio_df.cash_flows  # type: ignore[attr-defined]

    # ── Optional broker-only reconstruction (excludes off-broker assets) ────
    has_off_broker = bool(transactions.get("is_off_broker", pd.Series(dtype=bool)).any())
    portfolio_df_broker: Optional[pd.DataFrame] = None
    if has_off_broker:
        broker_only_txs = transactions[~transactions["is_off_broker"].astype(bool)].copy()
        if not broker_only_txs.empty:
            portfolio_df_broker = reconstruct_portfolio(
                broker_only_txs, isin_to_ticker, isin_to_currency,
                prices, fx_rates,
                run_tax=False,
                verbose=False,
                cashflow_ledger=cashflow_ledger,
                include_clearing_interest=ca_cfg.include_clearing_interest,
                warn_on_negative_cash=False,
            )

    # ── Benchmark simulation ─────────────────────────────────────────────────
    benchmark_df = simulate_benchmark(cash_flows, prices[args.benchmark], args.buy_and_hold)

    # ── TWR ──────────────────────────────────────────────────────────────────
    pv_twr = compute_twr(portfolio_df["portfolio_value"], portfolio_df["value_before_tx"])
    bv_twr = compute_twr(benchmark_df["benchmark_value"], benchmark_df["value_before_tx"])
    pv_twr_broker: Optional[pd.Series] = None
    if portfolio_df_broker is not None:
        pv_twr_broker = compute_twr(
            portfolio_df_broker["portfolio_value"],
            portfolio_df_broker["value_before_tx"],
        )

    common_idx = pv_twr.dropna().index.intersection(bv_twr.dropna().index)
    if common_idx.empty:
        sys.exit("[ERROR] No overlapping invested days between portfolio and benchmark.")

    pv_twr = pv_twr.loc[common_idx]
    bv_twr = bv_twr.loc[common_idx]

    pv_twr = pv_twr / pv_twr.iloc[0]
    bv_twr = bv_twr / bv_twr.iloc[0]

    if pv_twr_broker is not None:
        pv_twr_broker = pv_twr_broker.reindex(common_idx)
        first_valid = pv_twr_broker.dropna()
        pv_twr_broker = (
            pv_twr_broker / first_valid.iloc[0] if not first_valid.empty else pv_twr_broker
        )

    # ── Verbose diagnostics ──────────────────────────────────────────────────
    if args.verbose:
        diag = pd.DataFrame({
            "pf_value_before":   portfolio_df["value_before_tx"],
            "pf_value_after":    portfolio_df["portfolio_value"],
            "pf_cash":           portfolio_df["cash"],
            "bm_value_before":   benchmark_df["value_before_tx"],
            "bm_value_after":    benchmark_df["benchmark_value"],
            "pf_twr":            pv_twr,
            "bm_twr":            bv_twr,
        })
        for col in prices.columns:
            diag[f"price_{col}"] = prices[col]
        for col in prices.columns:
            diag[f"chg_{col}"] = prices[col].pct_change().mul(100).round(2)

        print("\n" + "=" * 70)
        print("  DIAGNOSTIC: last 20 rows of all intermediate series")
        print("=" * 70)
        pd.set_option("display.max_columns", None)
        pd.set_option("display.width", 200)
        pd.set_option("display.float_format", "{:.4f}".format)
        print(diag.tail(20).to_string())
        print("=" * 70)

        pf_daily = pv_twr.pct_change().abs()
        bm_daily = bv_twr.pct_change().abs()
        large_moves = pf_daily[pf_daily > 0.03].index.union(bm_daily[bm_daily > 0.03].index)
        if not large_moves.empty:
            print("\n  Large TWR moves (>3% in a single day):")
            for d in large_moves:
                print(f"    {d.date()}  pf_twr={pv_twr.get(d, float('nan')):.4f}  "
                      f"bm_twr={bv_twr.get(d, float('nan')):.4f}  "
                      f"pf_before={portfolio_df['value_before_tx'].get(d, float('nan')):.2f}  "
                      f"pf_after={portfolio_df['portfolio_value'].get(d, float('nan')):.2f}")
                for col in prices.columns:
                    chg_key = f"chg_{col}"
                    if chg_key in diag.columns and d in diag.index:
                        print(f"      {col}: price={diag.at[d, f'price_{col}']:.4f}  "
                              f"day_chg={diag.at[d, chg_key]:.2f}%")
        print("=" * 70 + "\n")

    # ── Metrics ──────────────────────────────────────────────────────────────
    pf_label = "Portfolio (incl. off-broker)" if pv_twr_broker is not None else "Portfolio"
    pm = compute_metrics(pv_twr, pf_label)
    bm = compute_metrics(bv_twr, f"Benchmark ({args.benchmark})")
    pm_broker: dict = {}
    if pv_twr_broker is not None and not pv_twr_broker.dropna().empty:
        pm_broker = compute_metrics(pv_twr_broker.dropna(), "Portfolio (broker-only)")

    # Benchmark correlation
    pf_ret = pv_twr.pct_change().dropna()
    bm_ret = bv_twr.pct_change().dropna()
    common_ret = pf_ret.index.intersection(bm_ret.index)
    if len(common_ret) > 1:
        corr = pf_ret.loc[common_ret].corr(bm_ret.loc[common_ret])
        pm["benchmark_correlation"] = round(corr, 4)
        bm["benchmark_correlation"] = 1.0
    else:
        pm["benchmark_correlation"] = "n/a"
        bm["benchmark_correlation"] = "n/a"

    if pm_broker:
        pf_b_ret = pv_twr_broker.dropna().pct_change().dropna()
        common_b = pf_b_ret.index.intersection(bm_ret.index)
        if len(common_b) > 1:
            pm_broker["benchmark_correlation"] = round(
                pf_b_ret.loc[common_b].corr(bm_ret.loc[common_b]), 4
            )
        else:
            pm_broker["benchmark_correlation"] = "n/a"

    # ── Performance summary ──────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("  PERFORMANCE SUMMARY  (Time-Weighted Return)")
    print("=" * 66)
    summary_metrics = [pm]
    if pm_broker:
        summary_metrics.append(pm_broker)
    summary_metrics.append(bm)
    for metrics in summary_metrics:
        print(f"\n  {metrics['label']}")
        print(f"    Period               : {metrics['start_date']} → {metrics['end_date']}")
        print(f"    Trading days         : {metrics['trading_days']}")
        print(f"    Total return         : {metrics['total_return_pct']:>8.2f} %")
        cagr_str = metrics['cagr_pct'] if isinstance(metrics['cagr_pct'], str) else f"{metrics['cagr_pct']:>8.2f} %"
        print(f"    CAGR                 : {cagr_str}")
        print(f"    Annualised vol       : {metrics['annualised_vol_pct']:>8.2f} %")
        print(f"    Sharpe ratio         : {metrics['sharpe_ratio']}")
        print(f"    Sortino ratio        : {metrics['sortino_ratio']}")
        print(f"    Calmar ratio         : {metrics['calmar_ratio']}")
        print(f"    Max drawdown         : {metrics['max_drawdown_pct']:>8.2f} %")
        print(f"    Benchmark correlation: {metrics['benchmark_correlation']}")
    print("=" * 66 + "\n")

    # ── Holdings table ───────────────────────────────────────────────────────
    holdings = build_holdings_table(
        portfolio_df.final_positions,  # type: ignore[attr-defined]
        prices,
        fx_rates,
        portfolio_df.ticker_currency,  # type: ignore[attr-defined]
        isin_to_ticker,
        isin_to_class,
        portfolio_df.lot_ledger,  # type: ignore[attr-defined]
    )

    print("\n" + "=" * 70)
    print("  CURRENT HOLDINGS")
    print("=" * 70)
    print_holdings_table(holdings)
    print_contribution_table(holdings)

    # ── Portfolio value summary ──────────────────────────────────────────────
    last_invested = portfolio_df["cumulative_invested"].iloc[-1]
    last_value = portfolio_df["portfolio_value"].iloc[-1]
    last_gain = portfolio_df["cumulative_gain"].iloc[-1]
    gain_pct = last_gain / last_invested * 100 if last_invested > 0 else 0.0

    print(f"\n  Capital invested (gross) : EUR {last_invested:,.2f}")
    print(f"  Portfolio value          : EUR {last_value:,.2f}")
    print(f"  Total gain/loss          : EUR {last_gain:,.2f} ({gain_pct:.2f}%)")
    print(f"  Cash held                : EUR {portfolio_df['cash'].iloc[-1]:,.2f}")

    if cashflow_ledger is not None:
        last_contributed = float(portfolio_df["cumulative_contributed"].iloc[-1])
        net_gain = float(last_value) - last_contributed
        net_gain_pct = (net_gain / last_contributed * 100) if last_contributed > 0 else 0.0
        print(f"  Net contributed          : EUR {last_contributed:,.2f}")
        print(f"    (deposits − withdrawals; net of clearing-account flows)")
        print(f"  Total return on net      : EUR {net_gain:,.2f} ({net_gain_pct:.2f}%)")

        external_flows: pd.Series = portfolio_df.external_flows  # type: ignore[attr-defined]
        if not external_flows.empty:
            mwr = compute_xirr(
                external_flows,
                final_value=float(last_value),
                final_date=portfolio_df.index[-1],
            )
            if mwr is not None:
                print(f"  Money-weighted return    : {mwr * 100:>8.2f} %/yr (XIRR)")
            else:
                print(f"  Money-weighted return    : n/a (failed to converge)")

    # ── Tax simulation output ────────────────────────────────────────────────
    if args.tax:
        print_tax_summary(portfolio_df.tax_timeline)  # type: ignore[attr-defined]

    # ── Optional CSV export ──────────────────────────────────────────────────
    if args.export_csv:
        out = pd.DataFrame({
            "portfolio_twr": pv_twr,
            "benchmark_twr": bv_twr,
            "cumulative_invested": portfolio_df["cumulative_invested"],
            "cumulative_contributed": portfolio_df["cumulative_contributed"],
            "portfolio_value_eur": portfolio_df["portfolio_value"],
            "cash_eur": portfolio_df["cash"],
            "cumulative_gain": portfolio_df["cumulative_gain"],
        })
        out.to_csv(args.export_csv)
        print(f"\n[INFO] Time series exported to {args.export_csv}")

    # ── Charts ───────────────────────────────────────────────────────────────
    if not args.no_plot:
        plot_performance(
            pv_twr, bv_twr, args.benchmark, args.buy_and_hold,
            portfolio_df, pm, bm,
            tax_timeline=portfolio_df.tax_timeline,  # type: ignore[attr-defined]
            show_tax=args.tax,
            portfolio_twr_broker=pv_twr_broker,
        )


if __name__ == "__main__":
    main()
