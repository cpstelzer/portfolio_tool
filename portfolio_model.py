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
  • Buys are modelled as external capital injection: money enters the
    portfolio and is immediately converted to shares. Net cash effect
    of a buy = 0; cumulative_invested increases by the cost.
  • Sells convert shares to cash that remains inside the portfolio.
    Cash only grows (from sell proceeds) and is never withdrawn.
  • Ausschüttung (cash distributions) are added to portfolio cash and
    do NOT reduce cumulative_invested; they represent income, not a
    return of capital.
  • No withdrawals are modelled. If a withdrawal transaction type is
    ever added it would reduce both cash and cumulative_invested, but
    that is out of scope for v2.
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
import yfinance as yf
import matplotlib.pyplot as plt
import matplotlib.dates as mdates


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
        help="Run Austrian KESt (27.5%) tax simulation and print yearly summary",
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
        default=Path.home() / ".portfolio_cache",
        dest="cache_dir",
        help="Directory for local price cache (default: ~/.portfolio_cache/)",
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
        sys.exit("[ERROR] yfinance returned no data. Check tickers and date range.")

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


def fetch_prices(
    tickers: list[str],
    start: str,
    end: str,
    cache_dir: Path | None = None,
    no_cache: bool = False,
    verbose: bool = False,
    spike_threshold: float = 0.15,
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
    need_download: dict[str, tuple[str, str]] = {}  # ticker → (dl_start, dl_end)

    for ticker in tickers:
        if cache_dir is not None and not no_cache:
            cached = _load_cache(cache_dir, ticker)
            if cached is not None and not cached.empty and "close" in cached.columns:
                last_cached = cached.index.max()
                target_end = pd.Timestamp(end)
                if last_cached >= target_end - pd.Timedelta(days=2):
                    # Cache is up to date
                    s = cached["close"].loc[start:]
                    all_frames[ticker] = s
                    if verbose:
                        print(f"[VERBOSE] Cache hit for {ticker}: {len(s)} rows "
                              f"(last={last_cached.date()})")
                    continue
                else:
                    # Partial cache — download the missing tail
                    dl_start = (last_cached + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                    need_download[ticker] = (dl_start, end)
                    all_frames[ticker] = cached["close"]
                    if verbose:
                        print(f"[VERBOSE] Partial cache for {ticker}; "
                              f"downloading from {dl_start}")
                    continue
        need_download[ticker] = (start, end)

    # Batch download needed tickers grouped by (start, end)
    if need_download:
        # Group tickers with the same date range for efficiency
        ranges: dict[tuple[str, str], list[str]] = {}
        for ticker, (dl_s, dl_e) in need_download.items():
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

                if cache_dir is not None and not no_cache:
                    existing = all_frames.get(ticker)
                    if existing is not None and not existing.empty:
                        combined = pd.concat([existing, new_series])
                        combined = combined[~combined.index.duplicated(keep="last")]
                        combined = combined.sort_index()
                        all_frames[ticker] = combined
                        _save_cache(cache_dir, ticker,
                                    pd.DataFrame({"close": combined}))
                    else:
                        all_frames[ticker] = new_series
                        _save_cache(cache_dir, ticker,
                                    pd.DataFrame({"close": new_series}))
                else:
                    existing = all_frames.get(ticker)
                    if existing is not None and not existing.empty:
                        combined = pd.concat([existing, new_series])
                        combined = combined[~combined.index.duplicated(keep="last")]
                        all_frames[ticker] = combined.sort_index()
                    else:
                        all_frames[ticker] = new_series

    if not all_frames:
        sys.exit("[ERROR] No price data available after cache/download.")

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
) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by trading date with columns:
        value_before_tx    — portfolio EUR value before today's transactions
        portfolio_value    — portfolio EUR value after today's transactions
        cash               — EUR cash held
        cumulative_invested — running sum of capital injected (EUR)
        cumulative_gain    — portfolio_value − cumulative_invested

    Attributes on the returned DataFrame:
        .cash_flows     — pd.Series of net capital deployed per day
        .lot_ledger     — LotLedger (for §3 tax simulation)
        .tax_timeline   — pd.DataFrame (filled only when run_tax=True)
        .final_positions — dict ticker → shares
        .ticker_currency — dict ticker → currency
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

    # ── Main day loop ────────────────────────────────────────────────────────
    positions: dict[str, float] = {}   # ticker → shares (aggregate)
    cash = 0.0
    cumulative_invested = 0.0
    cash_flows: dict[pd.Timestamp, float] = {}
    lot_ledger = LotLedger()
    tax_events: list[dict] = []
    cumulative_tax = 0.0
    rows_out = []

    for day in prices.index:
        value_before_tx = cash + _mark_to_market(positions, prices, day, fx_rates, ticker_currency)

        day_invested = 0.0

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

                if tx_type == "buy" or (tx_type not in ("sell",) and shares > 0):
                    # External capital injection
                    cash += cost_eur
                    cumulative_invested += cost_eur
                    day_invested += cost_eur
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

                cash -= cost_eur   # buy: nets to 0; sell: cash increases
                positions[ticker] = positions.get(ticker, 0.0) + shares

        if day_invested != 0:
            cash_flows[day] = cash_flows.get(day, 0.0) + day_invested

        value_after_tx = cash + _mark_to_market(positions, prices, day, fx_rates, ticker_currency)
        cumulative_gain = value_after_tx - cumulative_invested

        rows_out.append({
            "date": day,
            "value_before_tx": value_before_tx,
            "portfolio_value": value_after_tx,
            "cash": cash,
            "cumulative_invested": cumulative_invested,
            "cumulative_gain": cumulative_gain,
        })

    result = pd.DataFrame(rows_out).set_index("date")

    # Attach extra data as attributes
    result.cash_flows = pd.Series(cash_flows, name="cash_flow").sort_index()  # type: ignore[attr-defined]
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
) -> None:
    """
    Multi-panel chart:
      Panel 1: TWR comparison (portfolio vs benchmark)
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
                label="Portfolio", linewidth=2)
    bm_label = f"Benchmark ({benchmark_ticker})"
    bm_label += " [buy-and-hold]" if buy_and_hold else " [mirror flows]"
    ax_twr.plot(benchmark_twr.index, benchmark_twr.values,
                label=bm_label, linewidth=2, linestyle="--")
    ax_twr.axhline(1.0, color="grey", linewidth=0.8, linestyle=":")
    ax_twr.set_title("Portfolio vs Benchmark — Time-Weighted Return (TWR)", fontsize=13)
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
    )
    cash_flows: pd.Series = portfolio_df.cash_flows  # type: ignore[attr-defined]

    # ── Benchmark simulation ─────────────────────────────────────────────────
    benchmark_df = simulate_benchmark(cash_flows, prices[args.benchmark], args.buy_and_hold)

    # ── TWR ──────────────────────────────────────────────────────────────────
    pv_twr = compute_twr(portfolio_df["portfolio_value"], portfolio_df["value_before_tx"])
    bv_twr = compute_twr(benchmark_df["benchmark_value"], benchmark_df["value_before_tx"])

    common_idx = pv_twr.dropna().index.intersection(bv_twr.dropna().index)
    if common_idx.empty:
        sys.exit("[ERROR] No overlapping invested days between portfolio and benchmark.")

    pv_twr = pv_twr.loc[common_idx]
    bv_twr = bv_twr.loc[common_idx]

    pv_twr = pv_twr / pv_twr.iloc[0]
    bv_twr = bv_twr / bv_twr.iloc[0]

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
    pm = compute_metrics(pv_twr, "Portfolio")
    bm = compute_metrics(bv_twr, f"Benchmark ({args.benchmark})")

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

    # ── Performance summary ──────────────────────────────────────────────────
    print("\n" + "=" * 66)
    print("  PERFORMANCE SUMMARY  (Time-Weighted Return)")
    print("=" * 66)
    for metrics in (pm, bm):
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

    print(f"\n  Capital invested : EUR {last_invested:,.2f}")
    print(f"  Portfolio value  : EUR {last_value:,.2f}")
    print(f"  Total gain/loss  : EUR {last_gain:,.2f} ({gain_pct:.2f}%)")
    print(f"  Cash held        : EUR {portfolio_df['cash'].iloc[-1]:,.2f}")

    # ── Tax simulation output ────────────────────────────────────────────────
    if args.tax:
        print_tax_summary(portfolio_df.tax_timeline)  # type: ignore[attr-defined]

    # ── Optional CSV export ──────────────────────────────────────────────────
    if args.export_csv:
        out = pd.DataFrame({
            "portfolio_twr": pv_twr,
            "benchmark_twr": bv_twr,
            "cumulative_invested": portfolio_df["cumulative_invested"],
            "portfolio_value_eur": portfolio_df["portfolio_value"],
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
        )


if __name__ == "__main__":
    main()
