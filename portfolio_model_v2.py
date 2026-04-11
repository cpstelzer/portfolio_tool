"""
portfolio_model.py (v2 Draft)
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from dataclasses import dataclass
import os
import shutil
import re

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
    parser.add_argument("transaction_file", type=Path, help="Path to transaction CSV")
    parser.add_argument(
        "lookup_file",
        type=Path,
        nargs="?",
        default=None,
        help="Path to isin_to_yahoo.csv (optional).",
    )
    parser.add_argument("--benchmark", default="VWCE.DE", help="Yahoo Finance ticker for the benchmark ETF")
    parser.add_argument("--buy_and_hold", type=lambda x: x.lower() in ("true", "1", "yes"), default=True)
    parser.add_argument("--start_date", default=None)
    parser.add_argument("--end_date", default=None)
    parser.add_argument("--export_csv", default=None, type=Path)
    
    # New V2 arguments
    parser.add_argument("--manual-transactions", default=None, type=Path)
    parser.add_argument("--base-currency", default="EUR")
    parser.add_argument("--tax", action="store_true", help="Enable Austrian KESt simulation")
    parser.add_argument("--cache-dir", default="~/.portfolio_cache/", type=Path)
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--clear-cache", action="store_true")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--holdings", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--spike-threshold-eq", type=float, default=0.15)
    parser.add_argument("--spike-threshold-crypto", type=float, default=0.50)
    
    return parser.parse_args(argv)

# ---------------------------------------------------------------------------
# 2. Data loading & validation
# ---------------------------------------------------------------------------

REQUIRED_LOOKUP_COLS = {"ISIN", "YahooTicker"}

def _classify_tx_type(booking_info: str | float, shares: float) -> str:
    if pd.isna(booking_info):
        return "buy" if shares > 0 else "sell"
    bi = str(booking_info).lower()
    if "kauf" in bi:
        return "buy"
    if "verkauf" in bi:
        return "sell"
    if "thesaurierung" in bi:
        return "thesaurierung"
    if "fusion" in bi:
        return "fusion"
    if "ausschüttung" in bi or "ausschuettung" in bi or "erträgnisausschüttung" in bi:
        return "ausschuettung"
    if "spin-off" in bi or "kapitalmaßnahme" in bi:
        return "corporate_action"
    
    return "buy" if shares > 0 else "sell"


def load_transactions(path: Path, manual_path: Path | None, verbose: bool) -> pd.DataFrame:
    try:
        df = pd.read_csv(path, encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv(path, encoding='latin-1', sep=',')
        
    df.columns = [str(c).strip() for c in df.columns]

    broker_mapping = {
        "Buchungstag": "date",
        "Bezeichnung": "name",
        "ISIN": "ISIN",
        "Nominal (Stk.)": "shares",
        "Betrag": "gross_amount",
        "Kurs": "price",
        "Devisenkurs": "fx_rate",
        "TA.-Nr.": "tx_id",
        "Buchungsinformation": "booking_info",
    }

    if "Buchungstag" in df.columns:
        # Broker format directly
        rename_dict = {col: expected for col, expected in broker_mapping.items() if col in df.columns}
        df = df.rename(columns=rename_dict)
        # Drop columns not in the internal schema (skip unit columns)
        keep_cols = list(broker_mapping.values())
        keep_cols_present = [c for c in keep_cols if c in df.columns]
        df = df[keep_cols_present]
        # Remove empty rows or completely NaN
        df = df.dropna(how='all')
    else:
        # Simplified/backward compatible format
        pass

    # Standardize columns that may be missing
    for col in ("ISIN", "ticker", "currency", "booking_info"):
        if col not in df.columns:
            df[col] = ""
    for col in ("shares", "price", "gross_amount", "fx_rate"):
        if col not in df.columns:
            df[col] = np.nan

    # Manual transactions
    if manual_path and manual_path.exists():
        df_manual = pd.read_csv(manual_path)
        for col in ("ISIN", "ticker", "currency", "booking_info"):
            if col not in df_manual.columns:
                df_manual[col] = ""
        for col in ("shares", "price", "gross_amount", "fx_rate"):
            if col not in df_manual.columns:
                df_manual[col] = np.nan
        df = pd.concat([df, df_manual], ignore_index=True)

    df["ISIN"] = df["ISIN"].fillna("").astype(str).str.strip()
    df["ticker"] = df["ticker"].fillna("").astype(str).str.strip()
    
    empty_ids = (df["ISIN"] == "") & (df["ticker"] == "")
    if empty_ids.any():
        bad = (df.index[empty_ids] + 2).tolist()
        sys.exit(f"[ERROR] transaction rows {bad} have neither 'ISIN' nor 'ticker' set.")

    raw_dates = df["date"].astype(str).str.strip()
    iso = pd.to_datetime(raw_dates, format="%Y-%m-%d", errors="coerce")
    eu = pd.to_datetime(raw_dates, format="%d.%m.%Y", errors="coerce")
    df["date"] = iso.fillna(eu)
    if df["date"].isna().any():
        bad = (df.index[df["date"].isna()] + 2).tolist()
        sys.exit(f"[ERROR] Unparseable dates in rows {bad}.")

    def parse_num(val):
        if isinstance(val, str):
            val = val.replace(".", "").replace(",", ".")
        return float(val) if val != "" and pd.notna(val) else np.nan

    # In broker export, numbers are usually german format (12.345,67). But read_csv may have messed it up.
    # Let's clean standard strings
    df["shares"] = df["shares"].apply(parse_num)
    df["price"] = df["price"].apply(parse_num)
    df["gross_amount"] = df["gross_amount"].apply(parse_num)
    if df["fx_rate"].dtype == object:
        df["fx_rate"] = df["fx_rate"].apply(parse_num)

    df["tx_type"] = df.apply(lambda row: _classify_tx_type(row["booking_info"], row["shares"]), axis=1)

    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_lookup(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_LOOKUP_COLS - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] lookup file missing columns: {missing}")
    if "Currency" not in df.columns:
        df["Currency"] = ""
    if "asset_class" not in df.columns:
        df["asset_class"] = "other"
    return df

def resolve_tickers(transactions: pd.DataFrame, isin_to_ticker: dict[str, str]) -> pd.Series:
    def _resolve(row: pd.Series) -> str | None:
        direct = str(row.get("ticker", "") or "").strip()
        if direct: return direct
        isin = str(row.get("ISIN", "") or "").strip()
        if isin: return isin_to_ticker.get(isin)
        return None
    return transactions.apply(_resolve, axis=1)


# ---------------------------------------------------------------------------
# 3. Price & FX retrieval
# ---------------------------------------------------------------------------

_CRYPTO_SUFFIXES = ("-USD", "-EUR", "-GBP", "-USDT", "-BTC", "-ETH", "-JPY")

def _is_crypto_ticker(ticker: str) -> bool:
    upper = ticker.upper()
    return any(upper.endswith(suf) for suf in _CRYPTO_SUFFIXES)

def fetch_yfinance_cached(tickers: list[str], start: str, end: str, cache_dir: Path, no_cache: bool) -> pd.DataFrame:
    cache_dir = Path(os.path.expanduser(str(cache_dir)))
    if no_cache:
        pass
    else:
        cache_dir.mkdir(parents=True, exist_ok=True)
    
    frames = []
    
    if not tickers:
        return pd.DataFrame()
        
    for ticker in tickers:
        cache_file = cache_dir / f"{ticker.replace('=', '_')}.csv"
        df_cached = pd.DataFrame()
        
        needed_start = start
        if not no_cache and cache_file.exists():
            df_cached = pd.read_csv(cache_file, index_col=0, parse_dates=True)
            if not df_cached.empty:
                last_cached = df_cached.index.max().strftime("%Y-%m-%d")
                needed_start = (df_cached.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
                
        # If needed_start is still before end, download
        if pd.to_datetime(needed_start) <= pd.to_datetime(end):
            raw = yf.download(ticker, start=needed_start, end=end, auto_adjust=True, progress=False, threads=True)
            if not raw.empty:
                if isinstance(raw.columns, pd.MultiIndex):
                    px = raw["Close"].copy()
                else:
                    px = raw[["Close"]].copy()
                px.columns = [ticker]
                
                if not no_cache:
                    combined = pd.concat([df_cached, px])
                    combined = combined[~combined.index.duplicated(keep="last")]
                    combined.to_csv(cache_file)
                    df_cached = combined
            elif df_cached.empty:
                df_cached = pd.DataFrame(columns=[ticker])
        else:
            # We already have all needed data
            pass
            
        frames.append(df_cached)
        
    if not frames:
        return pd.DataFrame()
        
    res = pd.concat(frames, axis=1)
    # Ensure it's indexed properly
    res = res.loc[start:end]
    return res


def fetch_prices(tickers: list[str], start: str, end: str, args: argparse.Namespace) -> pd.DataFrame:
    if args.verbose:
        print(f"[INFO] Fetching prices for {len(tickers)} ticker(s) from {start} to {end} …")
        
    prices = fetch_yfinance_cached(tickers, start, end, args.cache_dir, args.no_cache)
    
    if prices.empty:
        sys.exit("[ERROR] yfinance returned no data. Check tickers and date range.")

    frac_missing = prices.isna().mean()
    bad = frac_missing[frac_missing > 0.85].index.tolist()
    if bad:
        warnings.warn(f"[WARN] Dropping tickers with insufficient data: {bad}", stacklevel=2)
        prices = prices.drop(columns=bad)

    prices = prices.ffill()

    # Spike detection - Reversal approach
    chg = prices.pct_change()
    for col in prices.columns:
        threshold = args.spike_threshold_crypto if _is_crypto_ticker(col) else args.spike_threshold_eq
        
        # Candidate spikes
        candidates = chg.index[chg[col].abs() > threshold].tolist()
        confirmed_spikes = []
        for t in candidates:
            # check reversal on t+1
            idx_pos = chg.index.get_loc(t)
            if idx_pos + 1 < len(chg):
                t_next = chg.index[idx_pos + 1]
                r_c = chg[col].iloc[idx_pos]
                r_n = chg[col].iloc[idx_pos + 1]
                if pd.notna(r_c) and pd.notna(r_n):
                    if (r_c * r_n < 0) and (abs(r_n) > 0.5 * abs(r_c)):
                        confirmed_spikes.append(t)
        
        if confirmed_spikes:
            if args.verbose:
                print(f"[INFO] {col}: Reversal-based spikes detected on {[d.date() for d in confirmed_spikes]}. Interpolating.")
            prices.loc[confirmed_spikes, col] = np.nan
            
    prices = prices.interpolate(method="time").ffill().bfill()

    if len(prices) > 1:
        prices = prices.iloc[:-1]

    return prices


def get_fx_rates(currencies: set[str], start: str, end: str, base: str, args) -> pd.DataFrame:
    pairs = []
    for cur in currencies:
        cur = str(cur).strip().upper()
        if cur and cur != base and cur != "NAN" and cur != "":
            pairs.append(f"{base}{cur}=X")  # e.g. EURUSD=X if base is EUR
            
    if not pairs:
        # Return a completely filled dataframe with 1.0 for the base? 
        # Easier to just not have it if not needed, or return empty
        return pd.DataFrame()
        
    rates = fetch_yfinance_cached(pairs, start, end, args.cache_dir, args.no_cache)
    rates = rates.ffill().bfill()
    return rates

if __name__ == "__main__":
    pass
