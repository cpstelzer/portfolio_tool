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
    if "ausschÃ¼ttung" in bi or "ausschuettung" in bi or "ertrÃ¤gnisausschÃ¼ttung" in bi:
        return "ausschuettung"
    if "spin-off" in bi or "kapitalmaÃŸnahme" in bi:
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

    if verbose:
        print("[INFO] Transaction Type Classification Summary:")
        print(df["tx_type"].value_counts().to_string())

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
        print(f"[INFO] Fetching prices for {len(tickers)} ticker(s) from {start} to {end} â€¦")
        
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



@dataclass
class Lot:
    isin: str
    ticker: str
    date: pd.Timestamp
    shares: float
    cost_price_eur: float
    thesaurierung_adj: float = 0.0


def _snap_to_trading_day(dates: pd.Series, trading_days: pd.DatetimeIndex) -> pd.Series:
    def snap(d: pd.Timestamp) -> pd.Timestamp | None:
        idx = trading_days.searchsorted(d)
        if idx >= len(trading_days): return None
        return trading_days[idx]
    return dates.map(snap)


def _mark_to_market(positions: dict[str, float], prices: pd.DataFrame, day: pd.Timestamp, base: str, lookup_df: pd.DataFrame, transactions: pd.DataFrame, fx_rates: pd.DataFrame, daily_alloc: dict[str, float] | None = None) -> float:
    # 1. Identify currency for each ticker
    val_eur = 0.0
    for t, sh in positions.items():
        if sh == 0 or t not in prices.columns or pd.isna(prices.at[day, t]):
            continue
        px = float(prices.at[day, t])
        
        # look up currency
        cur = ""
        # From lookup_df first?
        row_look = lookup_df[lookup_df["YahooTicker"] == t]
        if not row_look.empty:
            cur = row_look.iloc[0]["Currency"]
        if not cur:
            # fallback to transaction
            tx_look = transactions[transactions["ticker"] == t]
            if not tx_look.empty:
                cur = tx_look.iloc[0]["currency"]
        
        cur = str(cur).strip().upper()
        rate = 1.0
        if cur and cur != base and cur != "NAN" and cur != "":
            pair = f"{base}{cur}=X"
            if pair in fx_rates.columns and not pd.isna(fx_rates.at[day, pair]):
                rate = float(fx_rates.at[day, pair])
                
        # rate is BASE_CUR (e.g. EURUSD=X means 1 EUR = X USD). So px_eur = px / rate
        px_eur = px / rate if rate > 0 else px
        val_tmp = sh * px_eur
        val_eur += val_tmp
        if daily_alloc is not None:
            ac = get_asset_class(t, lookup_df)
            daily_alloc[ac] = daily_alloc.get(ac, 0.0) + val_tmp
        
    return val_eur
        

def get_asset_class(ticker: str, lookup_df: pd.DataFrame) -> str:
    row_look = lookup_df[lookup_df["YahooTicker"] == ticker]
    if not row_look.empty:
        return row_look.iloc[0]["asset_class"]
    return "other"

def reconstruct_portfolio(transactions: pd.DataFrame, isin_to_ticker: dict[str, str], lookup_df: pd.DataFrame, prices: pd.DataFrame, fx_rates: pd.DataFrame, args: argparse.Namespace) -> pd.DataFrame:
    transactions = transactions.copy()
    transactions["_yf_ticker"] = resolve_tickers(transactions, isin_to_ticker)
    
    transactions = transactions.dropna(subset=["_yf_ticker"])
    original_dates = transactions["date"].copy()
    transactions["date"] = _snap_to_trading_day(transactions["date"], prices.index)
    
    transactions = transactions.dropna(subset=["date"])
    
    tx_by_date = {}
    for _, row in transactions.iterrows():
        tx_by_date.setdefault(row["date"], []).append(row)
        
    positions: dict[str, float] = {}
    lots: list[Lot] = []
    tax_timeline = []
    
    """
    Assumptions logic:
    - Buys are modelled as external capital injection: money enters the portfolio and is immediately converted to shares. Net cash effect of a buy = 0.
    - Sells convert shares to cash. Cash remains in the portfolio.
    - No withdrawals. Cash only grows (from sells) and is never removed.
    - TWR uses Yahoo adjusted close (dividends reinvested in price). The cash tracking and tax simulation use actual distribution amounts from broker data.
    """
    
    cash = 0.0
    cumulative_invested = 0.0
    cumulative_tax = 0.0
    cash_flows = {}
    rows = []
    allocation_history = []
    
    # Track distributions/thes for verbose logging
    
    base_cur = args.base_currency.upper()
    
    for day in prices.index:
        val_before = cash + _mark_to_market(positions, prices, day, base_cur, lookup_df, transactions, fx_rates)
        
        day_invested = 0.0
        
        if day in tx_by_date:
            daily_txs = tx_by_date[day]
            
            # Pre-group thesaurierung manually if possible
            thes_txs = [t for t in daily_txs if t["tx_type"] == "thesaurierung"]
            regular_txs = [t for t in daily_txs if t["tx_type"] != "thesaurierung"]
            
            # Process Thesaurierung pairs
            matched_thes = []
            unmatched = list(thes_txs)
            while len(unmatched) >= 2:
                r1 = unmatched.pop(0)
                found = False
                for i, r2 in enumerate(unmatched):
                    if r1["ISIN"] == r2["ISIN"] and abs(r1["shares"]) == abs(r2["shares"]) and r1["shares"] * r2["shares"] < 0:
                        r_neg, r_pos = (r1, r2) if r1["shares"] < 0 else (r2, r1)
                        found = True
                        unmatched.pop(i)
                        
                        px_diff = (abs(r_pos["gross_amount"]) - abs(r_neg["gross_amount"])) 
                        if px_diff > 0:
                            amount_tax = px_diff
                            tax = amount_tax * 0.275
                            tax_adj_per_share = amount_tax / abs(r_pos["shares"])
                            
                            if args.tax:
                                cumulative_tax += tax
                                tax_timeline.append({
                                    "date": day, "event_type": "thesaurierung", "ticker": r_pos["_yf_ticker"], 
                                    "taxable_amount": amount_tax, "tax": tax, "cumulative_tax": cumulative_tax
                                })
                            
                            # Add to lots
                            t_lots = [l for l in lots if l.ticker == r_pos["_yf_ticker"] and l.shares > 1e-6]
                            total_target = abs(r_pos["shares"])
                            for l in t_lots:
                                if total_target <= 1e-6: break
                                l.thesaurierung_adj += tax_adj_per_share
                                total_target -= min(total_target, l.shares)
                                
                        break
                if not found:
                    pass # ignore unpaired thesaurierung
                    
            # Process regular transactions
            for tx in regular_txs:
                ticker = tx["_yf_ticker"]
                isin = tx.get("ISIN", "")
                shares = float(tx["shares"]) if pd.notna(tx["shares"]) else 0.0
                tx_t = tx["tx_type"]
                
                if tx_t == "fusion":
                    new_isin = isin # simplified, in real cases you'd map.
                    for l in lots:
                        if l.isin == isin:
                            l.ticker = ticker
                    continue
                    
                if tx_t == "ausschuettung":
                    amt = float(tx.get("gross_amount", 0.0))
                    if amt > 0:
                        if args.tax:
                            tax = amt * 0.275
                            cumulative_tax += tax
                            tax_timeline.append({
                                "date": day, "event_type": "ausschuettung", "ticker": ticker, 
                                "taxable_amount": amt, "tax": tax, "cumulative_tax": cumulative_tax
                            })
                            cash += (amt - tax) # Net cash received
                        else:
                            cash += amt
                    continue

                if shares == 0: continue

                px = np.nan
                if pd.notna(tx.get("price")):
                    px = float(tx["price"])
                elif pd.notna(tx.get("gross_amount")) and shares != 0:
                    px = abs(float(tx["gross_amount"])) / abs(shares)
                elif ticker in prices.columns and not pd.isna(prices.at[day, ticker]):
                    px = float(prices.at[day, ticker])
                    
                if pd.isna(px):
                    continue
                    
                # Fix currencies for cost basis
                cur = tx.get("currency", "")
                rate_to_eur = 1.0 # EUR/target. if base=EUR, e.g. EURUSD=X. target/base
                if pd.notna(tx.get("fx_rate")) and float(tx["fx_rate"]) != 0:
                    rate_to_eur = float(tx["fx_rate"])
                else:    
                    row_look = lookup_df[lookup_df["YahooTicker"] == ticker]
                    if not row_look.empty and row_look.iloc[0]["Currency"]:
                        cur = row_look.iloc[0]["Currency"]
                        
                    cur = str(cur).strip().upper()
                    if cur and cur != base_cur and cur != "NAN" and cur != "":
                        pair = f"{base_cur}{cur}=X"
                        if pair in fx_rates.columns and not pd.isna(fx_rates.at[day, pair]):
                            rate_to_eur = float(fx_rates.at[day, pair])
                            
                px_eur = px / rate_to_eur if rate_to_eur > 0 else px
                cost_eur = abs(shares) * px_eur
                
                if pd.notna(tx.get("gross_amount")):
                    broker_gross = abs(float(tx["gross_amount"]))
                    if broker_gross > 0:
                        diff_pct = abs(cost_eur - broker_gross) / broker_gross
                        if diff_pct > 0.02:
                            import warnings
                            warnings.warn(f"[WARN] {day.date()} {ticker}: Derived EUR value ({cost_eur:.2f}) diverges from broker gross_amount ({broker_gross:.2f}) by >2%!")
                
                if tx_t == "buy":
                    cash += cost_eur
                    cash -= cost_eur
                    positions[ticker] = positions.get(ticker, 0.0) + shares
                    day_invested += cost_eur
                    cumulative_invested += cost_eur
                    lots.append(Lot(isin=isin, ticker=ticker, date=day, shares=shares, cost_price_eur=px_eur))
                    
                elif tx_t == "sell":
                    sell_shares = abs(shares)
                    cash += cost_eur
                    positions[ticker] = max(0.0, positions.get(ticker, 0.0) - sell_shares)
                    
                    matching_lots = [l for l in lots if l.ticker == ticker and l.shares > 1e-6]
                    matching_lots.sort(key=lambda x: x.date)
                    
                    for l in matching_lots:
                        if sell_shares <= 1e-6: break
                        consume = min(l.shares, sell_shares)
                        l.shares -= consume
                        sell_shares -= consume
                        
                        adj_basis_px = l.cost_price_eur + l.thesaurierung_adj
                        gain = (px_eur - adj_basis_px) * consume
                        
                        cumulative_invested -= (consume * l.cost_price_eur)
                        
                        if gain > 0 and args.tax:
                            tax = gain * 0.275
                            cumulative_tax += tax
                            cash -= tax # pay tax out of cash
                            tax_timeline.append({
                                "date": day, "event_type": "sell", "ticker": ticker, 
                                "taxable_amount": gain, "tax": tax, "cumulative_tax": cumulative_tax
                            })

        if day_invested != 0:
            cash_flows[day] = cash_flows.get(day, 0.0) + day_invested
            
        daily_alloc = {"cash": cash}
        val_after = cash + _mark_to_market(positions, prices, day, base_cur, lookup_df, transactions, fx_rates, daily_alloc)
        allocation_history.append({"date": day, **daily_alloc})
        
        rows.append({
            "date": day,
            "value_before_tx": val_before,
            "portfolio_value": val_after,
            "cash": cash,
            "cumulative_invested": cumulative_invested
        })

    result = pd.DataFrame(rows).set_index("date")
    result.cash_flows = pd.Series(cash_flows, name="cash_flow").sort_index()
    if args.tax:
        result.tax_timeline = pd.DataFrame(tax_timeline)
    else:
        result.tax_timeline = pd.DataFrame()
        
    # Positions dump for end result
    result.final_positions = {k: v for k, v in positions.items() if v > 1e-6}
    
    result.allocation_history = pd.DataFrame(allocation_history).set_index("date").fillna(0.0)
    
    final_cost_basis = {}
    for l in lots:
        if l.shares > 1e-6:
             final_cost_basis[l.ticker] = final_cost_basis.get(l.ticker, 0.0) + (l.shares * l.cost_price_eur)
    result.final_cost_basis = final_cost_basis
    
    return result

# ---------------------------------------------------------------------------
# Benchmark & TWR (Unchanged mostly)
# ---------------------------------------------------------------------------
def simulate_benchmark(cash_flows: pd.Series, benchmark_prices: pd.Series, buy_and_hold: bool) -> pd.DataFrame:
    bm_shares = 0.0
    bm_cash = 0.0
    rows = []
    for day in benchmark_prices.index:
        px = benchmark_prices.at[day]
        if pd.isna(px) or px <= 0: continue
        equity = bm_shares * px
        idle_cash = bm_cash if not buy_and_hold else 0.0
        val_before = equity + idle_cash
        
        flow = cash_flows.get(day, 0.0)
        if buy_and_hold and flow > 0:
            bm_shares += flow / px
        elif not buy_and_hold:
            if flow > 0: bm_shares += flow / px
            elif flow < 0:
                bm_shares += flow / px
                bm_cash -= flow
                
        val_after = bm_shares * px + (bm_cash if not buy_and_hold else 0.0)
        rows.append({"date": day, "value_before_tx": val_before, "benchmark_value": val_after})
    return pd.DataFrame(rows).set_index("date")

def compute_twr(value_after: pd.Series, value_before: pd.Series) -> pd.Series:
    twr = pd.Series(np.nan, index=value_after.index, dtype=float)
    twr_val = None
    prev_after = None
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
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(twr: pd.Series, label: str, prices=None, benchmark_twr=None) -> dict:
    clean = twr.dropna()
    if clean.empty: return {}
    days_invested = (clean.index[-1] - clean.index[0]).days
    total_return = (clean.iloc[-1] / clean.iloc[0] - 1) * 100
    daily_ret = clean.pct_change().dropna()
    vol = daily_ret.std() * np.sqrt(252) * 100
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else np.nan
    cummax = clean.cummax()
    max_dd = ((clean - cummax) / cummax).min() * 100
    
    # CAGR
    cagr = ((clean.iloc[-1] / clean.iloc[0]) ** (365.0/max(days_invested, 1)) - 1) * 100 if days_invested > 0 else np.nan
    # Sortino (downside std)
    downside_std = daily_ret[daily_ret < 0].std() * np.sqrt(252) * 100
    sortino = (daily_ret.mean() * 252 * 100) / downside_std if downside_std > 0 else np.nan
    # Calmar
    calmar = (cagr / abs(max_dd)) if max_dd < 0 else np.nan
    
    corr = np.nan
    if benchmark_twr is not None:
        c_i = daily_ret.index.intersection(benchmark_twr.pct_change().dropna().index)
        corr = daily_ret.loc[c_i].corr(benchmark_twr.pct_change().dropna().loc[c_i])
        
    return {
        "label": label,
        "total_return_pct": round(total_return, 2),
        "annualised_vol_pct": round(vol, 2),
        "sharpe_ratio": round(sharpe, 3) if not np.isnan(sharpe) else "n/a",
        "max_drawdown_pct": round(max_dd, 2),
        "cagr_pct": round(cagr, 2),
        "sortino_ratio": round(sortino, 3) if not np.isnan(sortino) else "n/a",
        "calmar_ratio": round(calmar, 3) if not np.isnan(calmar) else "n/a",
        "benchmark_correlation": round(corr, 3) if not np.isnan(corr) else "n/a",
        "start_date": clean.index[0].date(),
        "end_date": clean.index[-1].date(),
        "trading_days": len(clean),
    }

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    
    if args.clear_cache:
        cache_dir = Path(os.path.expanduser(str(args.cache_dir)))
        if cache_dir.exists() and cache_dir.is_dir():
            shutil.rmtree(cache_dir)
            print(f"[INFO] Cleared cache directory: {cache_dir}")

    transactions = load_transactions(args.transaction_file, args.manual_transactions, args.verbose)
    
    if args.lookup_file is not None:
        lookup_df = load_lookup(args.lookup_file)
        isin_to_ticker = dict(zip(lookup_df["ISIN"], lookup_df["YahooTicker"]))
    else:
        lookup_df = pd.DataFrame(columns=["ISIN", "YahooTicker", "Currency", "asset_class"])
        isin_to_ticker = {}
        if (transactions["ISIN"] != "").any():
            sys.exit("[ERROR] ISINs present but no lookup file provided.")
            
    resolved = resolve_tickers(transactions, isin_to_ticker)
    unresolved = resolved.isna()
    if unresolved.any():
        print("[WARN] Some tickers could not be resolved.")
        
    if args.verbose:
        print("\n[INFO] Resolved Ticker Mappings:")
        for isin, tick in isin_to_ticker.items():
            print(f"  {isin} -> {tick}")

    portfolio_tickers = sorted(set(resolved.dropna().tolist()))
    all_tickers = sorted(set(portfolio_tickers + [args.benchmark]))
    
    start = args.start_date or transactions["date"].min().strftime("%Y-%m-%d")
    end = args.end_date or pd.Timestamp.today().strftime("%Y-%m-%d")
    
    prices = fetch_prices(all_tickers, start=start, end=end, args=args)
    if args.benchmark not in prices.columns:
        sys.exit(f"[ERROR] Benchmark {args.benchmark} missing.")
        
    # Build currencies subset
    currencies = set()
    for t in portfolio_tickers:
        row_look = lookup_df[lookup_df["YahooTicker"] == t]
        if not row_look.empty and pd.notna(row_look.iloc[0]["Currency"]):
            currencies.add(row_look.iloc[0]["Currency"])
        tx_look = transactions[transactions["ticker"] == t]
        if not tx_look.empty and pd.notna(tx_look.iloc[0]["currency"]):
            currencies.add(tx_look.iloc[0]["currency"])
            
    fx_rates = get_fx_rates(currencies, start, end, args.base_currency, args)
    
    if args.verbose and not fx_rates.empty:
        print(f"\n[INFO] FX Rates retrieved: {list(fx_rates.columns)}")

    portfolio_df = reconstruct_portfolio(transactions, isin_to_ticker, lookup_df, prices, fx_rates, args)
    cash_flows = portfolio_df.cash_flows
    
    benchmark_df = simulate_benchmark(cash_flows, prices[args.benchmark], args.buy_and_hold)
    
    pv_twr = compute_twr(portfolio_df["portfolio_value"], portfolio_df["value_before_tx"])
    bv_twr = compute_twr(benchmark_df["benchmark_value"], benchmark_df["value_before_tx"])
    
    common = pv_twr.dropna().index.intersection(bv_twr.dropna().index)
    pv_twr = pv_twr.loc[common] / pv_twr.loc[common].iloc[0]
    bv_twr = bv_twr.loc[common] / bv_twr.loc[common].iloc[0]
    
    if args.export_csv:
        export_df = pd.DataFrame({
            "TWR": pv_twr,
            "portfolio_value_eur": portfolio_df["portfolio_value"],
            "cumulative_invested": portfolio_df["cumulative_invested"],
            "cumulative_gain": portfolio_df["portfolio_value"] - portfolio_df["cumulative_invested"]
        }).fillna(0)
        export_df.to_csv(args.export_csv)
        print(f"\n[INFO] Exported timelines to {args.export_csv}")

    # Display diagnostics only if verbose
    if args.verbose:
        diag = pd.DataFrame({
            "pf_value_before": portfolio_df["value_before_tx"],
            "pf_value_after": portfolio_df["portfolio_value"],
            "pf_twr": pv_twr
        })
        print(diag.tail(20))
        
    # Asset Attribution / Holdings
    final_positions = portfolio_df.final_positions
    final_val = 0.0
    holdings = []
    last_day = prices.index[-1]
    
    for t, sh in final_positions.items():
        px = prices.at[last_day, t] if last_day in prices.index and t in prices.columns else 0.0
        row_look = lookup_df[lookup_df["YahooTicker"] == t]
        cur = row_look.iloc[0]["Currency"] if not row_look.empty else ""
        cur = cur or args.base_currency
        
        rate = 1.0
        pair = f"{args.base_currency}{cur.upper()}=X"
        if pair in fx_rates.columns and not pd.isna(fx_rates.at[last_day, pair]):
            rate = float(fx_rates.at[last_day, pair])
            
        px_eur = px / rate if rate > 0 else px
        val_eur = sh * px_eur
        final_val += val_eur
        ac = get_asset_class(t, lookup_df)
        holdings.append({
            "ticker": t, "shares": sh, "price_native": px, "value_eur": val_eur, "asset_class": ac
        })
        
    print("\n=== HOLDINGS ===")
    asset_alloc = {}
    for h in holdings:
        w = (h['value_eur']/final_val*100) if final_val > 0 else 0
        cb = portfolio_df.final_cost_basis.get(h['ticker'], 0.0)
        c_inv_total = portfolio_df["cumulative_invested"].iloc[-1]
        contrib = ((h['value_eur'] - cb) / c_inv_total * 100) if c_inv_total > 0 else 0.0
        print(f"{h['ticker']:<15} | Sh: {h['shares']:<10.4f} | Px: {h['price_native']:<10.2f} | Val({args.base_currency}): {h['value_eur']:<12.2f} | Weight: {w:5.1f}% | Contrib: {contrib:5.2f}%")
        asset_alloc[h['asset_class']] = asset_alloc.get(h['asset_class'], 0.0) + w
        
    print("\n=== ALLOCATION SUMMARY ===")
    for ac, w in sorted(asset_alloc.items(), key=lambda x: x[1], reverse=True):
        print(f"{ac:<15} | {w:.1f}%")
        
    if args.holdings: # if holding only, skip plotting
        return
        
    c_inv = portfolio_df["cumulative_invested"].iloc[-1]
    tot_val = portfolio_df["portfolio_value"].iloc[-1]
    print(f"\nFinal Portfolio Value ({args.base_currency}): {tot_val:.2f}")
    print(f"Total Capital Invested     : {c_inv:.2f}")
    print(f"Total Gain/Loss            : {tot_val - c_inv:.2f} ({((tot_val/c_inv-1)*100) if c_inv>0 else 0:.2f}%)")
    
    if args.tax and not portfolio_df.tax_timeline.empty:
        ttf = portfolio_df.tax_timeline.copy()
        ttf['year'] = ttf['date'].dt.year
        summ = ttf.groupby(['year', 'event_type'])['tax'].sum().reset_index()
        print("\n=== TAX SUMMARY ===")
        print(summ)
        print(f"Total KESt Paid: {portfolio_df.tax_timeline['tax'].sum():.2f}")
    
    pm = compute_metrics(pv_twr, "Portfolio", prices, bv_twr)
    bm = compute_metrics(bv_twr, f"Benchmark ({args.benchmark})", prices)
    print("\n=== PERFORMANCE ===")
    print(pm)
    print(bm)
    
    if not args.no_plot:
        roll_1y = pv_twr / pv_twr.shift(252) - 1
        
        fig, axes = plt.subplots(4, 1, figsize=(12, 16), gridspec_kw={'height_ratios': [2, 1, 1, 1]})
        ax1, ax2, ax3, ax4 = axes
        
        ax1.plot(pv_twr.index, pv_twr.values, label="Portfolio TWR")
        ax1.plot(bv_twr.index, bv_twr.values, label=f"Benchmark ({args.benchmark})", linestyle="--")
        ax1.set_title("Time-Weighted Return")
        ax1.legend()
        
        ax2.fill_between(portfolio_df.index, 0, portfolio_df["cumulative_invested"], label="Capital Invested", alpha=0.5)
        ax2.fill_between(portfolio_df.index, portfolio_df["cumulative_invested"], portfolio_df["portfolio_value"], label="Appreciation", alpha=0.3)
        ax2.set_title("Capital vs Appreciation (EUR)")
        ax2.legend()
        
        alloc_df = portfolio_df.allocation_history
        if not alloc_df.empty:
            ax3.stackplot(alloc_df.index, *[alloc_df[col] for col in alloc_df.columns], labels=alloc_df.columns)
            ax3.set_title("Allocation Drift")
            ax3.legend(loc='upper left')
            
        ax4.plot(roll_1y.index, roll_1y.values * 100, color="purple", label="Rolling 1-Year TWR (%)")
        ax4.set_title("Rolling 1-Year Return")
        ax4.axhline(0, color="black", linestyle="-", linewidth=0.5)
        ax4.legend()
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()

