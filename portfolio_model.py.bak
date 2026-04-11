"""
portfolio_model.py

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

Usage:
    python portfolio_model.py transaction.csv isin_to_yahoo.csv \
        --benchmark VWCE.DE \
        --buy_and_hold true

    # Crypto-only portfolio (no lookup file required)
    python portfolio_model.py crypto_transactions.csv --benchmark BTC-USD
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

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
    parser.add_argument("transaction_file", type=Path, help="Path to transaction.csv")
    parser.add_argument(
        "lookup_file",
        type=Path,
        nargs="?",
        default=None,
        help=(
            "Path to isin_to_yahoo.csv (optional). Only required when "
            "transactions identify assets by ISIN. Portfolios that use "
            "direct Yahoo tickers (crypto, commodities, etc.) for every "
            "row can omit this argument."
        ),
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
        help="Optional path to export the TWR time series as CSV",
    )
    return parser.parse_args(argv)


# ---------------------------------------------------------------------------
# 2. Data loading & validation
# ---------------------------------------------------------------------------

REQUIRED_TX_COLS = {"date", "shares"}
REQUIRED_LOOKUP_COLS = {"ISIN", "YahooTicker"}


def load_transactions(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = REQUIRED_TX_COLS - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] transaction file missing columns: {missing}")

    # Each transaction must identify the asset via ISIN (looked up) or
    # ticker (used directly). At least one of the two columns must exist.
    if "ISIN" not in df.columns and "ticker" not in df.columns:
        sys.exit(
            "[ERROR] transaction file must contain either an 'ISIN' column "
            "(resolved via the lookup file) or a 'ticker' column (used as a "
            "Yahoo Finance symbol directly for assets without an ISIN, e.g. "
            "BTC-USD, GC=F)."
        )

    # Normalise identifier columns to stripped strings; missing → "".
    for col in ("ISIN", "ticker"):
        if col not in df.columns:
            df[col] = ""
        df[col] = df[col].fillna("").astype(str).str.strip()

    empty_ids = (df["ISIN"] == "") & (df["ticker"] == "")
    if empty_ids.any():
        bad = (df.index[empty_ids] + 2).tolist()  # +2 for header row + 0-based
        sys.exit(
            f"[ERROR] transaction rows {bad} have neither 'ISIN' nor 'ticker' "
            "set. Every row must provide one of them."
        )

    # Accept both DD.MM.YYYY (European) and YYYY-MM-DD (ISO) formats.
    # Try ISO first then fall back to European DD.MM.YYYY. This is explicit
    # per-row so ambiguous ISO strings like "2022-01-03" are not misread as
    # 1 March 2022 under dayfirst=True.
    raw_dates = df["date"].astype(str).str.strip()
    iso = pd.to_datetime(raw_dates, format="%Y-%m-%d", errors="coerce")
    eu = pd.to_datetime(raw_dates, format="%d.%m.%Y", errors="coerce")
    df["date"] = iso.fillna(eu)
    if df["date"].isna().any():
        bad = (df.index[df["date"].isna()] + 2).tolist()
        sys.exit(
            f"[ERROR] Unparseable dates in rows {bad}. "
            "Use YYYY-MM-DD (ISO) or DD.MM.YYYY (European)."
        )
    df["shares"] = pd.to_numeric(df["shares"], errors="coerce")

    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
    else:
        df["price"] = np.nan

    if "gross_amount" in df.columns:
        df["gross_amount"] = pd.to_numeric(df["gross_amount"], errors="coerce")
    else:
        df["gross_amount"] = np.nan

    df = df.sort_values("date").reset_index(drop=True)
    return df


def load_lookup(path: Path) -> dict[str, str]:
    df = pd.read_csv(path)
    missing = REQUIRED_LOOKUP_COLS - set(df.columns)
    if missing:
        sys.exit(f"[ERROR] lookup file missing columns: {missing}")
    return dict(zip(df["ISIN"], df["YahooTicker"]))


def resolve_tickers(
    transactions: pd.DataFrame,
    isin_to_ticker: dict[str, str],
) -> pd.Series:
    """
    Return a Series (aligned to ``transactions.index``) giving the Yahoo
    Finance symbol each transaction should use, or ``None`` if it cannot
    be resolved.

    Precedence per row:
      1. Non-empty ``ticker`` column value → used verbatim. This is how
         non-ISIN assets (gold, crypto, FX, futures…) enter the system.
      2. Non-empty ``ISIN`` column value  → looked up in ``isin_to_ticker``.
      3. Otherwise → ``None`` (caller decides whether to warn or error).
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


# ---------------------------------------------------------------------------
# 3. Price retrieval
# ---------------------------------------------------------------------------

# Suffixes of Yahoo Finance crypto pair symbols. Cryptocurrencies regularly
# move more than 15 % in a single day (e.g. BTC-USD on 12-Mar-2020 fell ~40 %),
# so the standard equity-focused spike filter must be loosened — otherwise
# legitimate price action gets wiped out by interpolation.
_CRYPTO_SUFFIXES = ("-USD", "-EUR", "-GBP", "-USDT", "-BTC", "-ETH", "-JPY")


def _is_crypto_ticker(ticker: str) -> bool:
    """Heuristic detector for Yahoo Finance cryptocurrency pair symbols."""
    upper = ticker.upper()
    return any(upper.endswith(suf) for suf in _CRYPTO_SUFFIXES)


def fetch_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    """Download adjusted-close prices for all tickers; returns a wide DataFrame."""
    print(f"[INFO] Downloading prices for {len(tickers)} ticker(s) from {start} to {end} …")
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

    # yfinance returns MultiIndex columns when >1 ticker; single ticker → flat
    if isinstance(raw.columns, pd.MultiIndex):
        prices = raw["Close"]
    else:
        prices = raw[["Close"]]
        prices.columns = tickers

    # Warn and drop tickers with >85 % missing values.
    # A higher threshold (vs the naive 50 %) is intentional: some holdings are
    # legitimately closed or suspended mid-period (e.g. Russia ETF suspended
    # 2022, Gold Bugs fund merged 2023) but still need prices for the earlier
    # dates when the position was actually held.  After ffill, those tickers
    # will carry their last known price forward, which is harmless because the
    # portfolio reconstruction contributes 0 from a position once it is sold.
    frac_missing = prices.isna().mean()
    bad = frac_missing[frac_missing > 0.85].index.tolist()
    if bad:
        warnings.warn(f"[WARN] Dropping tickers with insufficient data: {bad}", stacklevel=2)
        prices = prices.drop(columns=bad)

    prices = prices.ffill()

    # ── Spike detection ──────────────────────────────────────────────────────
    # A single-day move beyond the threshold is treated as a data error
    # (bad adjusted-close from yfinance after a dividend/split restatement).
    # The bad day is replaced with a linearly interpolated value so that the
    # surrounding series is undisturbed. A warning names the ticker and date.
    #
    # Equities / ETFs use 15 %. Cryptocurrencies use 50 % because 15 %-20 %
    # intraday moves are normal in that asset class and would otherwise be
    # "corrected" into a flat line. Crypto also has no dividends or splits,
    # so the original motivation for the filter does not really apply.
    #
    # Two passes: pass 1 removes the spike; pass 2 removes the "echo" return
    # that appears on the day AFTER the spike (which now looks large because
    # the previous day's value changed).
    SPIKE_THRESHOLD_DEFAULT = 0.15
    SPIKE_THRESHOLD_CRYPTO = 0.50
    for _ in range(2):
        chg = prices.pct_change().abs()
        for col in prices.columns:
            threshold = (
                SPIKE_THRESHOLD_CRYPTO if _is_crypto_ticker(col)
                else SPIKE_THRESHOLD_DEFAULT
            )
            bad_days = prices.index[chg[col] > threshold].tolist()
            if bad_days:
                warnings.warn(
                    f"[WARN] {col}: implausible single-day price move on "
                    f"{[d.date() for d in bad_days]}. "
                    f"Replacing with interpolated values.",
                    stacklevel=2,
                )
                prices.loc[bad_days, col] = np.nan
        # Re-interpolate; ffill/bfill handle edges (e.g. spike on last day)
        prices = prices.interpolate(method="time").ffill().bfill()

    # Drop the final bar: yfinance may return a partial intraday value for
    # the current session (markets still open).
    if len(prices) > 1:
        prices = prices.iloc[:-1]

    return prices


# ---------------------------------------------------------------------------
# 4. Portfolio reconstruction
# ---------------------------------------------------------------------------

def _snap_to_trading_day(dates: pd.Series, trading_days: pd.DatetimeIndex) -> pd.Series:
    """
    Map each date forward to the nearest following trading day.
    Transactions on weekends or holidays are booked on the next market open.
    If a date is already a trading day it is unchanged.
    Dates after the last trading day are dropped (returns NaT).
    """
    def snap(d: pd.Timestamp) -> pd.Timestamp | None:
        idx = trading_days.searchsorted(d)          # first position >= d
        if idx >= len(trading_days):
            return None
        return trading_days[idx]
    return dates.map(snap)


def _mark_to_market(positions: dict[str, float], prices: pd.DataFrame, day: pd.Timestamp) -> float:
    return sum(
        positions.get(t, 0.0) * float(prices.at[day, t])
        for t in positions
        if t in prices.columns and not pd.isna(prices.at[day, t])
    )


def reconstruct_portfolio(
    transactions: pd.DataFrame,
    isin_to_ticker: dict[str, str],
    prices: pd.DataFrame,
) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by trading date with columns:
        value_before_tx  — portfolio value using today's prices, BEFORE today's transactions
        portfolio_value  — portfolio value using today's prices, AFTER today's transactions
        cash

    .cash_flows attribute: pd.Series of net capital deployed per day (for benchmark use).
    """
    transactions = transactions.copy()
    # ``_yf_ticker`` is the resolved Yahoo Finance symbol for each row. We use
    # an internal name (rather than overwriting a user-provided ``ticker``
    # column) so rows using direct tickers for non-ISIN assets round-trip
    # cleanly.
    transactions["_yf_ticker"] = resolve_tickers(transactions, isin_to_ticker)
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

    # Snap transaction dates to the next available trading day.
    # Transactions on weekends/holidays are booked at the next market open.
    original_dates = transactions["date"].copy()
    transactions["date"] = _snap_to_trading_day(transactions["date"], prices.index)
    snapped = transactions["date"] != original_dates
    for _, row in transactions[snapped].iterrows():
        orig = original_dates.at[row.name]
        warnings.warn(
            f"[INFO] Transaction on non-trading day {orig.date()} booked on {row['date'].date()}.",
            stacklevel=2,
        )
    # Drop transactions that fall after the last available price date
    out_of_range = transactions["date"].isna()
    if out_of_range.any():
        warnings.warn(
            f"[WARN] {out_of_range.sum()} transaction(s) fall after the last available price date and are skipped.",
            stacklevel=2,
        )
    transactions = transactions.dropna(subset=["date"])

    tx_by_date: dict[pd.Timestamp, list] = {}
    for _, row in transactions.iterrows():
        tx_by_date.setdefault(row["date"], []).append(row)

    positions: dict[str, float] = {}
    cash = 0.0
    cash_flows: dict[pd.Timestamp, float] = {}
    rows = []

    for day in prices.index:
        # ── Value BEFORE today's transactions (today's prices, yesterday's positions) ──
        value_before_tx = cash + _mark_to_market(positions, prices, day)

        # ── Process today's transactions ──
        day_invested = 0.0
        if day in tx_by_date:
            for tx in tx_by_date[day]:
                ticker = tx["_yf_ticker"]
                shares = float(tx["shares"])

                # Resolve execution price
                if pd.notna(tx["price"]):
                    px = float(tx["price"])
                elif pd.notna(tx["gross_amount"]) and shares != 0:
                    px = abs(float(tx["gross_amount"])) / abs(shares)
                elif ticker in prices.columns and not pd.isna(prices.at[day, ticker]):
                    px = float(prices.at[day, ticker])
                else:
                    warnings.warn(
                        f"[WARN] Cannot determine price for {ticker} on {day.date()}; skipping tx.",
                        stacklevel=2,
                    )
                    continue

                cost = shares * px  # positive = buy outlay, negative = sell proceeds

                if shares > 0:
                    # Buy: external capital injection — cash injected then immediately spent.
                    # Net cash change = 0; equity grows by the invested amount.
                    cash += cost
                cash -= cost        # spend on buy (nets 0) or receive on sell (increases cash)

                positions[ticker] = positions.get(ticker, 0.0) + shares
                day_invested += cost

        if day_invested != 0:
            cash_flows[day] = cash_flows.get(day, 0.0) + day_invested

        # ── Value AFTER today's transactions ──
        value_after_tx = cash + _mark_to_market(positions, prices, day)

        rows.append({
            "date": day,
            "value_before_tx": value_before_tx,
            "portfolio_value": value_after_tx,
            "cash": cash,
        })

    result = pd.DataFrame(rows).set_index("date")
    result.cash_flows = pd.Series(cash_flows, name="cash_flow").sort_index()  # type: ignore[attr-defined]
    return result


# ---------------------------------------------------------------------------
# 5. Benchmark simulation
# ---------------------------------------------------------------------------

def simulate_benchmark(
    cash_flows: pd.Series,
    benchmark_prices: pd.Series,
    buy_and_hold: bool,
) -> pd.DataFrame:
    """
    Returns a DataFrame with columns:
        value_before_tx  — benchmark value BEFORE applying today's cash flow
        benchmark_value  — benchmark value AFTER applying today's cash flow

    buy_and_hold=True  → inflows buy benchmark shares; outflows are ignored.
    buy_and_hold=False → inflows buy shares; outflows sell shares and park cash.
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

        # Value BEFORE applying today's flow
        value_before = equity + idle_cash

        flow = cash_flows.get(day, 0.0)  # positive = new capital invested, negative = withdrawn

        if buy_and_hold:
            # Only buy on inflows; ignore outflows (capital stays invested)
            if flow > 0:
                bm_shares += flow / px
        else:
            # Mirror portfolio flows exactly
            if flow > 0:
                bm_shares += flow / px       # buy: externally funded, no cash accumulation
            elif flow < 0:
                bm_shares += flow / px       # sell: reduce shares (flow/px is negative)
                bm_cash -= flow              # park proceeds (-flow is positive)

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
# 6. Time-Weighted Return (TWR)
# ---------------------------------------------------------------------------

def compute_twr(value_after: pd.Series, value_before: pd.Series) -> pd.Series:
    """
    Compute a daily Time-Weighted Return index starting at 1.0 on the first
    invested day.

    For each day t:
        sub-period factor = value_before[t] / value_after[t-1]

    This measures only price-driven returns, stripping out the mechanical
    effect of capital injections and withdrawals.

    Parameters
    ----------
    value_after  : portfolio/benchmark value AFTER processing cash flows
    value_before : portfolio/benchmark value BEFORE processing cash flows
                   (same prices, but positions from previous day)
    """
    twr = pd.Series(np.nan, index=value_after.index, dtype=float)
    twr_val: float | None = None
    prev_after: float | None = None

    for day in value_after.index:
        v_before = float(value_before.at[day])
        v_after = float(value_after.at[day])

        if twr_val is None:
            # Start counting from the first day the portfolio has any value
            if v_after > 0:
                twr_val = 1.0
                prev_after = v_after
                twr.at[day] = twr_val
        else:
            # Chain-link: today's pre-flow value vs yesterday's post-flow value
            if prev_after is not None and prev_after > 0 and not np.isnan(v_before):
                twr_val *= v_before / prev_after
            twr.at[day] = twr_val
            prev_after = v_after if v_after > 0 else prev_after

    return twr


# ---------------------------------------------------------------------------
# 7. Metrics
# ---------------------------------------------------------------------------

def compute_metrics(twr: pd.Series, label: str) -> dict:
    clean = twr.dropna()
    if clean.empty:
        return {}
    total_return = (clean.iloc[-1] / clean.iloc[0] - 1) * 100
    daily_ret = clean.pct_change().dropna()
    vol = daily_ret.std() * np.sqrt(252) * 100
    sharpe = (daily_ret.mean() / daily_ret.std() * np.sqrt(252)) if daily_ret.std() > 0 else np.nan
    cummax = clean.cummax()
    max_dd = ((clean - cummax) / cummax).min() * 100
    return {
        "label": label,
        "total_return_pct": round(total_return, 2),
        "annualised_vol_pct": round(vol, 2),
        "sharpe_ratio": round(sharpe, 3) if not np.isnan(sharpe) else "n/a",
        "max_drawdown_pct": round(max_dd, 2),
        "start_date": clean.index[0].date(),
        "end_date": clean.index[-1].date(),
        "trading_days": len(clean),
    }


# ---------------------------------------------------------------------------
# 8. Visualisation
# ---------------------------------------------------------------------------

def plot_performance(
    portfolio_twr: pd.Series,
    benchmark_twr: pd.Series,
    benchmark_ticker: str,
    buy_and_hold: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(portfolio_twr.index, portfolio_twr.values, label="Portfolio", linewidth=2)

    bm_label = f"Benchmark ({benchmark_ticker})"
    bm_label += " [buy-and-hold]" if buy_and_hold else " [mirror flows]"
    ax.plot(benchmark_twr.index, benchmark_twr.values, label=bm_label, linewidth=2, linestyle="--")

    ax.axhline(1.0, color="grey", linewidth=0.8, linestyle=":")
    ax.set_title("Portfolio vs Benchmark — Time-Weighted Return (TWR)", fontsize=14)
    ax.set_xlabel("Date")
    ax.set_ylabel("TWR Index (start = 1.0)")
    ax.legend()
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax.xaxis.set_major_locator(mdates.AutoDateLocator())
    fig.autofmt_xdate()
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 9. Main
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)

    # --- Load inputs --------------------------------------------------------
    transactions = load_transactions(args.transaction_file)

    # Lookup file is optional: a portfolio that identifies every holding
    # with a direct ``ticker`` (e.g. crypto-only) does not need one.
    if args.lookup_file is not None:
        isin_to_ticker = load_lookup(args.lookup_file)
    else:
        isin_to_ticker = {}
        if (transactions["ISIN"] != "").any():
            sys.exit(
                "[ERROR] transactions use ISINs but no lookup file was "
                "provided. Pass the lookup file as the second positional "
                "argument, or replace the ISINs with direct 'ticker' values."
            )

    # Resolve every transaction to a Yahoo ticker, either via the ISIN
    # lookup or via the direct ``ticker`` column (used for assets without
    # an ISIN such as gold, silver and cryptocurrencies).
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
            "Yahoo Finance symbol in the 'ticker' column of the transaction "
            "file (e.g. BTC-USD for Bitcoin, GC=F for gold futures)."
        )

    portfolio_tickers = sorted(set(resolved.tolist()))
    all_tickers = sorted(set(portfolio_tickers + [args.benchmark]))

    # --- Date range ---------------------------------------------------------
    start = args.start_date or transactions["date"].min().strftime("%Y-%m-%d")
    end = args.end_date or pd.Timestamp.today().strftime("%Y-%m-%d")

    # --- Price download -----------------------------------------------------
    prices = fetch_prices(all_tickers, start=start, end=end)

    if args.benchmark not in prices.columns:
        sys.exit(f"[ERROR] Benchmark ticker '{args.benchmark}' not found in downloaded data.")

    missing_tickers = [t for t in portfolio_tickers if t not in prices.columns]
    if missing_tickers:
        sys.exit(f"[ERROR] Missing price data for portfolio ticker(s): {missing_tickers}")

    # --- Portfolio reconstruction -------------------------------------------
    portfolio_df = reconstruct_portfolio(transactions, isin_to_ticker, prices)
    cash_flows: pd.Series = portfolio_df.cash_flows  # type: ignore[attr-defined]

    # --- Benchmark simulation -----------------------------------------------
    benchmark_df = simulate_benchmark(cash_flows, prices[args.benchmark], args.buy_and_hold)

    # --- Compute TWR for both -----------------------------------------------
    pv_twr = compute_twr(portfolio_df["portfolio_value"], portfolio_df["value_before_tx"])
    bv_twr = compute_twr(benchmark_df["benchmark_value"], benchmark_df["value_before_tx"])

    # --- Align on common valid dates ----------------------------------------
    common_idx = pv_twr.dropna().index.intersection(bv_twr.dropna().index)
    if common_idx.empty:
        sys.exit("[ERROR] No overlapping invested days between portfolio and benchmark.")

    pv_twr = pv_twr.loc[common_idx]
    bv_twr = bv_twr.loc[common_idx]

    # Both TWR series start at 1.0 on their first invested day by construction.
    # Re-anchor both to the first common day so they share the same baseline.
    pv_twr = pv_twr / pv_twr.iloc[0]
    bv_twr = bv_twr / bv_twr.iloc[0]

    # --- Diagnostics --------------------------------------------------------
    # Print the last 20 rows of every intermediate series so we can see
    # exactly what is driving the spike at the end of the chart.

    diag = pd.DataFrame({
        "pf_value_before":   portfolio_df["value_before_tx"],
        "pf_value_after":    portfolio_df["portfolio_value"],
        "pf_cash":           portfolio_df["cash"],
        "bm_value_before":   benchmark_df["value_before_tx"],
        "bm_value_after":    benchmark_df["benchmark_value"],
        "pf_twr":            pv_twr,
        "bm_twr":            bv_twr,
    })
    # Also attach per-ticker prices
    for col in prices.columns:
        diag[f"price_{col}"] = prices[col]

    # Compute daily pct-change on each price to surface any remaining spikes
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

    # Also flag any day in the TWR series where pf_twr moves more than 3 %
    pf_daily = pv_twr.pct_change().abs()
    bm_daily = bv_twr.pct_change().abs()
    large_moves = pf_daily[pf_daily > 0.03].index.union(bm_daily[bm_daily > 0.03].index)
    if not large_moves.empty:
        print("\n  Large TWR moves (>3 % in a single day):")
        for d in large_moves:
            row = diag.loc[d] if d in diag.index else {}
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

    # --- Metrics ------------------------------------------------------------
    pm = compute_metrics(pv_twr, "Portfolio")
    bm = compute_metrics(bv_twr, f"Benchmark ({args.benchmark})")

    print("\n" + "=" * 58)
    print("  PERFORMANCE SUMMARY  (Time-Weighted Return)")
    print("=" * 58)
    for metrics in (pm, bm):
        print(f"\n  {metrics['label']}")
        print(f"    Period           : {metrics['start_date']} → {metrics['end_date']}")
        print(f"    Trading days     : {metrics['trading_days']}")
        print(f"    Total return     : {metrics['total_return_pct']:>8.2f} %")
        print(f"    Annualised vol   : {metrics['annualised_vol_pct']:>8.2f} %")
        print(f"    Sharpe ratio     : {metrics['sharpe_ratio']}")
        print(f"    Max drawdown     : {metrics['max_drawdown_pct']:>8.2f} %")
    print("=" * 58 + "\n")

    # --- Optional CSV export ------------------------------------------------
    if args.export_csv:
        out = pd.DataFrame({"portfolio_twr": pv_twr, "benchmark_twr": bv_twr})
        out.to_csv(args.export_csv)
        print(f"[INFO] TWR time series exported to {args.export_csv}")

    # --- Chart --------------------------------------------------------------
    plot_performance(pv_twr, bv_twr, args.benchmark, args.buy_and_hold)


if __name__ == "__main__":
    main()
