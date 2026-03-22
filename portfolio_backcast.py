#!/usr/bin/env python3
"""
Reconstruct a portfolio dynamically from transaction data and compare it to a benchmark.

Inputs
------
1) transactions CSV with required columns:
       date, ISIN, shares

   optional columns:
       price
       gross_amount

2) lookup CSV with columns:
       ISIN, YahooTicker

Transaction interpretation
--------------------------
- shares > 0  : buy
- shares < 0  : sell

Cash-flow priority:
1. If gross_amount is present, use it directly.
2. Else if price is present, use gross_amount = -shares * price
3. Else infer from market close on execution day:
       gross_amount = -shares * market_price

Portfolio cash convention:
- negative gross_amount = cash leaves portfolio (buy)
- positive gross_amount = cash enters portfolio (sell)

Portfolio value
---------------
total_portfolio_value = security_positions_value + portfolio_cash

Benchmark modes
---------------
matched_flows:
    Benchmark receives matched investment flows and matched withdrawals.
    Benchmark shares may be sold.

buy_and_hold:
    Only positive external contributions are invested into benchmark.
    Benchmark shares are never sold.

Important assumptions
---------------------
- Transaction dates are mapped to the first trading day on or after the stated date.
- If price and gross_amount are absent, same-day close is used as execution approximation.
- No taxes, fees, spreads, or FX transaction costs are modeled unless embedded in gross_amount.
- Adjusted closes are used for valuation history.

Install
-------
conda install pandas matplotlib
pip install yfinance

Example
-------
python portfolio_from_transactions.py \
    transactions.csv \
    isin_to_yahoo.csv \
    --benchmark VWCE.DE \
    --benchmark-currency EUR \
    --base-currency EUR \
    --benchmark-mode matched_flows \
    --output-plot portfolio_vs_benchmark.png \
    --output-csv reconstructed_timeseries.csv \
    --show-components
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import pandas as pd
import yfinance as yf


# ---------------------------------------------------------------------
# FX mapping
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class FxMeta:
    ticker: str
    invert: bool


FX_MAP: Dict[Tuple[str, str], FxMeta] = {
    ("USD", "EUR"): FxMeta("EURUSD=X", True),
    ("GBP", "EUR"): FxMeta("EURGBP=X", True),
    ("CHF", "EUR"): FxMeta("EURCHF=X", True),
    ("JPY", "EUR"): FxMeta("EURJPY=X", True),

    ("EUR", "USD"): FxMeta("EURUSD=X", False),
    ("GBP", "USD"): FxMeta("GBPUSD=X", False),
    ("CHF", "USD"): FxMeta("USDCHF=X", True),
    ("JPY", "USD"): FxMeta("USDJPY=X", True),

    ("EUR", "GBP"): FxMeta("EURGBP=X", False),
    ("USD", "GBP"): FxMeta("GBPUSD=X", True),
    ("CHF", "GBP"): FxMeta("GBPCHF=X", True),
    ("JPY", "GBP"): FxMeta("GBPJPY=X", True),

    ("EUR", "CHF"): FxMeta("EURCHF=X", False),
    ("USD", "CHF"): FxMeta("USDCHF=X", False),
    ("GBP", "CHF"): FxMeta("GBPCHF=X", False),
    ("JPY", "CHF"): FxMeta("CHFJPY=X", False),

    ("EUR", "JPY"): FxMeta("EURJPY=X", False),
    ("USD", "JPY"): FxMeta("USDJPY=X", False),
    ("GBP", "JPY"): FxMeta("GBPJPY=X", False),
    ("CHF", "JPY"): FxMeta("CHFJPY=X", True),
}


# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct a portfolio from transactions and compare against a benchmark."
    )
    parser.add_argument(
        "transactions_csv",
        type=Path,
        help="CSV with required columns date, ISIN, shares and optional price, gross_amount",
    )
    parser.add_argument(
        "lookup_csv",
        type=Path,
        help="CSV with columns ISIN, YahooTicker",
    )
    parser.add_argument(
        "--benchmark",
        default="ACWI",
        help="Yahoo Finance ticker for benchmark ETF/index (default: ACWI)",
    )
    parser.add_argument(
        "--benchmark-currency",
        default="USD",
        help="Trading currency of benchmark ticker (default: USD)",
    )
    parser.add_argument(
        "--base-currency",
        default="EUR",
        help="Base currency for valuation and comparison (default: EUR)",
    )
    parser.add_argument(
        "--benchmark-mode",
        choices=["matched_flows", "buy_and_hold"],
        default="matched_flows",
        help="Benchmark logic: matched_flows or buy_and_hold (default: matched_flows)",
    )
    parser.add_argument(
        "--start-buffer-days",
        type=int,
        default=5,
        help="Extra days before first transaction for download alignment (default: 5)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional output CSV with reconstructed time series",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=None,
        help="Optional output plot path",
    )
    parser.add_argument(
        "--show-components",
        action="store_true",
        help="Print mapped holdings, currencies, execution details, and cash flows",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------
# Input loading
# ---------------------------------------------------------------------

def load_transactions(transactions_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(transactions_csv)

    required = {"date", "ISIN", "shares"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required transaction columns: {sorted(missing)}")

    allowed = {"date", "ISIN", "shares", "price", "gross_amount"}
    extra = set(df.columns) - allowed
    if extra:
        print(f"Warning: ignoring extra transaction columns: {sorted(extra)}", file=sys.stderr)

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="raise").dt.normalize()
    df["ISIN"] = df["ISIN"].astype(str).str.strip().str.upper()
    df["shares"] = pd.to_numeric(df["shares"], errors="raise")

    if "price" in df.columns:
        df["price"] = pd.to_numeric(df["price"], errors="coerce")
    else:
        df["price"] = pd.NA

    if "gross_amount" in df.columns:
        df["gross_amount"] = pd.to_numeric(df["gross_amount"], errors="coerce")
    else:
        df["gross_amount"] = pd.NA

    df = df.sort_values(["date", "ISIN"]).reset_index(drop=True)
    return df


def load_lookup(lookup_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(lookup_csv)

    required = {"ISIN", "YahooTicker"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required lookup columns: {sorted(missing)}")

    df = df.copy()
    df["ISIN"] = df["ISIN"].astype(str).str.strip().str.upper()
    df["YahooTicker"] = df["YahooTicker"].astype(str).str.strip()

    if df["ISIN"].duplicated().any():
        dup = df.loc[df["ISIN"].duplicated(), "ISIN"].tolist()
        raise ValueError(f"Duplicate ISINs found in lookup table: {dup}")

    return df


def merge_transactions_with_lookup(transactions: pd.DataFrame, lookup: pd.DataFrame) -> pd.DataFrame:
    merged = transactions.merge(lookup, on="ISIN", how="left")
    missing = merged.loc[merged["YahooTicker"].isna(), "ISIN"].unique().tolist()
    if missing:
        raise ValueError(f"No YahooTicker found in lookup table for ISIN(s): {missing}")
    return merged


# ---------------------------------------------------------------------
# Yahoo helpers
# ---------------------------------------------------------------------

def get_currency_from_ticker(ticker: str) -> Optional[str]:
    try:
        tk = yf.Ticker(ticker)

        fi = getattr(tk, "fast_info", None)
        if isinstance(fi, dict):
            cur = fi.get("currency")
            if isinstance(cur, str) and len(cur) == 3:
                return cur.upper()

        info = getattr(tk, "info", None)
        if isinstance(info, dict):
            for key in ("currency", "financialCurrency"):
                cur = info.get(key)
                if isinstance(cur, str) and len(cur) == 3:
                    return cur.upper()

        md = getattr(tk, "history_metadata", None)
        if isinstance(md, dict):
            for key in ("currency", "financialCurrency"):
                cur = md.get(key)
                if isinstance(cur, str) and len(cur) == 3:
                    return cur.upper()
    except Exception:
        pass

    return None


def attach_currencies(transaction_df: pd.DataFrame) -> pd.DataFrame:
    tickers = pd.Series(transaction_df["YahooTicker"].unique(), name="YahooTicker")
    ticker_to_currency = {}

    for ticker in tickers:
        cur = get_currency_from_ticker(ticker)
        if not cur:
            raise RuntimeError(f"Could not determine trading currency for ticker {ticker}")
        ticker_to_currency[ticker] = cur

    out = transaction_df.copy()
    out["currency"] = out["YahooTicker"].map(ticker_to_currency)
    return out


def download_adjusted_close(
    tickers: List[str],
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
) -> pd.DataFrame:
    tickers = list(dict.fromkeys(tickers))

    data = yf.download(
        tickers=tickers,
        start=start_date.strftime("%Y-%m-%d"),
        end=(end_date + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    if data.empty:
        raise RuntimeError("No data downloaded. Check ticker symbols or internet connection.")

    if len(tickers) == 1:
        if "Close" not in data.columns:
            raise RuntimeError(f"No Close column found for ticker {tickers[0]}.")
        return pd.DataFrame({tickers[0]: data["Close"]}).sort_index()

    close_frames = []
    level0 = set(data.columns.get_level_values(0))

    for ticker in tickers:
        if ticker not in level0:
            continue
        sub = data[ticker]
        if "Close" not in sub.columns:
            continue
        close_frames.append(sub["Close"].rename(ticker))

    if not close_frames:
        raise RuntimeError("Could not extract Close prices for any ticker.")

    return pd.concat(close_frames, axis=1).sort_index()


# ---------------------------------------------------------------------
# FX helpers
# ---------------------------------------------------------------------

def required_fx_pairs(currencies: List[str], base_currency: str) -> Dict[Tuple[str, str], FxMeta]:
    pairs = {}
    for cur in sorted(set(currencies)):
        if cur == base_currency:
            continue
        key = (cur, base_currency)
        if key not in FX_MAP:
            raise ValueError(f"No FX mapping available for converting {cur} -> {base_currency}")
        pairs[key] = FX_MAP[key]
    return pairs


def get_fx_series(from_currency: str, to_currency: str, prices: pd.DataFrame) -> pd.Series:
    if from_currency == to_currency:
        return pd.Series(1.0, index=prices.index, name=f"{from_currency}->{to_currency}")

    meta = FX_MAP[(from_currency, to_currency)]
    if meta.ticker not in prices.columns:
        raise RuntimeError(
            f"FX ticker {meta.ticker} for {from_currency}->{to_currency} not found in downloaded data."
        )

    fx = prices[meta.ticker].copy().sort_index().ffill()

    if meta.invert:
        fx = 1.0 / fx

    fx.name = f"{from_currency}->{to_currency}"
    return fx


# ---------------------------------------------------------------------
# Transaction execution alignment
# ---------------------------------------------------------------------

def trading_index_from_prices(prices: pd.DataFrame) -> pd.DatetimeIndex:
    idx = prices.index
    if not isinstance(idx, pd.DatetimeIndex):
        raise RuntimeError("Price table index must be a DatetimeIndex.")
    return idx


def map_to_execution_date(tx_date: pd.Timestamp, trading_days: pd.DatetimeIndex) -> pd.Timestamp:
    pos = trading_days.searchsorted(tx_date, side="left")
    if pos >= len(trading_days):
        raise RuntimeError(f"No trading day found on or after transaction date {tx_date.date()}")
    return trading_days[pos]


def attach_execution_dates(transactions: pd.DataFrame, trading_days: pd.DatetimeIndex) -> pd.DataFrame:
    out = transactions.copy()
    out["exec_date"] = out["date"].map(lambda d: map_to_execution_date(d, trading_days))
    return out


# ---------------------------------------------------------------------
# Cash flow logic
# ---------------------------------------------------------------------

def infer_transaction_gross_amount_base(
    row: pd.Series,
    prices: pd.DataFrame,
    base_currency: str,
) -> Tuple[float, str]:
    """
    Returns:
        gross_amount_base, source

    Convention:
        negative = cash leaves portfolio
        positive = cash enters portfolio
    """
    shares = float(row["shares"])
    exec_date = row["exec_date"]
    ticker = row["YahooTicker"]
    currency = row["currency"]

    fx_exec = float(get_fx_series(currency, base_currency, prices).loc[exec_date])

    if pd.notna(row.get("gross_amount", pd.NA)):
        gross_amount = float(row["gross_amount"])
        gross_amount_base = gross_amount * fx_exec
        return gross_amount_base, "gross_amount"

    if pd.notna(row.get("price", pd.NA)):
        price = float(row["price"])
        gross_amount = -shares * price
        gross_amount_base = gross_amount * fx_exec
        return gross_amount_base, "price"

    market_px = float(prices.loc[exec_date, ticker])
    gross_amount = -shares * market_px
    gross_amount_base = gross_amount * fx_exec
    return gross_amount_base, "market_price"


def attach_transaction_cashflows_base(
    transactions: pd.DataFrame,
    prices: pd.DataFrame,
    base_currency: str,
) -> pd.DataFrame:
    out = transactions.copy()
    gross_amounts_base = []
    sources = []

    for _, row in out.iterrows():
        amount, source = infer_transaction_gross_amount_base(row, prices, base_currency)
        gross_amounts_base.append(amount)
        sources.append(source)

    out["gross_amount_base"] = gross_amounts_base
    out["cashflow_source"] = sources
    return out


# ---------------------------------------------------------------------
# Portfolio reconstruction
# ---------------------------------------------------------------------

def build_daily_holdings(transactions: pd.DataFrame, daily_index: pd.DatetimeIndex) -> pd.DataFrame:
    daily_tx = (
        transactions.groupby(["exec_date", "YahooTicker"], as_index=False)["shares"]
        .sum()
        .pivot(index="exec_date", columns="YahooTicker", values="shares")
        .fillna(0.0)
    )

    daily_tx = daily_tx.reindex(daily_index, fill_value=0.0)
    holdings = daily_tx.cumsum()
    return holdings


def build_daily_cash_series(transactions: pd.DataFrame, daily_index: pd.DatetimeIndex) -> pd.Series:
    """
    Cash accounting:
        cash_t = cumulative sum of gross_amount_base
    because:
        buy  -> negative gross_amount_base -> cash decreases
        sell -> positive gross_amount_base -> cash increases
    """
    daily_cashflows = (
        transactions.groupby("exec_date", as_index=False)["gross_amount_base"]
        .sum()
        .set_index("exec_date")["gross_amount_base"]
    )

    daily_cashflows = daily_cashflows.reindex(daily_index, fill_value=0.0)
    cash_series = daily_cashflows.cumsum()
    cash_series.name = "portfolio_cash"
    return cash_series


def build_position_value_table(
    holdings: pd.DataFrame,
    ticker_currency: Dict[str, str],
    prices: pd.DataFrame,
    base_currency: str,
) -> pd.DataFrame:
    position_values = {}

    for ticker in holdings.columns:
        if ticker not in prices.columns:
            raise RuntimeError(f"Missing price series for {ticker}")

        cur = ticker_currency[ticker]
        px = prices[ticker].copy().sort_index()
        fx = get_fx_series(cur, base_currency, prices)

        position_values[ticker] = holdings[ticker] * px * fx

    return pd.DataFrame(position_values).sort_index()


def build_portfolio_value_series(position_values: pd.DataFrame, cash_series: pd.Series) -> pd.DataFrame:
    out = pd.DataFrame(index=position_values.index)
    out["portfolio_positions_value"] = position_values.sum(axis=1)
    out["portfolio_cash"] = cash_series.reindex(out.index).fillna(method="ffill").fillna(0.0)
    out["portfolio_value"] = out["portfolio_positions_value"] + out["portfolio_cash"]
    return out


# ---------------------------------------------------------------------
# Benchmark reconstruction
# ---------------------------------------------------------------------

def benchmark_contribution_from_tx(row: pd.Series, mode: str) -> float:
    """
    Convert transaction cash flow into benchmark contribution in base currency.

    gross_amount_base convention:
        buy  -> negative (cash leaves portfolio)
        sell -> positive (cash enters portfolio)

    matched_flows:
        benchmark receives exact opposite investment flow:
            contribution = -gross_amount_base

    buy_and_hold:
        invest only positive contributions into benchmark, never sell benchmark shares.
        Sells in the portfolio do not trigger benchmark sales.
        Therefore:
            contribution = max(-gross_amount_base, 0)
    """
    gross_amount_base = float(row["gross_amount_base"])
    invest_amount = -gross_amount_base

    if mode == "matched_flows":
        return invest_amount

    if mode == "buy_and_hold":
        return max(invest_amount, 0.0)

    raise ValueError(f"Unknown benchmark mode: {mode}")


def build_benchmark_state(
    transactions: pd.DataFrame,
    benchmark_ticker: str,
    benchmark_currency: str,
    prices: pd.DataFrame,
    base_currency: str,
    daily_index: pd.DatetimeIndex,
    mode: str,
) -> pd.DataFrame:
    if benchmark_ticker not in prices.columns:
        raise RuntimeError(f"Benchmark series {benchmark_ticker} was not downloaded.")

    benchmark_fx = get_fx_series(benchmark_currency, base_currency, prices)
    benchmark_price_in_base = prices[benchmark_ticker] * benchmark_fx

    rows = []
    for _, row in transactions.iterrows():
        exec_date = row["exec_date"]
        contribution_base = benchmark_contribution_from_tx(row, mode)

        bench_px_base = float(benchmark_price_in_base.loc[exec_date])
        if bench_px_base == 0:
            raise RuntimeError(f"Benchmark price is zero on {exec_date.date()}")

        benchmark_shares_change = contribution_base / bench_px_base

        # cash accounting on benchmark side:
        # contribution > 0 : invest cash into benchmark -> benchmark cash decreases
        # contribution < 0 : sell benchmark shares       -> benchmark cash increases
        benchmark_cash_change = -contribution_base

        rows.append(
            {
                "exec_date": exec_date,
                "benchmark_contribution_base": contribution_base,
                "benchmark_shares_change": benchmark_shares_change,
                "benchmark_cash_change": benchmark_cash_change,
            }
        )

    bench_tx = pd.DataFrame(rows)

    if bench_tx.empty:
        out = pd.DataFrame(index=daily_index)
        out["benchmark_contribution_base"] = 0.0
        out["benchmark_shares_change"] = 0.0
        out["benchmark_cash_change"] = 0.0
        out["benchmark_shares_held"] = 0.0
        out["benchmark_cash"] = 0.0
        out["benchmark_positions_value"] = 0.0
        out["benchmark_value"] = 0.0
        return out

    daily = (
        bench_tx.groupby("exec_date", as_index=False)[
            ["benchmark_contribution_base", "benchmark_shares_change", "benchmark_cash_change"]
        ]
        .sum()
        .set_index("exec_date")
        .reindex(daily_index, fill_value=0.0)
    )

    daily["benchmark_shares_held"] = daily["benchmark_shares_change"].cumsum()
    daily["benchmark_cash"] = daily["benchmark_cash_change"].cumsum()
    daily["benchmark_positions_value"] = daily["benchmark_shares_held"] * benchmark_price_in_base.reindex(daily_index)
    daily["benchmark_value"] = daily["benchmark_positions_value"] + daily["benchmark_cash"]

    return daily


# ---------------------------------------------------------------------
# Output helpers
# ---------------------------------------------------------------------

def normalize_to_100(series: pd.Series) -> pd.Series:
    series = series.dropna()
    first = series.iloc[0]
    return series / first * 100.0


def print_first_valid_dates(prices: pd.DataFrame) -> None:
    print("\nFirst valid date for each downloaded series:")
    for col in prices.columns:
        first = prices[col].first_valid_index()
        print(f"  {col:12s} {first}")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    try:
        transactions = load_transactions(args.transactions_csv)
        lookup = load_lookup(args.lookup_csv)
        transactions = merge_transactions_with_lookup(transactions, lookup)
        transactions = attach_currencies(transactions)

        unique_tickers = transactions["YahooTicker"].unique().tolist()
        unique_currencies = transactions["currency"].unique().tolist()

        start_date = transactions["date"].min() - pd.Timedelta(days=args.start_buffer_days)
        end_date = pd.Timestamp.today().normalize()

        fx_pairs = required_fx_pairs(
            currencies=list(unique_currencies) + [args.benchmark_currency],
            base_currency=args.base_currency,
        )
        fx_tickers = [meta.ticker for meta in fx_pairs.values()]

        download_tickers = unique_tickers + [args.benchmark] + fx_tickers
        prices = download_adjusted_close(
            tickers=download_tickers,
            start_date=start_date,
            end_date=end_date,
        )

        if args.show_components:
            print("\nMapped transactions:")
            print(
                transactions[["date", "ISIN", "YahooTicker", "shares", "price", "gross_amount", "currency"]]
                .to_string(index=False)
            )
            print_first_valid_dates(prices)

        trading_days = trading_index_from_prices(prices)
        transactions = attach_execution_dates(transactions, trading_days)
        transactions = attach_transaction_cashflows_base(
            transactions=transactions,
            prices=prices,
            base_currency=args.base_currency,
        )

        if args.show_components:
            print("\nTransactions with execution dates and base-currency cash flows:")
            print(
                transactions[
                    [
                        "date", "exec_date", "ISIN", "YahooTicker", "shares",
                        "price", "gross_amount", "gross_amount_base", "cashflow_source", "currency"
                    ]
                ].to_string(index=False)
            )

        daily_index = prices.index
        ticker_currency = (
            transactions[["YahooTicker", "currency"]]
            .drop_duplicates()
            .set_index("YahooTicker")["currency"]
            .to_dict()
        )

        holdings = build_daily_holdings(transactions, daily_index)
        position_values = build_position_value_table(
            holdings=holdings,
            ticker_currency=ticker_currency,
            prices=prices,
            base_currency=args.base_currency,
        )
        portfolio_cash = build_daily_cash_series(transactions, daily_index)
        portfolio_state = build_portfolio_value_series(position_values, portfolio_cash)

        benchmark_state = build_benchmark_state(
            transactions=transactions,
            benchmark_ticker=args.benchmark,
            benchmark_currency=args.benchmark_currency,
            prices=prices,
            base_currency=args.base_currency,
            daily_index=daily_index,
            mode=args.benchmark_mode,
        )

        combined = pd.concat(
            [
                portfolio_state,
                benchmark_state[
                    ["benchmark_positions_value", "benchmark_cash", "benchmark_value"]
                ],
            ],
            axis=1,
        ).dropna(how="any")

        if combined.empty:
            raise RuntimeError("No overlapping date range between portfolio and benchmark.")

        first_exec_date = transactions["exec_date"].min()
        combined = combined.loc[first_exec_date:].copy()

        if combined.empty:
            raise RuntimeError("No data remain after restricting to first execution date.")

        combined["portfolio_index"] = normalize_to_100(combined["portfolio_value"])
        combined["benchmark_index"] = normalize_to_100(combined["benchmark_value"])

        portfolio_return = combined["portfolio_index"].iloc[-1] / 100.0 - 1.0
        benchmark_return = combined["benchmark_index"].iloc[-1] / 100.0 - 1.0

        print("\nAnalysis summary")
        print(f"  Base currency      : {args.base_currency}")
        print(f"  Benchmark ticker   : {args.benchmark}")
        print(f"  Benchmark currency : {args.benchmark_currency}")
        print(f"  Benchmark mode     : {args.benchmark_mode}")
        print(f"  Start date         : {combined.index.min().date()}")
        print(f"  End date           : {combined.index.max().date()}")
        print(f"  Portfolio return   : {portfolio_return: .2%}")
        print(f"  Benchmark return   : {benchmark_return: .2%}")

        if args.output_csv is not None:
            combined.to_csv(args.output_csv, index_label="date")
            print(f"\nSaved time series to: {args.output_csv}")

        plt.figure(figsize=(11, 6))
        plt.plot(combined.index, combined["portfolio_index"], label="Portfolio")
        plt.plot(combined.index, combined["benchmark_index"], label=f"Benchmark ({args.benchmark})")
        plt.axhline(100, linewidth=1)

        plt.title("Portfolio vs benchmark")
        plt.xlabel("Date")
        plt.ylabel("Indexed performance (start = 100)")
        plt.legend()
        plt.tight_layout()

        if args.output_plot is not None:
            plt.savefig(args.output_plot, dpi=200, bbox_inches="tight")
            print(f"Saved plot to: {args.output_plot}")

        plt.show()

    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()