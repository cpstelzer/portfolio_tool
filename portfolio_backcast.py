#!/usr/bin/env python3
"""
Reconstruct a hypothetical portfolio development from a CSV file containing
Yahoo ticker symbols directly.

Required CSV columns:
    ISIN, YahooTicker, shares

Main features:
- uses YahooTicker directly from the CSV
- converts all holdings into one base currency
- avoids artificial early jumps by starting only when all holdings have data
- compares against a benchmark ETF/index

Assumptions:
- current holdings were held unchanged throughout the full analysis period
- share counts remained constant
- adjusted close prices approximate total-return behaviour where available
- no taxes, fees, cash flows, or transaction timing are modeled

Install:
    conda install pandas matplotlib
    pip install yfinance

Example:
    python portfolio_backcast.py portfolio.csv \
        --benchmark ACWI \
        --benchmark-currency USD \
        --base-currency EUR \
        --period 1y \
        --output-csv portfolio_timeseries.csv \
        --output-plot portfolio_chart.png \
        --show-components
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

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
        description="Reconstruct hypothetical portfolio development from CSV with Yahoo tickers."
    )
    parser.add_argument(
        "csv_file",
        type=Path,
        help="Input CSV file with columns: ISIN, YahooTicker, shares",
    )
    parser.add_argument(
        "--benchmark",
        default="ACWI",
        help="Yahoo Finance ticker for benchmark ETF/index (default: ACWI)",
    )
    parser.add_argument(
        "--benchmark-currency",
        default="USD",
        help="Trading currency of the benchmark ticker (default: USD)",
    )
    parser.add_argument(
        "--base-currency",
        default="EUR",
        help="Base currency for valuation and comparison (default: EUR)",
    )
    parser.add_argument(
        "--period",
        default="1y",
        help="Yahoo Finance period string, e.g. 1y, 2y, 6mo (default: 1y)",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Optional path to save daily time series as CSV",
    )
    parser.add_argument(
        "--output-plot",
        type=Path,
        default=None,
        help="Optional path to save the chart",
    )
    parser.add_argument(
        "--show-components",
        action="store_true",
        help="Print holdings, currencies, and first valid dates",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------
# Input handling
# ---------------------------------------------------------------------

def load_portfolio(csv_file: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_file)

    required = {"ISIN", "YahooTicker", "shares"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.copy()
    df["ISIN"] = df["ISIN"].astype(str).str.strip().str.upper()
    df["YahooTicker"] = df["YahooTicker"].astype(str).str.strip()
    df["shares"] = pd.to_numeric(df["shares"], errors="raise")

    return df


# ---------------------------------------------------------------------
# Yahoo helpers
# ---------------------------------------------------------------------

def download_adjusted_close(tickers: List[str], period: str) -> pd.DataFrame:
    tickers = list(dict.fromkeys(tickers))

    data = yf.download(
        tickers=tickers,
        period=period,
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
        out = pd.DataFrame({tickers[0]: data["Close"]})
        return out.sort_index()

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

    out = pd.concat(close_frames, axis=1).sort_index()
    return out


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


def attach_currencies(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    currencies = []

    for ticker in df["YahooTicker"]:
        cur = get_currency_from_ticker(ticker)
        if not cur:
            raise RuntimeError(f"Could not determine trading currency for ticker {ticker}")
        currencies.append(cur)

    df["currency"] = currencies
    return df


# ---------------------------------------------------------------------
# FX conversion
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
# Portfolio construction
# ---------------------------------------------------------------------

def convert_positions_to_base(
    portfolio_df: pd.DataFrame,
    prices: pd.DataFrame,
    base_currency: str,
) -> pd.DataFrame:
    position_values = {}

    for _, row in portfolio_df.iterrows():
        ticker = row["YahooTicker"]
        shares = float(row["shares"])
        asset_currency = row["currency"]

        if ticker not in prices.columns:
            raise RuntimeError(f"Missing price series for {ticker}")

        asset_price = prices[ticker].copy().sort_index()
        fx_series = get_fx_series(asset_currency, base_currency, prices)

        position_values[ticker] = asset_price * fx_series * shares

    return pd.DataFrame(position_values).sort_index()


def determine_first_complete_date(df: pd.DataFrame) -> pd.Timestamp:
    valid_mask = df.notna().all(axis=1)
    if not valid_mask.any():
        raise RuntimeError("No date found where all required series are simultaneously available.")
    return valid_mask[valid_mask].index[0]


def build_portfolio_series(
    portfolio_df: pd.DataFrame,
    prices: pd.DataFrame,
    base_currency: str,
) -> Tuple[pd.Series, pd.DataFrame]:
    tickers = portfolio_df["YahooTicker"].tolist()

    required_cols = set(tickers)
    for cur in portfolio_df["currency"].unique():
        if cur != base_currency:
            required_cols.add(FX_MAP[(cur, base_currency)].ticker)

    px = prices[list(required_cols)].copy().sort_index()
    px = px.ffill()

    position_values = convert_positions_to_base(portfolio_df, px, base_currency)

    first_complete_date = determine_first_complete_date(position_values)
    position_values = position_values.loc[first_complete_date:].copy()
    position_values = position_values.dropna(how="any")

    portfolio_value = position_values.sum(axis=1)
    portfolio_value.name = f"portfolio_value_{base_currency}"

    return portfolio_value, position_values


def build_benchmark_series(
    benchmark_ticker: str,
    benchmark_currency: str,
    prices: pd.DataFrame,
    base_currency: str,
) -> pd.Series:
    if benchmark_ticker not in prices.columns:
        raise RuntimeError(f"Benchmark series {benchmark_ticker} was not downloaded.")

    benchmark_price = prices[benchmark_ticker].copy().sort_index().ffill()
    benchmark_fx = get_fx_series(benchmark_currency, base_currency, prices)

    benchmark_in_base = benchmark_price * benchmark_fx
    benchmark_in_base.name = f"benchmark_value_{base_currency}"

    return benchmark_in_base


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
        portfolio = load_portfolio(args.csv_file)
        portfolio = attach_currencies(portfolio)

        if args.show_components:
            print("\nPortfolio holdings:")
            print(portfolio[["ISIN", "YahooTicker", "shares", "currency"]].to_string(index=False))

        fx_pairs = required_fx_pairs(
            currencies=portfolio["currency"].tolist() + [args.benchmark_currency],
            base_currency=args.base_currency,
        )
        fx_tickers = [meta.ticker for meta in fx_pairs.values()]

        download_tickers = portfolio["YahooTicker"].tolist() + [args.benchmark] + fx_tickers
        prices = download_adjusted_close(download_tickers, args.period)

        if args.show_components:
            print_first_valid_dates(prices)

        portfolio_value, position_values = build_portfolio_series(
            portfolio_df=portfolio,
            prices=prices,
            base_currency=args.base_currency,
        )

        benchmark_value = build_benchmark_series(
            benchmark_ticker=args.benchmark,
            benchmark_currency=args.benchmark_currency,
            prices=prices,
            base_currency=args.base_currency,
        )

        combined = pd.concat([portfolio_value, benchmark_value], axis=1).dropna(how="any")
        if combined.empty:
            raise RuntimeError("No overlapping date range between portfolio and benchmark.")

        combined["portfolio_index"] = normalize_to_100(combined[portfolio_value.name])
        combined["benchmark_index"] = normalize_to_100(combined[benchmark_value.name])

        portfolio_return = combined["portfolio_index"].iloc[-1] / 100.0 - 1.0
        benchmark_return = combined["benchmark_index"].iloc[-1] / 100.0 - 1.0

        print("\nAnalysis summary")
        print(f"  Base currency     : {args.base_currency}")
        print(f"  Benchmark ticker  : {args.benchmark}")
        print(f"  Benchmark currency: {args.benchmark_currency}")
        print(f"  Start date        : {combined.index.min().date()}")
        print(f"  End date          : {combined.index.max().date()}")
        print(f"  Portfolio return  : {portfolio_return: .2%}")
        print(f"  Benchmark return  : {benchmark_return: .2%}")

        if args.output_csv is not None:
            combined.to_csv(args.output_csv, index_label="date")
            print(f"\nSaved time series to: {args.output_csv}")

        plt.figure(figsize=(11, 6))
        plt.plot(combined.index, combined["portfolio_index"], label="Hypothetical portfolio")
        plt.plot(combined.index, combined["benchmark_index"], label=f"Benchmark ({args.benchmark})")
        plt.axhline(100, linewidth=1)

        plt.title("Hypothetical portfolio development vs benchmark")
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