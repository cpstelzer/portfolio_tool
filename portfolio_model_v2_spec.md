# Portfolio Model v2 — Specification

## Overview

This document specifies ten improvements to `portfolio_model.py`. It is intended as a hand-off to coding agents. The existing codebase reconstructs historical portfolio performance from transaction data and compares it to a benchmark using Time-Weighted Returns (TWR). All existing functionality must be preserved; the items below are additive or replace clearly identified subsystems.

**Provided files for development/testing:**
- `portfolio_model.py` — current codebase to be improved.
- `transactions_detailed.csv` — real broker export (~100 rows, 2023-12 to 2026-03). Use as the primary test input. See §Test Data for format details.
- `isin_to_yahoo.csv` — ISIN-to-Yahoo-ticker lookup table.
- `transaction.csv` — legacy simplified input (superseded by `transactions_detailed.csv`, but the model should still accept this format for backward compatibility via auto-detection).

---

## Test Data

A concrete broker export is provided as **`transactions_detailed.csv`** in the codebase. Agents should use this file for development and testing. It is a real portfolio export covering 2023-12–2026-03 (~100 rows) and contains buys, sells, thesaurierung events, and distributions. Key properties:

- **Encoding:** Latin-1 (ISO 8859-1). The loader must handle this (not UTF-8).
- **Separator:** comma.
- **Columns:** `Buchungstag`, `Valuta`, `Bezeichnung`, `ISIN`, `Nominal (Stk.)`, (unit), `Betrag`, (unit), `Kurs`, (unit), `Devisenkurs`, `TA.-Nr.`, `Buchungsinformation`.
- The file contains interleaved columns with unit labels (e.g. `Stück`, `€`). These occupy their own CSV columns and must be skipped during parsing.
- All amounts and prices are in EUR (Devisenkurs is always `1.00` in this file).
- `Buchungsinformation` contains the booking type strings used for auto-classification (§9): `Ausführung ORDER Kauf`, `Ausführung ORDER Verkauf`, `Thesaurierung transparenter Fonds`, `Erträgnisausschüttung`, etc.

The loader should parse this broker format directly — do **not** require the user to manually reformat it. The column mapping from broker format to internal representation is:

| Broker column | Internal field |
|---|---|
| `Buchungstag` | `date` |
| `Bezeichnung` | `name` |
| `ISIN` | `ISIN` |
| `Nominal (Stk.)` | `shares` |
| `Betrag` | `gross_amount` |
| `Kurs` | `price` |
| `Devisenkurs` | `fx_rate` |
| `TA.-Nr.` | `tx_id` (optional, for deduplication/traceability) |
| `Buchungsinformation` | `booking_info` |

Earlier transaction history (pre-2023-12) that is not covered by this broker export can be provided via the same file format or via `transactions_manual.csv` (see below).

---

## Input File Changes

### Primary input: `transactions_detailed.csv` (broker export)

The primary transaction input is a CSV exported from the broker (Flatex/DADAT format). The model parses it directly using the column mapping above. All broker-reported events live in this one file: buys, sells, thesaurierung, distributions (Erträgnisausschüttung), and fusions. The auto-classifier (§9) routes each row to the correct processing path.

No separate `tax_events.csv` is needed — tax events are identified by their `Buchungsinformation` text and processed inline.

### Secondary input: `transactions_manual.csv` (off-broker transactions)

A separate, simpler CSV for transactions not executed through the broker — e.g. buying/selling physical gold, private OTC trades, or any asset the broker doesn't know about.

**Columns:**

| Column | Type | Required | Description |
|---|---|---|---|
| `date` | string | yes | `DD.MM.YYYY` or `YYYY-MM-DD` |
| `name` | string | yes | Human-readable description (e.g. `Physical Gold 1oz`) |
| `ticker` | string | yes | Yahoo Finance symbol for price tracking (e.g. `GC=F` for gold) |
| `shares` | float | yes | Signed: positive = buy, negative = sell |
| `price` | float | yes | Per-unit execution price |
| `currency` | string | yes | ISO 4217 code of the price (`USD`, `EUR`, etc.) |
| `fx_rate` | float | no | EUR/foreign rate at execution. If omitted, the model uses its own daily FX rate |
| `notes` | string | no | Free text (e.g. `Bought from dealer X`) |

This file is **optional**. If provided (via `--manual-transactions` CLI arg), its rows are merged into the main transaction stream by date before portfolio reconstruction. These rows always classify as `buy` or `sell` based on the sign of `shares` — no `booking_info` parsing needed.

**Key difference from broker file:** this format is hand-maintained, uses explicit column names (no unit-label columns), and always requires `ticker`, `price`, and `currency` since there's no broker to infer them from.

### `isin_to_yahoo.csv` — add currency column

Add a `Currency` column (ISO 4217) so the model knows each ticker's price denomination without guessing:

```
ISIN,YahooTicker,Name,Currency,Notes
IE00B5BMR087,SXR8.DE,iShares Core S&P 500,EUR,XETRA
```

For direct-ticker assets (gold, crypto), the `currency` column in `transactions.csv` serves the same purpose.

---

## Improvement Specifications

### 1. Currency Handling

**Problem.** The model sums mark-to-market values across tickers as if all prices are in the same currency. Gold futures (GC=F) and crypto pairs (BTC-USD) are priced in USD on Yahoo Finance, while XETRA ETFs are in EUR.

**Solution.**

- Base portfolio currency: **EUR**.
- Each ticker has a known price currency, sourced from: (a) the `Currency` column in `isin_to_yahoo.csv`, or (b) the `currency` column in `transactions.csv` for direct-ticker rows.
- On startup, identify all foreign currencies in the portfolio. Download daily FX rates from Yahoo Finance (e.g. `EURUSD=X` → gives EUR per 1 USD; invert as needed). Store as a `fx_rates: pd.DataFrame` with columns per currency pair indexed by trading day.
- `_mark_to_market()` converts each position's value to EUR before summing: `value_eur = shares × price_foreign × (1 / fx_rate)` where `fx_rate` is the EUR/foreign rate for that day.
- For transaction cost basis: use the broker-reported `fx_rate` when available. Fall back to the model's daily FX rate when not.
- Add a CLI flag `--base-currency` (default `EUR`) for future flexibility, but only EUR needs to work now.

**Validation.** After implementation, the model's computed EUR value on transaction dates should closely match the broker-reported `gross_amount` in EUR. Add a warning if they diverge by more than 2%.

---

### 2. Portfolio Value Tracking

**Problem.** The model only outputs TWR (a unitless return index). Users also want to see the actual EUR value of the portfolio over time and understand how much is capital vs. appreciation.

**Additions.**

- Track two cumulative series indexed by trading day:
  - `cumulative_invested`: running sum of all capital injected (buy transactions in EUR). Decreases on sells by the original cost basis of the sold lot (not the sale proceeds).
  - `portfolio_value_eur`: the mark-to-market portfolio value in EUR (already computed in `reconstruct_portfolio`, but now currency-corrected per §1).
- Derived series:
  - `cumulative_gain = portfolio_value_eur − cumulative_invested`
  - `cumulative_gain_pct = cumulative_gain / cumulative_invested × 100`
- Report at the end of the run:
  - Current total portfolio value in EUR.
  - Per-holding breakdown: ticker, shares held, current price (native currency), current value (EUR), weight (%).
  - Total capital invested, total gain/loss, gain/loss %.
- New plot: stacked area chart with `cumulative_invested` (bottom) and `cumulative_gain` (top, can be negative) over time. This shows how much of the portfolio's current value is "your money" vs. market appreciation.
- Export these series in the optional `--export_csv` output alongside TWR.

---

### 3. Austrian Tax Simulation

**Problem.** No tax modelling exists. Austrian KESt (27.5% flat) applies to: (a) realised capital gains on sells, (b) deemed distributions of accumulating ETFs (Thesaurierung), (c) actual distributions of distributing ETFs (Ausschüttung).

**Design.**

#### 3a. Per-Lot Cost Basis Tracking (FIFO)

Replace the current `positions: dict[str, float]` (aggregate share count per ticker) with a **lot-based ledger**:

```python
@dataclass
class Lot:
    ticker: str
    date: pd.Timestamp
    shares: float          # remaining shares in this lot
    cost_price_eur: float  # per-share cost in EUR at acquisition
    thesaurierung_adj: float  # cumulative per-share thesaurierung adjustment (increases cost basis)
```

- On buy: create a new `Lot`.
- On sell: consume lots in FIFO order (oldest first). For each consumed lot, compute realised gain: `(sell_price_eur − adjusted_cost_basis_eur) × shares_sold`.
- `adjusted_cost_basis_eur = cost_price_eur + thesaurierung_adj` per share.

#### 3b. Thesaurierung (Accumulating ETF Deemed Distribution)

When a `thesaurierung` event is encountered in `transactions_detailed.csv` (identified by auto-classification §9):

**Important: broker thesaurierung row structure.** The broker records thesaurierung as paired rows per lot: a negative-shares row at the old NAV followed by a positive-shares row at the new (higher) NAV, with the same absolute share count. For example, for 40 shares of SXR8:
- Row 1: shares=-40, Betrag=-13440.12, Kurs=336.00 (removal at old NAV)
- Row 2: shares=+40, Betrag=+13900.17, Kurs=347.50 (re-addition at new NAV)

The **taxable amount** is the difference in Betrag: 13900.17 − 13440.12 = 460.05. The per-share cost basis increase is (347.50 − 336.00) = 11.50. The model must pair these rows (match by ISIN + date + absolute share count) and compute the delta rather than treating them as separate buy/sell transactions.

1. Find all lots of the matching ISIN with shares ≥ the event's `shares` (the broker reports per-lot).
2. Compute tax: `amount × 0.275`.
3. Increase the lot's `thesaurierung_adj` by `amount / shares` (per-share cost basis uplift).
4. Record the tax payment in the tax timeline.

#### 3c. Ausschüttung (Actual Distribution)

When an `ausschuettung` event is encountered:

1. Tax: `amount × 0.275`.
2. The distribution is cash received; add to portfolio cash (or treat as reinvested if shares increase in a paired buy transaction).
3. Record in tax timeline.

#### 3d. Fusion (Fund Merger)

When a `fusion` event is encountered:

1. Swap the ISIN/ticker on all matching lots to `new_ISIN`.
2. Carry over cost basis and thesaurierung adjustments unchanged.
3. Update `isin_to_ticker` mapping if `new_ISIN` has a different Yahoo ticker.

#### 3e. Sell Taxation

On every sell transaction:

1. Consume FIFO lots as described in §3a.
2. For each lot consumed: `gain = (sell_price_eur − adjusted_cost_basis_eur) × shares`.
3. If `gain > 0`: tax = `gain × 0.275`.
4. If `gain < 0`: record as loss (losses can offset gains within the same year in Austria, but cross-year loss carry-forward is not permitted for private investors — document this limitation).
5. Record in tax timeline.

#### 3f. Tax Timeline Output

Produce a `tax_timeline` DataFrame:

| Column | Description |
|---|---|
| `date` | Date of tax event |
| `event_type` | `sell`, `thesaurierung`, `ausschuettung` |
| `ticker` | Affected instrument |
| `taxable_amount` | Gross taxable amount in EUR |
| `tax` | KESt due (27.5%) |
| `cumulative_tax` | Running total |

- Print a yearly summary table: total KESt per year, broken down by event type.
- Plot cumulative tax paid over time as an optional chart.
- Make the entire tax simulation opt-in via `--tax` CLI flag.

---

### 4. Spike Filter Replacement

**Problem.** The current fixed-threshold filter (15% equity / 50% crypto) clips legitimate market moves (e.g. COVID crash in March 2020 saw many ETFs move 10-15% in a day). Interpolating real price action corrupts the data.

**Replacement: reversal-based detection.**

A data error from Yahoo Finance (e.g. bad adjusted-close after a dividend restatement) typically produces a spike that reverses the next day — the price jumps and then snaps back. A real crash does not reverse immediately.

Algorithm:

1. Compute daily returns `r[t]` for each ticker.
2. Flag day `t` as a candidate spike if `|r[t]| > threshold` (keep current thresholds as the candidate filter, not the final filter).
3. A candidate is confirmed as a data error only if the move substantially reverses on day `t+1`: specifically, `r[t+1]` has opposite sign to `r[t]` **and** `|r[t+1]| > 0.5 × |r[t]|` (i.e., more than half the spike is "given back").
4. Confirmed spikes: replace day `t` price with linear interpolation, then re-check.
5. Non-reversing large moves (real crashes / rallies): leave untouched.

Additionally:

- Make the candidate threshold configurable via `--spike-threshold` (default 0.15 for equities, 0.50 for crypto).
- Log all detected and corrected spikes at `--verbose` level.

---

### 5. Cash Tracking, Deposits & Withdrawals

**Original problem.** The cash model was implicit and undocumented; no withdrawals existed.

**Resolution.** The model now has two cash modes, selected via `config.yml → clearing_account.use_clearing_account` (default `true`):

**Legacy mode (`use_clearing_account: false`).**
- Buys are modelled as external capital injection: money enters the portfolio and is immediately converted to shares. Net cash effect of a buy = 0.
- Sells convert shares to cash. Cash remains in the portfolio.
- No deposits or withdrawals. Cash only grows (from sells) and is never removed.

**Clearing-account mode (`use_clearing_account: true`, default).**
- The broker's `cashflow_detailed.csv` ledger is loaded and classified into `deposit`, `withdrawal`, `trade_leg`, `distribution`, `thesaurierung`, `clearing_interest`, and `unclassified` categories.
- Empty info fields are reclassified as deposits/withdrawals when the counterparty IBAN contains the configured `reference_iban_suffix`.
- `cash` is a real running balance: deposits add, withdrawals subtract, optional `Zinsabschluss` interest applies, sells/distributions credit, buys debit.
- A new series `cumulative_contributed = deposits − withdrawals` is reported alongside the legacy `cumulative_invested` (which still tracks gross buy volume).
- The summary prints both total-return percentages (gross-invested and net-contributed) plus the **money-weighted return (XIRR)**.
- The "mirror flows" benchmark mirrors real external flows.
- Unmatched ledger rows or settlement-quirk negative cash balances emit warnings but do not abort.

---

### 6. Per-Asset & Per-Sector Attribution

**Additions.**

- At the end of the run, print a **contribution table** showing each holding's contribution to total portfolio return over the analysis period. Contribution = `(weight × asset_return)` summed daily, or simplified as `(end_value − cost_basis) / total_invested`.
- Group holdings by asset class. Infer asset class from a new optional column `asset_class` in `isin_to_yahoo.csv` (values: `equity`, `bond`, `commodity`, `crypto`, `other`). If absent, default to `other`.
- Print allocation summary: current weight per asset class.
- Optional: plot allocation drift over time (stacked area chart by asset class).

---

### 7. Additional Risk Metrics

Add to the existing `compute_metrics()` output:

| Metric | Formula |
|---|---|
| CAGR | `(end/start)^(365/days) − 1` |
| Sortino ratio | `mean(daily_ret) / downside_std × √252`, where downside_std uses only negative returns |
| Calmar ratio | `CAGR / |max_drawdown|` |
| Rolling 1-year return | Series: for each day, TWR over trailing 252 trading days. Plot as a subplot below the main TWR chart |
| Benchmark correlation | Pearson correlation of daily portfolio returns vs. daily benchmark returns |

---

### 8. Dividend / Distribution Handling

**Context.** Some holdings are distributing ETFs. Yahoo Finance adjusted close already accounts for reinvested dividends, which is correct for TWR. But the model should also track actual cash distributions received.

**Additions.**

- Distributing ETFs that appear in `transactions_detailed.csv` as `ausschuettung` events (auto-classified per §9): the `amount` is recorded as cash received (added to portfolio cash in `reconstruct_portfolio`).
- Document the assumption: TWR uses Yahoo adjusted close (dividends reinvested in price). The cash tracking and tax simulation use actual distribution amounts from broker data. These are intentionally different views.
- No attempt to scrape dividend data from Yahoo; rely entirely on broker-reported events.

---

### 9. Auto-Classification of Transaction Types

**Problem.** The current model infers buy/sell purely from the sign of `shares`. Richer classification enables tax simulation and better reporting.

**Implementation.**

Parse the `booking_info` column from `transactions.csv` using pattern matching to assign a `tx_type`:

| Pattern in `booking_info` | `tx_type` |
|---|---|
| Contains `Kauf` or `ORDER Kauf` | `buy` |
| Contains `Verkauf` or `ORDER Verkauf` | `sell` |
| Contains `Thesaurierung` | `thesaurierung` |
| Contains `Fusion` | `fusion` |
| Contains `Ausschüttung` or `Erträgnisausschüttung` | `ausschuettung` |
| Contains `Spin-off` or `Kapitalmaßnahme` | `corporate_action` |
| No match | Fall back to sign of `shares`: positive → `buy`, negative → `sell` |

- The `tx_type` is used downstream by the tax simulation (§3) and attribution (§6).
- If `booking_info` is empty or absent, fall back to the sign-based heuristic. This ensures backward compatibility with the simplified CSV format.
- Log classified types at `--verbose` level so the user can verify.

---

### 10. Robustness & UX

#### 10a. Local Price Cache

**Problem.** Every run re-downloads all price history from Yahoo Finance. Slow and wasteful.

**Implementation.**

- Cache directory: `~/.portfolio_cache/` (configurable via `--cache-dir`).
- One CSV file per ticker: `{ticker}.csv` with columns `date, close`.
- On each run:
  1. For each needed ticker, check if a cache file exists.
  2. If yes, load it and determine the last cached date.
  3. Download only the missing date range from Yahoo Finance (from `last_cached_date + 1` to `end_date`).
  4. Append new rows to the cache file.
  5. If no cache exists, download the full range and create the file.
- Add `--no-cache` flag to force a full re-download.
- Add `--clear-cache` flag to delete all cached files.
- FX rate data (§1) is also cached using the same mechanism.

#### 10b. Diagnostics Behind `--verbose`

- Move the entire diagnostic block (last 20 rows, large-move detection, per-ticker price/change dump) behind a `--verbose` flag. Default runs show only the performance summary, holdings table, and charts.
- Under `--verbose`, also print: resolved ticker mapping, transaction type classifications, spike filter actions, FX rates used.

#### 10c. Minor Cleanups

- Remove the unused `gross_amount` fallback for price resolution if `booking_info`-based classification makes it redundant. Or keep it as a documented last resort.
- Add `--no-plot` flag to suppress chart display (useful for scripting / CI).
- Add `--holdings` flag to print the current holdings breakdown table without running the full analysis.

---

## Implementation Priority

Suggested ordering for implementation (dependencies flow downward):

1. **§10b** — Move diagnostics behind `--verbose` (small, unblocks cleaner output for everything else).
2. **§9** — Auto-classification of transaction types (required by §3).
3. **§1** — Currency handling (required by §2 and §3 for correct EUR values).
4. **§4** — Spike filter replacement (improves price data quality for all downstream steps).
5. **§10a** — Price cache (quality of life, speeds up iteration during development of remaining items).
6. **§5** — Document cash/sell assumptions (small, clarifying).
7. **§3** — Austrian tax simulation including per-lot cost basis (largest item; depends on §1, §9).
8. **§2** — Portfolio value tracking and capital vs. appreciation decomposition (depends on §1, benefits from §3 lot tracking).
9. **§8** — Dividend/distribution handling (small, integrates with §3).
10. **§6** — Per-asset and per-sector attribution.
11. **§7** — Additional risk metrics.
12. **§10c** — Minor cleanups and extra CLI flags.

---

## Out of Scope

- Cross-year tax loss carry-forward (not permitted for Austrian private investors).
- Multi-user or web interface.
- Real-time / streaming price updates.
- Automatic OeKB data retrieval (deemed distribution data comes from broker export).
- Google Sheets or database integration.
