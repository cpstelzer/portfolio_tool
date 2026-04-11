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


def _mark_to_market(positions: dict[str, float], prices: pd.DataFrame, day: pd.Timestamp, base: str, lookup_df: pd.DataFrame, transactions: pd.DataFrame, fx_rates: pd.DataFrame) -> float:
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
        val_eur += sh * px_eur
        
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
    """
    
    cash = 0.0
    cumulative_invested = 0.0
    cumulative_tax = 0.0
    cash_flows = {}
    rows = []
    
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
            
        val_after = cash + _mark_to_market(positions, prices, day, base_cur, lookup_df, transactions, fx_rates)
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
    
    portfolio_df = reconstruct_portfolio(transactions, isin_to_ticker, lookup_df, prices, fx_rates, args)
    cash_flows = portfolio_df.cash_flows
    
    benchmark_df = simulate_benchmark(cash_flows, prices[args.benchmark], args.buy_and_hold)
    
    pv_twr = compute_twr(portfolio_df["portfolio_value"], portfolio_df["value_before_tx"])
    bv_twr = compute_twr(benchmark_df["benchmark_value"], benchmark_df["value_before_tx"])
    
    common = pv_twr.dropna().index.intersection(bv_twr.dropna().index)
    pv_twr = pv_twr.loc[common] / pv_twr.loc[common].iloc[0]
    bv_twr = bv_twr.loc[common] / bv_twr.loc[common].iloc[0]
    
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
    for h in holdings:
        w = (h['value_eur']/final_val*100) if final_val > 0 else 0
        print(f"{h['ticker']:<15} | Sh: {h['shares']:<10.4f} | Px: {h['price_native']:<10.2f} | Val({args.base_currency}): {h['value_eur']:<12.2f} | {w:.1f}%")
        
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
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [2, 1]})
        
        ax1.plot(pv_twr.index, pv_twr.values, label="Portfolio TWR")
        ax1.plot(bv_twr.index, bv_twr.values, label=f"Benchmark ({args.benchmark})", linestyle="--")
        ax1.set_title("Time-Weighted Return")
        ax1.legend()
        
        ax2.fill_between(portfolio_df.index, 0, portfolio_df["cumulative_invested"], label="Capital Invested", alpha=0.5)
        gain = portfolio_df["portfolio_value"] - portfolio_df["cumulative_invested"]
        ax2.fill_between(portfolio_df.index, portfolio_df["cumulative_invested"], portfolio_df["portfolio_value"], label="Appreciation", alpha=0.3)
        ax2.set_title("Capital vs Appreciation (EUR)")
        ax2.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()
