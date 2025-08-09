class StatArbOptionsBacktester:
    def __init__(self, initial_capital=100000, max_position_size=0.05, max_concurrent_trades=10):
        self.initial_capital = initial_capital
        self.max_position_size = max_position_size  # 5% max position size
        self.max_concurrent_trades = max_concurrent_trades
        self.risk_free_rate = 0.05
        
        # Portfolio tracking
        self.cash = initial_capital
        self.portfolio_value = []
        self.portfolio_dates = []
        self.active_trades = {}
        self.completed_trades = []
        self.trade_id_counter = 0
        
        # Benchmark tracking
        self.benchmark_values = []
        self.benchmark_dates = []
        
        # Scanner instance
        self.scanner = None
        
    def initialize_scanner(self, lookback_period=252, deviation_threshold_min=2.25, deviation_threshold_max=2.5):
        """Initialize the scanner with given parameters"""
        from blackarb import sp500arbscan  # Import your scanner class
        
        self.scanner = sp500arbscan(
            lookback_period=lookback_period,
            deviation_threshold_min=deviation_threshold_min,
            deviation_threshold_max=deviation_threshold_max
        )
        
    def get_sp500_symbols(self):
        """Get S&P 500 symbols from Wikipedia (same as your blackarb.py strategy)"""
        print("Fetching S&P 500 symbols from Wikipedia...")
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        df = tables[0]  # find table 
        symbols = df['Symbol'].tolist()
        
        # Replace '.' with '-' for yfinance compatibility
        symbols = [symbol.replace('.', '-') for symbol in symbols]
        print(f"Found {len(symbols)} S&P 500 symbols")
        return symbols
        
    def get_historical_sp500_data(self, start_date, end_date, min_move=0.40):
        """Get historical S&P 500 data for backtesting using Wikipedia symbols"""
        print(f"Fetching S&P 500 data from {start_date} to {end_date}...")
        
        # Get S&P 500 symbols from Wikipedia (like your strategy)
        symbols = self.get_sp500_symbols()
        
        
        symbols = symbols[:503]
        
        # Download historical data in batches
        all_data = {}
        batch_size = 10  
        
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            try:
                print(f"Downloading batch {i//batch_size + 1}/{(len(symbols)-1)//batch_size + 1}")
                data = yf.download(batch, start=start_date, end=end_date, progress=False)
                
                if data.empty:
                    print(f"No data returned for batch {batch}")
                    continue
                
                # Handle MultiIndex columns from yfinance
                if hasattr(data.columns, 'levels'):  # MultiIndex columns
                    # Look for Close prices
                    for symbol in batch:
                        try:
                            #Diff column structures
                            close_col = None
                            if ('Close', symbol) in data.columns:
                                close_col = ('Close', symbol)
                            elif (symbol, 'Close') in data.columns:
                                close_col = (symbol, 'Close')
                            
                            if close_col:
                                prices = data[close_col].dropna()
                                if len(prices) > 0:
                                    all_data[symbol] = prices
                                    print(f"Added {symbol}: {len(prices)} data points")
                        except Exception as se:
                            print(f"Error processing {symbol}: {se}")
                            continue
                else:
                    # Single columns
                    if len(batch) == 1:
                        symbol = batch[0]
                        if 'Close' in data.columns:
                            prices = data['Close'].dropna()
                            if len(prices) > 0:
                                all_data[symbol] = prices
                                print(f"Added {symbol}: {len(prices)} data points")
                
                time.sleep(0.5)  # Rate limiting
            except Exception as e:
                print(f"Error downloading batch {batch}: {e}")
                continue
        
        print(f"Downloaded data for {len(all_data)} symbols")
        
        # Filter for significant movers 
        print("Filtering for significant movers...")
        filtered_data = {}
        
        for symbol, prices in all_data.items():
            if len(prices) < 50:  # Need sufficient data
                continue
            
            try:
                # Check for significant moves over the period
                if len(prices) > 0:
                    total_move = abs((prices.iloc[-1] - prices.iloc[0]) / prices.iloc[0])
                    if total_move >= min_move:
                        filtered_data[symbol] = prices
                    else:
                        # stable stocks for pairs trading
                        if len(filtered_data) < 50:  # Include some stable stocks
                            filtered_data[symbol] = prices
            except Exception as e:
                print(f"Error processing {symbol}: {e}")
                continue
        
        print(f"Found {len(filtered_data)} symbols with significant moves")
        return filtered_data
    
    def get_benchmark_data(self, start_date, end_date):
        """Get S&P 500 benchmark data"""
        print("Fetching S&P 500 benchmark data...")
        spy_data = yf.download('SPY', start=start_date, end=end_date, progress=False)
        
        # Handle both single and multi-level columns
        if hasattr(spy_data.columns, 'levels'):
            # MultiIndex columns
            if ('Close', 'SPY') in spy_data.columns:
                return spy_data[('Close', 'SPY')]
            elif 'Close' in spy_data.columns.get_level_values(0):
                return spy_data['Close'].iloc[:, 0]  # Get first Close column
        else:
            # Single level columns
            if 'Close' in spy_data.columns:
                return spy_data['Close']
        
        print("Warning: Could not find Close price in SPY data")
        return pd.Series()
    
    def calculate_option_prices_historical(self, S, K, T, r, sigma, option_type='call'):
        """Calculate historical option prices using Black-Scholes"""
        if T <= 0:
            return max(S - K, 0) if option_type == 'call' else max(K - S, 0)
        
        try:
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            
            if option_type == 'call':
                price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
            else:
                price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
            
            return max(price, 0.01)  # Minimum option price
        except:
            return max(S - K, 0) if option_type == 'call' else max(K - S, 0)
    
    def calculate_hedge_ratio(self, asset1_prices, asset2_prices):
        """Calculate optimal hedge ratio using linear regression (same as your strategy)"""
        data = pd.DataFrame({'asset1': asset1_prices, 'asset2': asset2_prices}).dropna()
        if len(data) < 10:
            return 1.0
        
        X = data['asset2'].values.reshape(-1, 1)
        y = data['asset1'].values
        
        model = LinearRegression()
        model.fit(X, y)
        return model.coef_[0]
    
    def find_opportunities_on_date(self, price_data, date, lookback_days=252):
        """Find statistical arbitrage opportunities on a specific date (using your strategy logic)"""
        opportunities = []
        
        # Get price data up to this date for correlation analysis
        end_idx = None
        for symbol in price_data:
            if date in price_data[symbol].index:
                end_idx = price_data[symbol].index.get_loc(date)
                break
        
        if end_idx is None or end_idx < lookback_days:
            return opportunities
        
        # Build correlation matrix using lookback period
        lookback_data = {}
        for symbol, prices in price_data.items():
            if len(prices) > end_idx and end_idx >= lookback_days:
                lookback_data[symbol] = prices.iloc[max(0, end_idx-lookback_days):end_idx+1]
        
        if len(lookback_data) < 10:  # Need sufficient symbols
            return opportunities
        
        # Create DataFrame and calculate correlations (same as your find_correlated_pairs_optimized)
        df = pd.DataFrame(lookback_data).dropna()
        if len(df) < 50:  # Need sufficient observations
            return opportunities
        
        correlations = df.corr()
        symbols = list(df.columns)
        
        # Find highly correlated pairs (using your correlation logic)
        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                symbol1, symbol2 = symbols[i], symbols[j]
                correlation = correlations.loc[symbol1, symbol2]
                
                # Use same correlation threshold as your strategy
                if pd.isna(correlation) or abs(correlation) < 0.87:
                    continue
                
                # Calculate spread and z-score (same as your strategy)
                prices1 = df[symbol1]
                prices2 = df[symbol2]
                
                hedge_ratio = self.calculate_hedge_ratio(prices1, prices2)
                spread = prices1 - hedge_ratio * prices2
                spread_mean = spread.mean()
                spread_std = spread.std()
                
                if spread_std <= 0:
                    continue
                
                current_z_score = (spread.iloc[-1] - spread_mean) / spread_std
                
                # Check if within deviation thresholds (same as your strategy)
                abs_z_score = abs(current_z_score)
                if 2.0 <= abs_z_score <= 2.5:  # Use your original deviation thresholds
                    current_price1 = prices1.iloc[-1]
                    current_price2 = prices2.iloc[-1]
                    
                    opportunities.append({
                        'asset1': symbol1,
                        'asset2': symbol2,
                        'correlation': correlation,
                        'hedge_ratio': hedge_ratio,
                        'z_score': current_z_score,
                        'current_price1': current_price1,
                        'current_price2': current_price2,
                        'direction': 'short' if current_z_score > 0 else 'long',
                        'deviated_asset': symbol1,
                        'confidence': min(abs_z_score / 3.0, 1.0)
                    })
        
        return opportunities
    
    def create_options_strategy(self, opportunity, date, historical_vol=0.3):
        """Create options strategy based on opportunity (same logic as your determine_option_strats)"""
        symbol = opportunity['deviated_asset']
        current_price = opportunity['current_price1']
        direction = opportunity['direction']
        z_score = opportunity['z_score']
        
        # Set expiration based on z-score magnitude (same as your strategy)
        abs_z_score = abs(z_score)
        if abs_z_score > 2.43:
            days_to_expiration = 21
        elif abs_z_score > 2.35:
            days_to_expiration = 35
        else:
            days_to_expiration = 45
        
        T = days_to_expiration / 365
        
        # Calculate strikes (same as your strategy)
        if direction == 'short':  # Bear Put Spread
            long_strike = round(current_price)
            short_strike = round(current_price * 0.95)
            
            long_price = self.calculate_option_prices_historical(
                current_price, long_strike, T, self.risk_free_rate, historical_vol, 'put'
            )
            short_price = self.calculate_option_prices_historical(
                current_price, short_strike, T, self.risk_free_rate, historical_vol, 'put'
            )
            
            strategy_type = "Bear Put Spread"
            max_profit = long_strike - short_strike - (long_price - short_price)
            
        else:  # Bull Call Spread
            long_strike = round(current_price)
            short_strike = round(current_price * 1.05)
            
            long_price = self.calculate_option_prices_historical(
                current_price, long_strike, T, self.risk_free_rate, historical_vol, 'call'
            )
            short_price = self.calculate_option_prices_historical(
                current_price, short_strike, T, self.risk_free_rate, historical_vol, 'call'
            )
            
            strategy_type = "Bull Call Spread"
            max_profit = short_strike - long_strike - (long_price - short_price)
        
        net_cost = long_price - short_price
        max_loss = abs(net_cost)
        
        # Calculate position size
        max_position_value = self.initial_capital * self.max_position_size
        cost_per_contract = abs(net_cost) * 100  # Options are quoted per share, 100 shares per contract
        
        if cost_per_contract > 0:
            position_size = min(
                int(max_position_value / cost_per_contract),
                int(self.cash / cost_per_contract),
                10  # Maximum 10 contracts per trade
            )
        else:
            position_size = 0
        
        if position_size <= 0:
            return None
        
        strategy = {
            'symbol': symbol,
            'strategy_type': strategy_type,
            'direction': direction,
            'entry_date': date,
            'expiration_date': date + timedelta(days=days_to_expiration),
            'long_strike': long_strike,
            'short_strike': short_strike,
            'long_price': long_price,
            'short_price': short_price,
            'net_cost': net_cost,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'position_size': position_size,
            'total_cost': abs(net_cost) * position_size * 100,
            'opportunity': opportunity,
            'historical_vol': historical_vol,
            'z_score': z_score
        }
        
        return strategy
    
    def enter_trade(self, strategy, date):
        """Enter a new options trade"""
        if len(self.active_trades) >= self.max_concurrent_trades:
            return False, "Maximum concurrent trades reached"
        
        if self.cash < strategy['total_cost'] or strategy['total_cost'] <= 0:
            return False, "Insufficient capital or invalid trade cost"
        
        self.trade_id_counter += 1
        trade_id = f"{strategy['symbol']}_{self.trade_id_counter}"
        
        trade = strategy.copy()
        trade.update({
            'trade_id': trade_id,
            'status': 'active',
            'current_pnl': 0,
            'current_value': strategy['total_cost']
        })
        
        self.cash -= strategy['total_cost']
        self.active_trades[trade_id] = trade
        
        return True, trade_id
    
    def update_trade_value(self, trade_id, current_price, date):
        """Update trade value based on current stock price"""
        if trade_id not in self.active_trades:
            return False
        
        trade = self.active_trades[trade_id]
        
        # Check if expired
        if date >= trade['expiration_date']:
            return self.close_trade_expiration(trade_id, current_price, date)
        
        # Calculate time to expiration
        days_remaining = (trade['expiration_date'] - date).days
        T = max(days_remaining / 365, 0.001)  # Minimum time value
        
        # Recalculate option prices
        if trade['direction'] == 'short':  # Bear Put Spread
            long_value = self.calculate_option_prices_historical(
                current_price, trade['long_strike'], T, self.risk_free_rate, trade['historical_vol'], 'put'
            )
            short_value = self.calculate_option_prices_historical(
                current_price, trade['short_strike'], T, self.risk_free_rate, trade['historical_vol'], 'put'
            )
        else:  # Bull Call Spread
            long_value = self.calculate_option_prices_historical(
                current_price, trade['long_strike'], T, self.risk_free_rate, trade['historical_vol'], 'call'
            )
            short_value = self.calculate_option_prices_historical(
                current_price, trade['short_strike'], T, self.risk_free_rate, trade['historical_vol'], 'call'
            )
        
        current_net_value = long_value - short_value
        current_total_value = current_net_value * trade['position_size'] * 100
        
        trade['current_value'] = current_total_value
        trade['current_pnl'] = current_total_value - trade['total_cost']
        
        # Check for profit target (50% of max profit)
        profit_target = trade['max_profit'] * trade['position_size'] * 100 * 0.75
        if trade['current_pnl'] >= profit_target:
            return self.close_trade_profit(trade_id, current_price, date)
        
        # Check for stop loss (80% of max loss)
        stop_loss = trade['max_loss'] * trade['position_size'] * 100 * 0.06
        if trade['current_pnl'] <= -stop_loss:
            return self.close_trade_loss(trade_id, current_price, date)
        
        return True
    
    def close_trade_expiration(self, trade_id, current_price, date):
        """Close trade at expiration"""
        return self.close_trade(trade_id, current_price, date, 'expiration')
    
    def close_trade_profit(self, trade_id, current_price, date):
        """Close trade at profit target"""
        return self.close_trade(trade_id, current_price, date, 'profit_target')
    
    def close_trade_loss(self, trade_id, current_price, date):
        """Close trade at stop loss"""
        return self.close_trade(trade_id, current_price, date, 'stop_loss')
    
    def close_trade(self, trade_id, current_price, date, reason):
        """Close a trade and realize P&L"""
        if trade_id not in self.active_trades:
            return False
        
        trade = self.active_trades[trade_id]
        
        # Calculate final value
        if reason == 'expiration':
            # At expiration, calculate intrinsic value
            if trade['direction'] == 'short':  # Bear Put Spread
                long_intrinsic = max(trade['long_strike'] - current_price, 0)
                short_intrinsic = max(trade['short_strike'] - current_price, 0)
            else:  # Bull Call Spread
                long_intrinsic = max(current_price - trade['long_strike'], 0)
                short_intrinsic = max(current_price - trade['short_strike'], 0)
            
            final_value = (long_intrinsic - short_intrinsic) * trade['position_size'] * 100
        else:
            final_value = trade['current_value']
        
        final_pnl = final_value - trade['total_cost']
        
        # Complete trade record
        completed_trade = trade.copy()
        completed_trade.update({
            'exit_date': date,
            'exit_price': current_price,
            'final_value': final_value,
            'final_pnl': final_pnl,
            'exit_reason': reason,
            'status': 'closed'
        })
        
        # Update portfolio
        self.cash += final_value
        self.completed_trades.append(completed_trade)
        del self.active_trades[trade_id]
        
        return True
    
    def calculate_portfolio_value(self, date):
        """Calculate total portfolio value"""
        total_value = self.cash
        
        for trade in self.active_trades.values():
            total_value += trade.get('current_value', trade['total_cost'])
        
        return total_value
    
    def update_benchmark(self, spy_data, date):
        """Update benchmark performance"""
        if date in spy_data.index:
            current_spy_price = spy_data.loc[date]
            initial_spy_price = spy_data.iloc[0]
            benchmark_value = self.initial_capital * (current_spy_price / initial_spy_price)
            
            self.benchmark_values.append(benchmark_value)
            self.benchmark_dates.append(date)
    
    def run_backtest(self, start_date='2021-08-01', end_date='2025-08-01'):
        """Run the complete backtest"""
        print(f"Running Statistical Arbitrage Options Backtest from {start_date} to {end_date}")
        print("="*80)
        
        # Get historical data using Wikipedia S&P 500 symbols (like your strategy)
        price_data = self.get_historical_sp500_data(start_date, end_date)
        
        if not price_data:
            print("Failed to get price data")
            return False
        
        # Get benchmark data
        spy_data = self.get_benchmark_data(start_date, end_date)
        if spy_data.empty:
            print("Failed to get benchmark data")
            return False
        
        # Create date range for backtesting
        start_dt = pd.to_datetime(start_date)
        end_dt = pd.to_datetime(end_date)
        all_dates = pd.date_range(start=start_dt, end=end_dt, freq='B')  # Business days
        
        # Reset portfolio
        self.cash = self.initial_capital
        self.active_trades = {}
        self.completed_trades = []
        self.portfolio_value = []
        self.portfolio_dates = []
        self.benchmark_values = []
        self.benchmark_dates = []
        
        print(f"Backtesting over {len(all_dates)} business days...")
        
        for i, date in enumerate(all_dates):
            if i % 50 == 0:
                print(f"Processing day {i+1}/{len(all_dates)}: {date.strftime('%Y-%m-%d')}")
            
            # Update benchmark
            self.update_benchmark(spy_data, date)
            
            # Update existing trades
            trades_to_close = []
            for trade_id, trade in list(self.active_trades.items()):
                symbol = trade['symbol']
                if symbol in price_data and date in price_data[symbol].index:
                    current_price = price_data[symbol].loc[date]
                    if not self.update_trade_value(trade_id, current_price, date):
                        trades_to_close.append(trade_id)
            
            # Look for new opportunities (weekly scan) switched to daily 
            if i % 1 == 0 and len(self.active_trades) < self.max_concurrent_trades:
                opportunities = self.find_opportunities_on_date(price_data, date)
                
                for opp in opportunities[:5]:  # Limit to top 5 opportunities per scan
                    if len(self.active_trades) >= self.max_concurrent_trades:
                        break
                    
                    # Calculate historical volatility (same as your strategy)
                    symbol = opp['deviated_asset']
                    if symbol in price_data:
                        try:
                            prices = price_data[symbol].loc[:date].tail(30)
                            if len(prices) > 10:
                                returns = prices.pct_change().dropna()
                                hist_vol = returns.std() * np.sqrt(252)
                                hist_vol = max(hist_vol, 0.10)  # Minimum 15% vol, 10% vol
                            else:
                                hist_vol = 0.3
                        except:
                            hist_vol = 0.3
                        
                        strategy = self.create_options_strategy(opp, date, hist_vol)
                        if strategy:
                            success, trade_id = self.enter_trade(strategy, date)
                            if not success:
                                break
            
            # Calculate portfolio value
            portfolio_val = self.calculate_portfolio_value(date)
            self.portfolio_value.append(portfolio_val)
            self.portfolio_dates.append(date)
        
        # Close any remaining trades
        final_date = all_dates[-1]
        for trade_id in list(self.active_trades.keys()):
            trade = self.active_trades[trade_id]
            symbol = trade['symbol']
            if symbol in price_data:
                final_price = price_data[symbol].iloc[-1]
                self.close_trade(trade_id, final_price, final_date, 'end_of_backtest')
        
        print(f"Backtest completed!")
        print(f"Total trades: {len(self.completed_trades)}")
        final_portfolio_value = self.calculate_portfolio_value(final_date)
        print(f"Final portfolio value: ${final_portfolio_value:,.2f}")
        if self.benchmark_values:
            final_benchmark_value = float(self.benchmark_values[-1])
            print(f"Final benchmark value: ${final_benchmark_value:,.2f}")
        else:
            print("No benchmark data available")
        
        return True
    
    def analyze_results(self):
        """Analyze backtest results with benchmark comparison"""
        if not self.portfolio_value or not self.completed_trades:
            print("No results to analyze")
            return None
        
        # Portfolio performance
        portfolio_series = pd.Series(self.portfolio_value, index=self.portfolio_dates)
        benchmark_series = pd.Series(self.benchmark_values, index=self.benchmark_dates)
        
        # Align dates for comparison
        common_dates = portfolio_series.index.intersection(benchmark_series.index)
        portfolio_aligned = portfolio_series.loc[common_dates]
        benchmark_aligned = benchmark_series.loc[common_dates]
        
        # Calculate metrics for portfolio
        total_return = (portfolio_aligned.iloc[-1] / self.initial_capital - 1) * 100
        years = len(portfolio_aligned) / 252
        annual_return = ((portfolio_aligned.iloc[-1] / self.initial_capital) ** (1/years) - 1) * 100
        
        # Portfolio volatility and Sharpe
        portfolio_returns = portfolio_aligned.pct_change().dropna()
        volatility = portfolio_returns.std() * np.sqrt(252) * 100
        sharpe = (annual_return - 5) / volatility if volatility > 0 else 0
        
        # Portfolio drawdown
        cummax = portfolio_aligned.cummax()
        drawdown = (portfolio_aligned - cummax) / cummax
        max_drawdown = drawdown.min() * 100
        
        # Calculate metrics for benchmark
        benchmark_total_return = (benchmark_aligned.iloc[-1] / self.initial_capital - 1) * 100
        benchmark_annual_return = ((benchmark_aligned.iloc[-1] / self.initial_capital) ** (1/years) - 1) * 100
        
        # Benchmark volatility and Sharpe
        benchmark_returns = benchmark_aligned.pct_change().dropna()
        benchmark_volatility = benchmark_returns.std() * np.sqrt(252) * 100
        benchmark_sharpe = (benchmark_annual_return - 5) / benchmark_volatility if benchmark_volatility > 0 else 0
        
        # Benchmark drawdown
        benchmark_cummax = benchmark_aligned.cummax()
        benchmark_drawdown = (benchmark_aligned - benchmark_cummax) / benchmark_cummax
        benchmark_max_drawdown = benchmark_drawdown.min() * 100
        
        # Calculate outperformance metrics
        excess_return = annual_return - benchmark_annual_return
        information_ratio = excess_return / volatility if volatility > 0 else 0
        
        # Beta calculation
        if len(portfolio_returns) > 1 and len(benchmark_returns) > 1:
            covariance = np.cov(portfolio_returns.values, benchmark_returns.values)[0, 1]
            benchmark_variance = np.var(benchmark_returns.values)
            beta = covariance / benchmark_variance if benchmark_variance > 0 else 1
        else:
            beta = 1
        
        # Trade analysis
        trades_df = pd.DataFrame(self.completed_trades)
        if not trades_df.empty:
            winning_trades = len(trades_df[trades_df['final_pnl'] > 0])
            total_trades = len(trades_df)
            win_rate = winning_trades / total_trades * 100
            avg_win = trades_df[trades_df['final_pnl'] > 0]['final_pnl'].mean() if winning_trades > 0 else 0
            avg_loss = trades_df[trades_df['final_pnl'] < 0]['final_pnl'].mean() if (total_trades - winning_trades) > 0 else 0
            profit_factor = abs(avg_win * winning_trades / (avg_loss * (total_trades - winning_trades))) if avg_loss != 0 else 0
        else:
            win_rate = avg_win = avg_loss = profit_factor = total_trades = 0
        
        results = {
            'portfolio_series': portfolio_aligned,
            'benchmark_series': benchmark_aligned,
            'total_return': total_return,
            'annual_return': annual_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'benchmark_total_return': benchmark_total_return,
            'benchmark_annual_return': benchmark_annual_return,
            'benchmark_volatility': benchmark_volatility,
            'benchmark_sharpe': benchmark_sharpe,
            'benchmark_max_drawdown': benchmark_max_drawdown,
            'excess_return': excess_return,
            'information_ratio': information_ratio,
            'beta': beta,
            'total_trades': total_trades,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'trades_df': trades_df
        }
        
        return results
    
    def plot_results(self, results):
        """Plot backtest results with benchmark comparison"""
        fig = plt.figure(figsize=(20, 16))
        
        portfolio_series = results['portfolio_series']
        benchmark_series = results['benchmark_series']
        
        # 1. Portfolio vs Benchmark Value Over Time (Large plot)
        ax1 = plt.subplot(3, 3, (1, 4))
        ax1.plot(portfolio_series.index, portfolio_series, linewidth=2.5, color='#2E86AB', label='Statistical Arbitrage Strategy')
        ax1.plot(benchmark_series.index, benchmark_series, linewidth=2.5, color='#F18F01', label='S&P 500 Buy & Hold')
        ax1.set_title('Portfolio Value: Strategy vs S&P 500 Buy & Hold', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Portfolio Value ($)', fontsize=12)
        ax1.legend(fontsize=12)
        ax1.grid(True, alpha=0.3)
        ax1.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'${x:,.0f}'))
        
        # 2. Cumulative Returns Comparison
        ax2 = plt.subplot(3, 3, 2)
        portfolio_returns = (portfolio_series / self.initial_capital - 1) * 100
        benchmark_returns = (benchmark_series / self.initial_capital - 1) * 100
        ax2.plot(portfolio_returns.index, portfolio_returns, linewidth=2, color='#2E86AB', label='Strategy')
        ax2.plot(benchmark_returns.index, benchmark_returns, linewidth=2, color='#F18F01', label='S&P 500')
        ax2.set_title('Cumulative Returns (%)', fontsize=14, fontweight='bold')
        ax2.set_ylabel('Return (%)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Drawdown Comparison
        ax3 = plt.subplot(3, 3, 3)
        portfolio_cummax = portfolio_series.cummax()
        portfolio_drawdown = (portfolio_series - portfolio_cummax) / portfolio_cummax * 100
        benchmark_cummax = benchmark_series.cummax()
        benchmark_drawdown = (benchmark_series - benchmark_cummax) / benchmark_cummax * 100
        
        ax3.fill_between(portfolio_drawdown.index, portfolio_drawdown, 0, alpha=0.6, color='#2E86AB', label='Strategy')
        ax3.fill_between(benchmark_drawdown.index, benchmark_drawdown, 0, alpha=0.6, color='#F18F01', label='S&P')
        ax3.set_title('Drawdown Comparison (%)', fontsize=14, fontweight='bold')
        ax3.set_ylabel('Drawdown (%)')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Performance Metrics Table
        ax4 = plt.subplot(3, 3, 5)
        ax4.axis('tight')
        ax4.axis('off')
        
        metrics_data = [
            ['Metric', 'Strategy', 'S&P 500', 'Difference'],
            ['Total Return (%)', f'{results["total_return"]:.1f}%', f'{results["benchmark_total_return"]:.1f}%', f'{results["total_return"] - results["benchmark_total_return"]:.1f}%'],
            ['Annual Return (%)', f'{results["annual_return"]:.1f}%', f'{results["benchmark_annual_return"]:.1f}%', f'{results["excess_return"]:.1f}%'],
            ['Volatility (%)', f'{results["volatility"]:.1f}%', f'{results["benchmark_volatility"]:.1f}%', f'{results["volatility"] - results["benchmark_volatility"]:.1f}%'],
            ['Sharpe Ratio', f'{results["sharpe_ratio"]:.2f}', f'{results["benchmark_sharpe"]:.2f}', f'{results["sharpe_ratio"] - results["benchmark_sharpe"]:.2f}'],
            ['Max Drawdown (%)', f'{results["max_drawdown"]:.1f}%', f'{results["benchmark_max_drawdown"]:.1f}%', f'{results["max_drawdown"] - results["benchmark_max_drawdown"]:.1f}%'],
            ['Beta', f'{results["beta"]:.2f}', '1.00', f'{results["beta"] - 1:.2f}']
        ]
        
        table = ax4.table(cellText=metrics_data[1:], colLabels=metrics_data[0], cellLoc='center', loc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
       
        for i in range(len(metrics_data)):
            for j in range(len(metrics_data[0])):
                if i == 0:  # Header row
                    table[(i, j)].set_facecolor('#40466e')
                    table[(i, j)].set_text_props(weight='bold', color='white')
                else:
                    if j == 3:  # Difference column
                        val = float(metrics_data[i][j].replace('%', ''))
                        if val > 0:
                            table[(i, j)].set_facecolor('#c8e6c9')
                        elif val < 0:
                            table[(i, j)].set_facecolor('#ffcdd2')
                        else:
                            table[(i, j)].set_facecolor('#f5f5f5')
        
        ax4.set_title('Performance Comparison', fontsize=14, fontweight='bold')
        
        # 5. Monthly Returns Heatmap (Strategy)
        ax5 = plt.subplot(3, 3, 6)
        monthly_returns = portfolio_series.resample('M').last().pct_change().dropna() * 100
        
        if len(monthly_returns) > 0:
            monthly_data = pd.DataFrame({'returns': monthly_returns})
            monthly_data['year'] = monthly_data.index.year
            monthly_data['month'] = monthly_data.index.month
            
            pivot_data = monthly_data.pivot_table(values='returns', index='year', columns='month', fill_value=0)
            
            sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlGn', center=0, 
                       cbar_kws={'label': 'Monthly Return (%)'}, ax=ax5)
            ax5.set_title('Strategy Monthly Returns (%)', fontsize=12, fontweight='bold')
            ax5.set_xlabel('Month')
            ax5.set_ylabel('Year')
        
        # 6. Trade Distribution
        ax6 = plt.subplot(3, 3, 7)
        if not results['trades_df'].empty:
            pnl_data = results['trades_df']['final_pnl']
            ax6.hist(pnl_data, bins=20, alpha=0.7, color='#2E86AB', edgecolor='black')
            ax6.axvline(x=0, color='red', linestyle='--', linewidth=2)
            ax6.set_title('Trade P&L Distribution', fontsize=12, fontweight='bold')
            ax6.set_xlabel('P&L ($)')
            ax6.set_ylabel('Frequency')
            ax6.grid(True, alpha=0.3)
        
        # 7. Rolling Sharpe Ratio (60-day window)
        ax7 = plt.subplot(3, 3, 8)
        portfolio_daily_returns = portfolio_series.pct_change().dropna()
        rolling_sharpe = (portfolio_daily_returns.rolling(60).mean() * 252 - 5) / (portfolio_daily_returns.rolling(60).std() * np.sqrt(252))
        
        benchmark_daily_returns = benchmark_series.pct_change().dropna()
        benchmark_rolling_sharpe = (benchmark_daily_returns.rolling(60).mean() * 252 - 5) / (benchmark_daily_returns.rolling(60).std() * np.sqrt(252))
        
        ax7.plot(rolling_sharpe.index, rolling_sharpe, linewidth=2, color='#2E86AB', label='Strategy')
        ax7.plot(benchmark_rolling_sharpe.index, benchmark_rolling_sharpe, linewidth=2, color='#F18F01', label='S&P 500')
        ax7.axhline(y=0, color='black', linestyle='--', alpha=0.5)
        ax7.set_title('Rolling 60-Day Sharpe Ratio', fontsize=12, fontweight='bold')
        ax7.set_ylabel('Sharpe Ratio')
        ax7.legend()
        ax7.grid(True, alpha=0.3)
        
        # 8. Trade Statistics
        ax8 = plt.subplot(3, 3, 9)
        ax8.axis('tight')
        ax8.axis('off')
        
        if not results['trades_df'].empty:
            trade_stats = [
                ['Trade Statistics', 'Value'],
                ['Total Trades', f'{results["total_trades"]}'],
                ['Win Rate (%)', f'{results["win_rate"]:.1f}%'],
                ['Average Win ($)', f'${results["avg_win"]:.2f}'],
                ['Average Loss ($)', f'${results["avg_loss"]:.2f}'],
                ['Profit Factor', f'{results["profit_factor"]:.2f}'],
                ['Best Trade ($)', f'${results["trades_df"]["final_pnl"].max():.2f}'],
                ['Worst Trade ($)', f'${results["trades_df"]["final_pnl"].min():.2f}']
            ]
        else:
            trade_stats = [
                ['Trade Statistics', 'Value'],
                ['Total Trades', '0'],
                ['Win Rate (%)', '0.0%'],
                ['Average Win ($)', '$0.00'],
                ['Average Loss ($)', '$0.00'],
                ['Profit Factor', '0.00'],
                ['Best Trade ($)', '$0.00'],
                ['Worst Trade ($)', '$0.00']
            ]
        
        stats_table = ax8.table(cellText=trade_stats[1:], colLabels=trade_stats[0], cellLoc='center', loc='center')
        stats_table.auto_set_font_size(False)
        stats_table.set_fontsize(10)
        stats_table.scale(1, 1.5)
        
        # Style the trade stats table
        for i in range(len(trade_stats)):
            for j in range(len(trade_stats[0])):
                if i == 0:  # Header row
                    stats_table[(i, j)].set_facecolor('#40466e')
                    stats_table[(i, j)].set_text_props(weight='bold', color='white')
                else:
                    stats_table[(i, j)].set_facecolor('#f5f5f5')
        
        ax8.set_title('Trade Performance Statistics', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('statistical_arbitrage_backtest_results.png', dpi=300, bbox_inches='tight')
        plt.show()
    
    def generate_trade_log(self):
        """Generate detailed trade log"""
        if not self.completed_trades:
            print("No completed trades to log")
            return pd.DataFrame()
        
        trade_log = []
        for trade in self.completed_trades:
            trade_log.append({
                'Trade ID': trade['trade_id'],
                'Symbol': trade['symbol'],
                'Strategy': trade['strategy_type'],
                'Direction': trade['direction'],
                'Entry Date': trade['entry_date'].strftime('%Y-%m-%d'),
                'Exit Date': trade['exit_date'].strftime('%Y-%m-%d'),
                'Days Held': (trade['exit_date'] - trade['entry_date']).days,
                'Entry Price': f"${trade['opportunity']['current_price1']:.2f}",
                'Exit Price': f"${trade['exit_price']:.2f}",
                'Long Strike': f"${trade['long_strike']:.2f}",
                'Short Strike': f"${trade['short_strike']:.2f}",
                'Position Size': trade['position_size'],
                'Initial Cost': f"${trade['total_cost']:.2f}",
                'Final Value': f"${trade['final_value']:.2f}",
                'P&L': f"${trade['final_pnl']:.2f}",
                'Return %': f"{(trade['final_pnl'] / trade['total_cost'] * 100):.1f}%",
                'Exit Reason': trade['exit_reason'],
                'Z-Score': f"{trade['z_score']:.2f}",
                'Correlation': f"{trade['opportunity']['correlation']:.3f}"
            })
        
        trade_df = pd.DataFrame(trade_log)
        
        # Save to CSV
        trade_df.to_csv('trade_log.csv', index=False)
        print(f"Trade log saved to 'trade_log.csv' with {len(trade_df)} trades")
        
        return trade_df
    
    def print_summary_report(self, results):
        """Print comprehensive summary report"""
        print("\n" + "="*80)
        print("STATISTICAL ARBITRAGE OPTIONS STRATEGY BACKTEST RESULTS")
        print("="*80)
        
        print(f"\nPORTFOLIO PERFORMANCE:")
        print(f"Initial Capital: ${self.initial_capital:,.2f}")
        print(f"Final Portfolio Value: ${results['portfolio_series'].iloc[-1]:,.2f}")
        print(f"Total Return: {results['total_return']:.2f}%")
        print(f"Annualized Return: {results['annual_return']:.2f}%")
        print(f"Volatility: {results['volatility']:.2f}%")
        print(f"Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"Maximum Drawdown: {results['max_drawdown']:.2f}%")
        print(f"Beta: {results['beta']:.2f}")
        
        print(f"\nBENCHMARK (S&P 500) PERFORMANCE:")
        print(f"Total Return: {results['benchmark_total_return']:.2f}%")
        print(f"Annualized Return: {results['benchmark_annual_return']:.2f}%")
        print(f"Volatility: {results['benchmark_volatility']:.2f}%")
        print(f"Sharpe Ratio: {results['benchmark_sharpe']:.2f}")
        print(f"Maximum Drawdown: {results['benchmark_max_drawdown']:.2f}%")
        
        print(f"\nOUTPERFORMANCE METRICS:")
        print(f"Excess Return: {results['excess_return']:.2f}%")
        print(f"Information Ratio: {results['information_ratio']:.2f}")
        print(f"Alpha (vs S&P 500): {results['annual_return'] - results['beta'] * results['benchmark_annual_return']:.2f}%")
        
        print(f"\nTRADE STATISTICS:")
        print(f"Total Trades: {results['total_trades']}")
        print(f"Win Rate: {results['win_rate']:.1f}%")
        print(f"Average Win: ${results['avg_win']:.2f}")
        print(f"Average Loss: ${results['avg_loss']:.2f}")
        print(f"Profit Factor: {results['profit_factor']:.2f}")
        
        if not results['trades_df'].empty:
            print(f"Best Trade: ${results['trades_df']['final_pnl'].max():.2f}")
            print(f"Worst Trade: ${results['trades_df']['final_pnl'].min():.2f}")
            
            # Monthly trade count
            trades_by_month = results['trades_df'].groupby(results['trades_df']['entry_date'].dt.to_period('M')).size()
            print(f"Average Trades per Month: {trades_by_month.mean():.1f}")
        
        print("\n" + "="*80)


def main():
    """Main function to run the backtest"""
    # Initialize backtester
    backtester = StatArbOptionsBacktester(
        initial_capital=100000,
        max_position_size=0.05,
        max_concurrent_trades=10
    )
    
    # Run backtest with updated time period
    success = backtester.run_backtest(
        start_date='2021-08-01',
        end_date='2025-08-01'
    )
    
    if not success:
        print("Backtest failed")
        return
    
    # Analyze results
    results = backtester.analyze_results()
    if results is None:
        print("No results to analyze")
        return
    
    # Generate comprehensive output
    backtester.print_summary_report(results)
    backtester.plot_results(results)
    trade_log = backtester.generate_trade_log()
    
    # Display sample of trade log
    if not trade_log.empty:
        print(f"\nSAMPLE TRADE LOG (showing first 10 trades):")
        print("="*80)
        print(trade_log.head(10).to_string(index=False))


if __name__ == "__main__":
    main()

def run_full_backtest():
    """Run your blackarb.py strategy against S&P 500 with full analysis"""
    
    print("="*80)
    print(" STATISTICAL ARBITRAGE STRATEGY BACKTEST")
    print("Testing your blackarb.py strategy vs S&P 500 Buy & Hold")
    print("="*80)
    
    # CUSTOMIZABLE PARAMETERS
    INITIAL_CAPITAL = 100000        # Starting capital
    MAX_POSITION_SIZE = 0.05        # 5% max position per trade
    MAX_CONCURRENT_TRADES = 10      # Maximum simultaneous trades
    START_DATE = '2021-08-01'       # Backtest start date
    END_DATE = '2025-08-01'         # Backtest end date
    
    print(f" Configuration:")
    print(f"   • Initial Capital: ${INITIAL_CAPITAL:,}")
    print(f"   • Max Position Size: {MAX_POSITION_SIZE*100}% per trade")
    print(f"   • Max Concurrent Trades: {MAX_CONCURRENT_TRADES}")
    print(f"   • Test Period: {START_DATE} to {END_DATE}")
    print("="*80)
    
    # Initialize backtester
    backtester = StatArbOptionsBacktester(
        initial_capital=INITIAL_CAPITAL,
        max_position_size=MAX_POSITION_SIZE,
        max_concurrent_trades=MAX_CONCURRENT_TRADES
    )
    
    # Run backtest
    print("🔄 Starting backtest... (this may take 3-5 minutes)")
    success = backtester.run_backtest(
        start_date=START_DATE,
        end_date=END_DATE
    )
    
    if not success:
        print("Backtest failed")
        return None
    
    # Analyze results
    results = backtester.analyze_results()
    if results is None:
        print(" No results to analyze")
        return None
    
    # Print comprehensive summary
    backtester.print_summary_report(results)
    
    # Generate detailed trade log
    trade_log = backtester.generate_trade_log()
    
    # Create visualization
    backtester.plot_results(results)
    
    # Print key takeaways
    print("\n" + "="*80)
    print(" KEY TAKEAWAYS:")
    print("="*80)
    
    strategy_return = results['total_return']
    benchmark_return = results['benchmark_total_return']
    
    if strategy_return > benchmark_return:
        print(f" Your strategy OUTPERFORMED the S&P 500!")
        print(f"   Strategy: {strategy_return:.1f}% vs S&P 500: {benchmark_return:.1f}%")
        print(f"    Excess return: +{strategy_return - benchmark_return:.1f}%")
    else:
        print(f" Your strategy underperformed the S&P 500")
        print(f"    Strategy: {strategy_return:.1f}% vs S&P 500: {benchmark_return:.1f}%")
        print(f"   Shortfall: {strategy_return - benchmark_return:.1f}%")
    
    print(f"\n💼 Trade Performance:")
    print(f"   • Total trades executed: {results['total_trades']}")
    print(f"   • Win rate: {results['win_rate']:.1f}%")
    print(f"   • Profit factor: {results['profit_factor']:.2f}")
    print(f"   • Best trade: ${results['trades_df']['final_pnl'].max():.2f}" if not results['trades_df'].empty else "   • No trades executed")
    print(f"   • Worst trade: ${results['trades_df']['final_pnl'].min():.2f}" if not results['trades_df'].empty else "")
    
    print(f"\n Risk Metrics:")
    print(f"   • Strategy volatility: {results['volatility']:.1f}%")
    print(f"   • S&P 500 volatility: {results['benchmark_volatility']:.1f}%")
    print(f"   • Max drawdown: {results['max_drawdown']:.1f}%")
    print(f"   • Sharpe ratio: {results['sharpe_ratio']:.2f}")
    print(f"   • Beta: {results['beta']:.2f}")
    
    print("\n" + "="*80)
    
    return results, trade_log
