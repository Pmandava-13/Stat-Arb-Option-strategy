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
    print("STATISTICAL ARBITRAGE STRATEGY BACKTEST")
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
    print(" Starting backtest... (this may take 3-5 minutes)")
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
        print(f"Your strategy OUTPERFORMED the S&P 500!")
        print(f"Strategy: {strategy_return:.1f}% vs S&P 500: {benchmark_return:.1f}%")
        print(f" Excess return: +{strategy_return - benchmark_return:.1f}%")
    else:
        print(f"Your strategy underperformed the S&P 500")
        print(f" Strategy: {strategy_return:.1f}% vs S&P 500: {benchmark_return:.1f}%")
        print(f" Shortfall: {strategy_return - benchmark_return:.1f}%")
    
    print(f"\n Trade Performance:")
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


results, trade_log = run_full_backtest()
