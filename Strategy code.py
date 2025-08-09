import numpy as np
import pandas as pd
import yfinance as yf
from scipy.stats import norm
from datetime import datetime, timedelta
import warnings
import requests
from bs4 import BeautifulSoup
from sklearn.linear_model import LinearRegression
import time
import concurrent.futures
from threading import Lock
warnings.filterwarnings('ignore')

class sp500arbscan:
    def __init__(self, lookback_period=252, deviation_threshold_min=2.25, deviation_threshold_max=2.5):
        self.lookback_period = lookback_period
        self.deviation_threshold_min = deviation_threshold_min
        self.deviation_threshold_max = deviation_threshold_max
        self.risk_free_rate = 0.05
        self.correlated_pairs = []
        self.current_opportunities = []
        self.sp500_symbols = []
        self.price_data = None
        self.lock = Lock()
        
    def get_sp500_symbols(self):

        print("Fetching S&P 500 symbols...")
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        tables = pd.read_html(url)
        df = tables[0]  # find table 
        symbols = df['Symbol'].tolist()

        
        self.sp500_symbols = [symbol.replace('.', '-') for symbol in symbols]# Replace '.' with '-' 


    def download_in_batches(self, symbols, period='1y', batch_size=50):
        all_data = {}
        for i in range(0, len(symbols), batch_size):
            batch = symbols[i:i + batch_size]
            try:
                data = yf.download(batch, period=period, group_by='ticker', threads=False, progress=False)
                for symbol in batch:
                    if isinstance(data.columns, pd.MultiIndex) and symbol in data.columns.get_level_values(0):
                        all_data[symbol] = data[symbol]['Close']
                    elif symbol in data.columns:
                        all_data[symbol] = data[symbol]['Close']  # fallback single-index case
            except Exception as e:
                print(f"Batch {batch} failed: {e}")
        return all_data

    
    
    def filter_movers(self, period='1y', pct_change=0.4):
        all_data = self.download_in_batches(self.sp500_symbols, period=period)

        filter_symbols = []
        filtered_prices={}

        counter = 0

        for symbol, df in all_data.items():
            try:
                prices = df.dropna()
                if len(prices) < 2:
                    continue
                pct_move = abs((prices[-1] - prices[0]) / prices[0])
                if pct_move >= pct_change:
                    filter_symbols.append(symbol)
                    filtered_prices[symbol] = prices
             
                    counter += 1
            except Exception as e:
                print(f"Error with {symbol}: {e}")
                continue

        self.sp500_symbols = filter_symbols
        self.price_data = filtered_prices
        print(f"Filtered to {counter} movers.")
        return counter
# aug 02 /2025 - i sorted out through the stocks in the snp500 to find stocks that have moved over 30% in either direction in a year and stored them in sp500_symbols



    
   

    def find_correlated_pairs_optimized(self, min_correlation=0.87, max_pairs=100):
        if self.price_data is None:
            print("No price data available. Please fetch data first.")
            return []

        # Combine all Series into one DataFrame
        price_df = pd.DataFrame(self.price_data)

        symbols = list(price_df.columns)
        total_possible_pairs = len(symbols) * (len(symbols) - 1) // 2
        print(f"Finding correlated pairs from {len(symbols)} symbols ({total_possible_pairs} possible pairs)...")
        print(f"Looking for correlations >= {min_correlation}")

        # Calculate correlation matrix
        print("Calculating correlation matrix...")
        correlations = price_df.corr()
    
        pairs = []
        pairs_checked = 0

        for i in range(len(symbols)):
            for j in range(i + 1, len(symbols)):
                pairs_checked += 1
                symbol1, symbol2 = symbols[i], symbols[j]

                correlation = correlations.loc[symbol1, symbol2]
    
                if pd.isna(correlation):
                    continue

                if abs(correlation) >= min_correlation:
                    prices1 = price_df[symbol1]
                    prices2 = price_df[symbol2]

                    hedge_ratio = self.calculate_hedge_ratio(prices1, prices2)
                    spread = prices1 - hedge_ratio * prices2
                    spread_mean = spread.mean()
                    spread_std = spread.std()

                    if spread_std > 0:
                        current_z_score = (spread.iloc[-1] - spread_mean) / spread_std

                        pairs.append({
                            'asset1': symbol1,
                            'asset2': symbol2,
                            'correlation': correlation,
                            'hedge_ratio': hedge_ratio,
                            'spread_mean': spread_mean,
                            'spread_std': spread_std,
                            'current_spread': spread.iloc[-1],
                            'z_score': current_z_score,
                            'current_price1': prices1.iloc[-1],
                            'current_price2': prices2.iloc[-1]
                        })

                if pairs_checked % 1000 == 0:
                    print(f"Checked {pairs_checked:,}/{total_possible_pairs:,} pairs, found {len(pairs)} correlated pairs")

                if len(pairs) >= max_pairs:
                    print(f"Reached maximum pairs limit ({max_pairs}), stopping search")
                    break
            if len(pairs) >= max_pairs:
                break

        print(f"Completed: Checked {pairs_checked:,} pairs, found {len(pairs)} with |correlation| >= {min_correlation}")

        pairs = sorted(pairs, key=lambda x: abs(x['correlation']), reverse=True)
        self.correlated_pairs = pairs
        return pairs
        
    def calculate_hedge_ratio(self, asset1_prices, asset2_prices):
        #Calculate hedge ratio using linear regression
        
        data = pd.DataFrame({'asset1': asset1_prices, 'asset2': asset2_prices}).dropna() #dropna
        
        if len(data) < 10:  # Not enough data points
            return 1.0
        
        X = data['asset2'].values.reshape(-1, 1)
        y = data['asset1'].values
        
        model = LinearRegression()
        model.fit(X, y)
        
        return model.coef_[0]



    
    def detect_deviation_opportunities(self, pairs):
        #Detect pairs that have deviated beyond threshold
        opportunities = []
        
        print(f"Scanning {len(pairs)} pairs for deviation opportunities...")
        print(f"Looking for z-score between {self.deviation_threshold_min} and {self.deviation_threshold_max}")
        
        for pair in pairs:
            z_score = pair['z_score']
            abs_z_score = abs(z_score)
            
            if self.deviation_threshold_min <= abs_z_score <= self.deviation_threshold_max:
                # Determine which asset is deviated and direction
                if z_score > 0:
                    # Asset1 is overpriced relative to hedged Asset2
                    deviated_asset = pair['asset1']
                    reference_asset = pair['asset2']
                    direction = 'short'  # Short the overpriced asset
                    deviation_type = 'overpriced'
                else:
                    # Asset1 is underpriced relative to hedged Asset2
                    deviated_asset = pair['asset1']
                    reference_asset = pair['asset2']
                    direction = 'long'  # Long the underpriced asset
                    deviation_type = 'underpriced'
                
                opportunities.append({
                    'pair': pair,
                    'deviated_asset': deviated_asset,
                    'reference_asset': reference_asset,
                    'direction': direction,
                    'deviation_type': deviation_type,
                    'z_score': z_score,
                    'abs_z_score': abs_z_score,
                    'confidence': min(abs_z_score / 3.0, 1.0),  # Confidence based on z-score
                    'current_price': pair['current_price1'] if deviated_asset == pair['asset1'] else pair['current_price2']
                })
        
        print(f"Found {len(opportunities)} deviation opportunities")
        self.current_opportunities = opportunities
        return opportunities
    
    def black_scholes_call(self, S, K, T, r, sigma):
        #Black-Scholes call option pricing
        if T <= 0:
            return max(S - K, 0)
        
        try:
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T)) 
            d2 = d1 - sigma*np.sqrt(T) #
            
            call_price = S*norm.cdf(d1) - K*np.exp(-r*T)*norm.cdf(d2)
            return max(call_price, 0)
        except:
            return max(S - K, 0)
    
    def black_scholes_put(self, S, K, T, r, sigma):
        #Black-Scholes put option pricing
        if T <= 0:
            return max(K - S, 0)
        
        try:
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            
            put_price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
            return max(put_price, 0)
        except:
            return max(K - S, 0)
    
    def calculate_greeks(self, S, K, T, r, sigma, option_type='call'):
        #Calculate option Greeks
        if T <= 0:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
        
        try:
            d1 = (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*np.sqrt(T))
            d2 = d1 - sigma*np.sqrt(T)
            
            # Delta
            if option_type == 'call':
                delta = norm.cdf(d1)
            else:
                delta = norm.cdf(d1) - 1
            
            # Gamma
            gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
            
            # Theta
            if option_type == 'call':
                theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                        - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
            else:
                theta = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) 
                        + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
            
            # Vega
            vega = S * norm.pdf(d1) * np.sqrt(T) / 100
            
            # Rho
            if option_type == 'call':
                rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
            else:
                rho = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
            
            return {
                'delta': delta,
                'gamma': gamma,
                'theta': theta,
                'vega': vega,
                'rho': rho
            }
        except:
            return {'delta': 0, 'gamma': 0, 'theta': 0, 'vega': 0, 'rho': 0}
    
    def calculate_volatility(self, symbol, period_days=252):
        """Calculate historical volatility for a symbol"""
        try:
            if self.price_data is not None and symbol in self.price_data.columns:
                prices = self.price_data[symbol].dropna()
                if len(prices) > period_days:
                    prices = prices.tail(period_days)
                returns = prices.pct_change().dropna()
                volatility = returns.std() * np.sqrt(252)
                return max(volatility, 0.1)  # Minimum volatility of 10%
            else:
                # Fallback: fetch individual data
                data = yf.download(symbol, period='1y', progress=False)['Adj Close']
                returns = data.pct_change().dropna()
                volatility = returns.std() * np.sqrt(252)
                return max(volatility, 0.1)
        except:
            return 0.3  # Default volatility of 30%
    
    def determine_option_strats(self, opportunity):
        deviated_symbol = opportunity['deviated_asset']
        current_price = opportunity['current_price']
        direction = opportunity['direction']
        abs_z_score = abs(opportunity['z_score'])
        confidence = opportunity['confidence']

        # Calculate volatility
        volatility = self.calculate_volatility(deviated_symbol)

        # Set expiration days based on z-score
        if abs_z_score > 2.42:
            days_to_expiration = 21
        elif abs_z_score > 2.3:
            days_to_expiration = 35
        else:
            days_to_expiration = 45
        T = days_to_expiration / 365

        # Calculate strike prices
        if direction == 'short':  # Bear Put Spread
            long_strike = round(current_price)
            short_strike = round(current_price * 0.95)
        else:  # Bull Call Spread
            long_strike = round(current_price)
            short_strike = round(current_price * 1.05)

        # Calculate option prices and greeks for both strikes
        long_option_price = None
        short_option_price = None
        long_option_greeks = None
        short_option_greeks = None
        strategy_type = None

        if direction == 'short':
            strategy_type = "Bear Put Spread"
            long_option_price = self.black_scholes_put(current_price, long_strike, T, self.risk_free_rate, volatility)
            short_option_price = self.black_scholes_put(current_price, short_strike, T, self.risk_free_rate, volatility)
            long_option_greeks = self.calculate_greeks(current_price, long_strike, T, self.risk_free_rate, volatility, 'put')
            short_option_greeks = self.calculate_greeks(current_price, short_strike, T, self.risk_free_rate, volatility, 'put')
            max_profit = long_strike - short_strike - (long_option_price - short_option_price)
        else:
            strategy_type = "Bull Call Spread"
            long_option_price = self.black_scholes_call(current_price, long_strike, T, self.risk_free_rate, volatility)
            short_option_price = self.black_scholes_call(current_price, short_strike, T, self.risk_free_rate, volatility)
            long_option_greeks = self.calculate_greeks(current_price, long_strike, T, self.risk_free_rate, volatility, 'call')
            short_option_greeks = self.calculate_greeks(current_price, short_strike, T, self.risk_free_rate, volatility, 'call')
            max_profit = short_strike - long_strike - (long_option_price - short_option_price)

        net_cost = long_option_price - short_option_price
        max_loss = abs(net_cost)

        net_delta = long_option_greeks['delta'] - short_option_greeks['delta']
        net_gamma = long_option_greeks['gamma'] - short_option_greeks['gamma']
        net_theta = long_option_greeks['theta'] - short_option_greeks['theta']
        net_vega = long_option_greeks['vega'] - short_option_greeks['vega']

        expected_move = current_price * volatility * np.sqrt(T)

        strategy_summary = {
            'symbol': deviated_symbol,
            'strategy_type': strategy_type,
            'direction': direction,
            'long_strike': long_strike,
            'short_strike': short_strike,
            'long_option_price': long_option_price,
            'short_option_price': short_option_price,
            'net_cost': net_cost,
            'max_profit': max_profit,
            'max_loss': max_loss,
            'risk_reward_ratio': abs(max_profit) / max_loss if max_loss > 0 else None,
            'breakeven_probability': confidence,
            'days_to_expiration': days_to_expiration,
            'expected_move': expected_move,
            'volatility_used': volatility,
            'net_greeks': {
            'delta': net_delta,
            'gamma': net_gamma,
            'theta': net_theta,
            'vega': net_vega
            },
            'profit_probability': confidence,
            'z_score': opportunity['z_score'],
            'current_price': current_price
        }

        return strategy_summary
    
    def run_full_scan(self, min_correlation=0.87, max_pairs=100):
        #Run the complete scan process
        print("Starting S&P 500 Statistical Arbitrage Scan...")
        print("="*60)
        
        # Step 1: Get S&P 500 symbols
        symbols = self.get_sp500_symbols()
        
        # Step 2: Fetch price data
        price_data = self.fetch_data_batch(symbols)
        
        if price_data is None or price_data.empty:
            print("Failed to fetch price data. Exiting.")
            return []
        
        # Step 3: Find correlated pairs
        pairs = self.find_correlated_pairs_optimized(min_correlation, max_pairs)
        
        if not pairs:
            print("No correlated pairs found. Lower the correlation threshold.")
            return []
        
        # Step 4: Detect deviation opportunities
        opportunities = self.detect_deviation_opportunities(pairs)
        
        if not opportunities:
            print("No deviation opportunities found, widen range.")
            return []
        
        # Step 5: Generate options strategies
        print(f"Generating options strategies for {len(opportunities)} opportunities...")
        strategies = []
        
        for i, opportunity in enumerate(opportunities):
            try:
                strategy = self.determine_options_strats(opportunity)
                strategies.append(strategy)
                print(f"Generated strategy {i+1}/{len(opportunities)}")
            except Exception as e:
                print(f"Error generating strategy for {opportunity['deviated_asset']}: {e}")
        
        print(f"Successfully generated {len(strategies)} strategies")
        return strategies
    
   
