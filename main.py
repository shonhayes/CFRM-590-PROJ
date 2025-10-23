import yfinance as yf
import pandas as pd
import numpy as np
from portfolio_optimization import mean_variance_optimization
import pandas_datareader as pdr
from datetime import datetime
import matplotlib.pyplot as plt

# ETF tickers and labels (excluding AGGG and IEMG)
tickers = {
    "MSCI_ACWI_IMI": "ACWI",    # ACWI ETF as proxy
    "MSCI_EAFE": "EFA",
    "MSCI_US_AllCap": "VTI",
    "MSCI_US_REIT": "VNQ",
    "Bloomberg_US_Agg": "AGG",
    "ICE_US_TBill_3M": "BIL",
    "ICE_US_HighYield": "HYG",
    "SPGSCI_TR": "GSG"
}

expense_ratios = {
    "ACWI": 0.0032,   # iShares MSCI ACWI ETF (0.32%)
    "EFA": 0.0032,    # iShares MSCI EAFE ETF (0.32%)
    "VTI": 0.0003,    # Vanguard Total Stock Market ETF (0.03%)
    "VNQ": 0.0013,    # Vanguard Real Estate ETF (0.13%)
    "AGG": 0.0003,    # iShares Core U.S. Aggregate Bond ETF (0.03%)
    "BIL": 0.0014,    # SPDR Bloomberg 1-3 Month T-Bill ETF (~0.14%)
    "HYG": 0.0049,    # iShares iBoxx $ High Yield Corporate Bond ETF (0.49%)
    "GSG": 0.0075     # iShares S&P GSCI Commodity-Indexed Trust (0.75%)
}

trading_costs_bps = {
    "ACWI": 3,
    "EFA": 3,
    "VTI": 2,
    "VNQ": 5,
    "AGG": 2,
    "BIL": 2,
    "HYG": 7,
    "GSG": 15
}

# Asset order
assets = ["ACWI", "EFA", "VTI", "VNQ", "AGG", "BIL", "HYG", "GSG"]

# Convert to decimal cost per $ traded
trading_costs_array = np.array([(1+trading_costs_bps[t] / 10000)**(1/252)-1 for t in assets])

# Download price history
price_data = yf.download(list(tickers.values()), start="2000-01-01")

# Handle different yfinance data structures
if isinstance(price_data.columns, pd.MultiIndex):
    # Multi-level columns (newer yfinance versions)
    # Check if Adj Close exists, otherwise use Close
    if ('Adj Close', list(tickers.values())[0]) in price_data.columns:
        price_data = price_data['Adj Close']
        print("a")
    elif ('Close', list(tickers.values())[0]) in price_data.columns:
        price_data = price_data['Close']
        print("b")
    else:
        # Try to get adjusted close data
        try:
            price_data = price_data.xs('Adj Close', level=0, axis=1)
            print("c")
        except KeyError:
            price_data = price_data.xs('Close', level=0, axis=1)
else:
    # Single level columns (older versions)
    if "Adj Close" in price_data.columns:
        price_data = price_data["Adj Close"]
    else:
        price_data = price_data["Close"]

# Rename columns to match index names
price_data = price_data.rename(columns={v: k for k, v in tickers.items()})

# Compute daily returns
returns = price_data.pct_change().dropna()

# Convert annual expense ratios to daily expense ratios and subtract from returns
# Daily expense ratio = (1 + annual_expense_ratio)^(1/252) - 1
daily_expense_ratios = {}
for ticker, annual_expense in expense_ratios.items():
    daily_expense_ratios[ticker] = (1 + annual_expense) ** (1/252) - 1

# Subtract daily expense ratios from returns to get net returns
for ticker in expense_ratios.keys():
    if ticker in returns.columns:
        returns[ticker] = returns[ticker] - daily_expense_ratios[ticker]


# Find the start date for portfolio calculation (5 years after the most recent series start)
# Get the earliest non-null date for each series
series_start_dates = returns.apply(lambda x: x.first_valid_index())
most_recent_start = series_start_dates.max()

# Calculate start date (5 years after the most recent series start)
portfolio_start_date = most_recent_start + pd.DateOffset(years=5)

# Filter returns data to start from the calculated date
portfolio_returns_data = returns.loc[portfolio_start_date:]

print(f"Portfolio returns data shape: {portfolio_returns_data.shape}")
print(f"Date range: {portfolio_returns_data.index[0]} to {portfolio_returns_data.index[-1]}")

# Calculate 60% ACWI + 40% US AGG portfolio returns
acwi_weight = 0.60
agg_weight = 0.40

portfolio_returns = (acwi_weight * portfolio_returns_data['MSCI_ACWI_IMI'] + 
                    agg_weight * portfolio_returns_data['Bloomberg_US_Agg'])

# Find the earliest date where all series have 5 years of data
earliest_start = series_start_dates.max()  # Most recent start among all series
five_years_later = earliest_start + pd.DateOffset(years=5)

latest_end = pd.Timestamp("2024-12-31")

# Generate year-end dates for rolling windows
# Start from the first year-end after we have 5 years of data
start_year = five_years_later.year
end_year = latest_end.year

# Initialize DataFrames to store all portfolio returns
benchmark_returns_df = pd.DataFrame()
optimal_returns_df = pd.DataFrame()

# Initialize DataFrame to store optimal portfolio weights for each year
optimal_weights_df = pd.DataFrame()

# Rolling 5-year window loop
count = 0
for year in range(start_year, end_year + 1):
    # Define the 5-year window (5 years ending at year-end)
    window_end = pd.Timestamp(f"{year}-12-31")
    window_start = window_end - pd.DateOffset(years=5)

    # Extract the 5-year window of returns
    window_returns = returns.loc[window_start:window_end]
    
    # Check if we have complete data for all series in this window
    # (all series should have data for the entire 5-year period)
    has_complete_data = True
    for col in window_returns.columns:
        if window_returns[col].isna().any():
            has_complete_data = False
            break
     
    # Calculate portfolio returns for this window (60% ACWI + 40% AGG)
    window_portfolio_returns = (acwi_weight * window_returns['MSCI_ACWI_IMI'] + 
                                agg_weight * window_returns['Bloomberg_US_Agg'])
    
    geometric_daily_return_bench = (1 + window_portfolio_returns).prod() ** (1 / len(window_portfolio_returns)) - 1
    
    # Calculate mean-variance optimal portfolio weights using geometric daily return as target

    if count == 0:
        optimal_weights, portfolio_return, portfolio_volatility = mean_variance_optimization(
            returns_df=window_returns,
            target_return=geometric_daily_return_bench+0.00005,
            allow_short_selling=False,  # No short selling
            max_weight=1.0,  # No maximum weight constraint
            min_weight=0.0,
            trading_cost_per_dollar=trading_costs_array   # No minimum weight constraint
        )
    else:
        optimal_weights, portfolio_return, portfolio_volatility = mean_variance_optimization(
            returns_df=window_returns,
            target_return=geometric_daily_return_bench+(1 + 0.02)**(1/252)-1,
            allow_short_selling=False,  # No short selling
            max_weight=1.0,  # No maximum weight constraint
            min_weight=0.0, 
            prev_weights=optimal_weights,
            trading_cost_per_dollar=trading_costs_array   # No minimum weight constraint
                # No minimum weight constraint
        )
    

    count += 1
    
    # Store optimal weights for this year
    weights_dict = {}
    for i, asset in enumerate(window_returns.columns):
        weights_dict[asset] = np.abs(optimal_weights[i])  # Use absolute values for display
    
    # Add to weights DataFrame
    optimal_weights_df[year] = pd.Series(weights_dict)
    
    # Calculate out-of-sample returns for the year following the window
    next_year_start = window_end + pd.DateOffset(days=1)
    next_year_end = min(window_end + pd.DateOffset(years=1), returns.index[-1])
    
    # Get the out-of-sample data
    out_of_sample_returns = returns.loc[next_year_start:next_year_end]
    # Calculate benchmark portfolio returns out-of-sample (60% ACWI + 40% AGG)
    benchmark_out_of_sample = (acwi_weight * out_of_sample_returns['MSCI_ACWI_IMI'] + 
                                agg_weight * out_of_sample_returns['Bloomberg_US_Agg'])
    
    # Calculate optimal portfolio returns out-of-sample (keep as pandas Series with date index)
    optimal_out_of_sample = pd.Series(
        np.sum(optimal_weights * out_of_sample_returns.values, axis=1),
        index=out_of_sample_returns.index,
        name='optimal_returns'
    )
    
    # Calculate geometric daily returns for out-of-sample period
    benchmark_geometric_out_of_sample = (1 + benchmark_out_of_sample).prod() ** (1 / len(benchmark_out_of_sample)) - 1
    optimal_geometric_out_of_sample = (1 + optimal_out_of_sample).prod() ** (1 / len(optimal_out_of_sample)) - 1
    
    # Store returns in DataFrames (concatenate to existing data)
    if benchmark_returns_df.empty:
        benchmark_returns_df = pd.DataFrame(benchmark_out_of_sample, columns=['benchmark_returns'])
    else:
        benchmark_returns_df = pd.concat([benchmark_returns_df, pd.DataFrame(benchmark_out_of_sample, columns=['benchmark_returns'])])
    
    if optimal_returns_df.empty:
        optimal_returns_df = pd.DataFrame(optimal_out_of_sample, columns=['optimal_returns'])
    else:
        optimal_returns_df = pd.concat([optimal_returns_df, pd.DataFrame(optimal_out_of_sample, columns=['optimal_returns'])])


# Get the date range from our portfolio data
start_date = benchmark_returns_df.index[0].strftime('%Y-%m-%d')
end_date = benchmark_returns_df.index[-1].strftime('%Y-%m-%d')

# Fetch DTB3 (3-Month Treasury Bill) from FRED
risk_free_rate = pdr.get_data_fred('DTB3', start=start_date, end=end_date)
risk_free_rate = risk_free_rate['DTB3'].dropna()

# Convert annual rate to daily rate
risk_free_daily = (1 + risk_free_rate / 100) ** (1/252) - 1


# Align risk-free rate with portfolio data
risk_free_aligned = risk_free_daily.reindex(benchmark_returns_df.index, method='ffill').fillna(0)

# Calculate metrics for both portfolios
def calculate_metrics(returns_series, risk_free_series, name):
    """Calculate risk and return metrics for a portfolio"""
    
    # Annualized return
    annual_return = returns_series.mean() * 252
    
    # Annualized standard deviation
    annual_std = returns_series.std() * np.sqrt(252)
    
    # Sharpe ratio
    excess_returns = returns_series - risk_free_series
    sharpe_ratio = (excess_returns.mean() * 252) / (excess_returns.std() * np.sqrt(252))
    
    # 97.5% CVaR (Conditional Value at Risk)
    # CVaR is the expected loss given that the loss exceeds the VaR threshold
    var_975 = np.percentile(returns_series, 2.5)  # 2.5% VaR (97.5% confidence)
    cvar_975 = returns_series[returns_series <= var_975].mean()
    
    return {
        'Portfolio': name,
        'Annual Return': annual_return,
        'Annual Std Dev': annual_std,
        'Sharpe Ratio': sharpe_ratio,
        '97.5% CVaR': cvar_975
    }

# Calculate metrics for both portfolios
benchmark_metrics = calculate_metrics(
    benchmark_returns_df.iloc[:, 0], 
    risk_free_aligned, 
    'Benchmark (60% ACWI, 40% AGG)'
)

optimal_metrics = calculate_metrics(
    optimal_returns_df.iloc[:, 0], 
    risk_free_aligned, 
    'Optimal Portfolio   '
)

# Create comparison table
comparison_data = [benchmark_metrics, optimal_metrics]
comparison_df = pd.DataFrame(comparison_data)

print("="*80)
print(f"Analysis Period: {benchmark_returns_df.index[0].strftime('%Y-%m-%d')} to {benchmark_returns_df.index[-1].strftime('%Y-%m-%d')}")
print(f"Total Observations: {len(benchmark_returns_df)}")
print(f"Risk-Free Rate Source: FRED DTB3 (3-Month Treasury Bill)")
print()

# Display formatted table
print(f"{'Portfolio':<30} {'Annual Return':<15} {'Annual Std Dev':<15} {'Sharpe Ratio':<12} {'97.5% CVaR':<12}")
print("-" * 85)

for _, row in comparison_df.iterrows():
    print(f"{row['Portfolio']:<33} "
          f"{row['Annual Return']:<15.2%} "
          f"{row['Annual Std Dev']:<15.2%} "
          f"{row['Sharpe Ratio']:<12.2f} "
          f"{row['97.5% CVaR']:<12.2%}")

print("-" * 85)

# Transpose the DataFrame so years are on x-axis and assets are series
weights_for_plot = optimal_weights_df.T

# Define colors for each asset
colors = {
    'MSCI_ACWI_IMI': '#1f77b4',      # Blue
    'MSCI_EAFE': '#ff7f0e',          # Orange  
    'MSCI_US_AllCap': '#2ca02c',     # Green
    'MSCI_US_REIT': '#d62728',       # Red
    'Bloomberg_US_Agg': '#9467bd',   # Purple
    'ICE_US_TBill_3M': '#8c564b',    # Brown
    'ICE_US_HighYield': '#e377c2',   # Pink
    'SPGSCI_TR': '#7f7f7f'           # Gray
}

# Create the stackplot
plt.figure(figsize=(14, 8))

# Get the color list in the same order as the DataFrame columns
asset_colors = [colors.get(asset, '#000000') for asset in weights_for_plot.columns]

# Create the stackplot with proper x-axis handling
years = weights_for_plot.index.values

plt.stackplot(years, 
                *[weights_for_plot[asset] for asset in weights_for_plot.columns],
                labels=weights_for_plot.columns,
                colors=asset_colors,
                alpha=0.8)

# Customize the plot
plt.title('Optimal Portfolio Composition Over Time', fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Portfolio Weight', fontsize=12)
plt.ylim(0, 1)

# Force the plot to use the full width with explicit axis control
ax = plt.gca()

# Set x-axis ticks to show all years with proper spacing
ax.set_xticks(years)
ax.set_xticklabels(years, rotation=45)

# Add legend
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# Add grid
plt.grid(True, alpha=0.3)

# Force tight layout to maximize plot area usage
plt.tight_layout()

# Show the plot
plt.show()

# Calculate average weights across all years
avg_weights = weights_for_plot.mean()
print(f"\nAverage weights across all years:")
for asset, avg_weight in avg_weights.items():
    print(f"  {asset}: {avg_weight:.1%}")

# Create cumulative return plot
print(f"\n" + "="*80)
print("CUMULATIVE RETURN COMPARISON")
print("="*80)


# Calculate cumulative returns
benchmark_cumulative = (1 + benchmark_returns_df['benchmark_returns']).cumprod()
optimal_cumulative = (1 + optimal_returns_df['optimal_returns']).cumprod()


plt.figure(figsize=(14, 8))

# Plot both cumulative return series
plt.plot(benchmark_cumulative.index, benchmark_cumulative.values, 
            label='Benchmark Portfolio (60% ACWI + 40% AGG)', 
            linewidth=2, color='#1f77b4', alpha=0.8)
plt.plot(optimal_cumulative.index, optimal_cumulative.values, 
            label='Optimal Portfolio', 
            linewidth=2, color='#ff7f0e', alpha=0.8)

plt.title('Cumulative Return Comparison: Optimal vs Benchmark Portfolio', 
            fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Cumulative Return', fontsize=12)

# Format y-axis as percentage
ax = plt.gca()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))
plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()


print(f"\n" + "="*80)
print("ROLLING 1-YEAR VARIANCE COMPARISON")
print("="*80)


benchmark_rolling_var = benchmark_returns_df['benchmark_returns'].rolling(window=252).var()
optimal_rolling_var = optimal_returns_df['optimal_returns'].rolling(window=252).var()

# Convert to annualized volatility (square root of variance * sqrt(252))
benchmark_rolling_vol = benchmark_rolling_var.apply(lambda x: np.sqrt(x * 252) if not pd.isna(x) else np.nan)
optimal_rolling_vol = optimal_rolling_var.apply(lambda x: np.sqrt(x * 252) if not pd.isna(x) else np.nan)

plt.figure(figsize=(14, 8))

# Plot both rolling volatility series
plt.plot(benchmark_rolling_vol.index, benchmark_rolling_vol.values, 
            label='Benchmark Portfolio (60% ACWI + 40% AGG)', 
            linewidth=2, color='#1f77b4', alpha=0.8)
plt.plot(optimal_rolling_vol.index, optimal_rolling_vol.values, 
            label='Optimal Portfolio', 
            linewidth=2, color='#ff7f0e', alpha=0.8)


plt.title('Rolling 1-Year Volatility Comparison: Optimal vs Benchmark Portfolio', 
            fontsize=16, fontweight='bold')
plt.xlabel('Date', fontsize=12)
plt.ylabel('Annualized Volatility', fontsize=12)

# Format y-axis as percentage
ax = plt.gca()
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.1%}'))

plt.legend(fontsize=11)
plt.grid(True, alpha=0.3)
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Create annual returns table by calendar year
print(f"\n" + "="*80)
print("ANNUAL RETURNS BY CALENDAR YEAR")
print("="*80)

# Group returns by calendar year and calculate annual returns
benchmark_annual = benchmark_returns_df.groupby(benchmark_returns_df.index.year)['benchmark_returns'].apply(lambda x: (1 + x).prod() - 1)
optimal_annual = optimal_returns_df.groupby(optimal_returns_df.index.year)['optimal_returns'].apply(lambda x: (1 + x).prod() - 1)

# Create a DataFrame for the comparison table
annual_comparison = pd.DataFrame({
    'Benchmark Return': benchmark_annual,
    'Optimal Return': optimal_annual
})

# Calculate outperformance
annual_comparison['Outperformance'] = annual_comparison['Optimal Return'] - annual_comparison['Benchmark Return']

# Display formatted table
print(f"{'Year':<8} {'Benchmark':<12} {'Optimal':<12} {'Outperformance':<15}")
print("-" * 50)

for year, row in annual_comparison.iterrows():
    print(f"{year:<8} {row['Benchmark Return']:<12.2%} {row['Optimal Return']:<12.2%} {row['Outperformance']:<15.2%}")

# Calculate summary statistics
print(f"\n" + "="*50)


