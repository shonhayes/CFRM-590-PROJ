import yfinance as yf
import pandas as pd
import numpy as np
from portfolio_optimization import mean_variance_optimization, mean_cvar_optimization
from returns_aggregation import daily_to_weekly_returns
import pandas_datareader as pdr
from datetime import datetime
import matplotlib.pyplot as plt
import random

# Just in case for the convex optimization
random.seed(590)

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

expense_ratios = { # I have removed expense ratios as expenses are already reflected in the etf prices
    "ACWI": 0,   # iShares MSCI ACWI ETF (0.32%)
    "EFA": 0,    # iShares MSCI EAFE ETF (0.32%)
    "VTI": 0,    # Vanguard Total Stock Market ETF (0.03%)
    "VNQ": 0,    # Vanguard Real Estate ETF (0.13%)
    "AGG": 0,    # iShares Core U.S. Aggregate Bond ETF (0.03%)
    "BIL": 0,    # SPDR Bloomberg 1-3 Month T-Bill ETF (~0.14%)
    "HYG": 0,    # iShares iBoxx $ High Yield Corporate Bond ETF (0.49%)
    "GSG": 0     # iShares S&P GSCI Commodity-Indexed Trust (0.75%)
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

# Download price history (cap data before Nov 1, 2025)
price_data = yf.download(list(tickers.values()), start="2000-01-01", end="2025-11-01")

# Handle different yfinance data structures
if isinstance(price_data.columns, pd.MultiIndex):
    # Multi-level columns (newer yfinance versions)
    # Check if Adj Close exists, otherwise use Close
    if ('Adj Close', list(tickers.values())[0]) in price_data.columns:
        price_data = price_data['Adj Close']
        print("a")
    elif ('Close', list(tickers.values())[0]) in price_data.columns:
        price_data = price_data['Close']
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

# Compute daily returns and cap to data before Nov 1, 2025
returns = price_data.pct_change().dropna()
returns = returns.loc[: "2025-10-31"]

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

print("="*80)
print(f"Portfolio returns data shape: {portfolio_returns_data.shape}")
print(f"Date range: {most_recent_start} to {portfolio_returns_data.index[-1]}")

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
optimal_returns_df = pd.DataFrame()
cvar_returns_df = pd.DataFrame()

# Initialize DataFrame to store optimal portfolio weights for each year
optimal_weights_df = pd.DataFrame()
cvar_weights_df = pd.DataFrame()

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
    
    reference_geometric_daily_return = (1 + window_portfolio_returns).prod() ** (1 / len(window_portfolio_returns)) - 1
    
    # Calculate mean-variance optimal portfolio weights using geometric daily return as target

    if count == 0:
        # Mean-Variance optimization
        optimal_weights, portfolio_return, portfolio_volatility = mean_variance_optimization(
            returns_df=window_returns,
            target_return=reference_geometric_daily_return+0.00005,
            allow_short_selling=False,  # No short selling
            max_weight=1.0,  # No maximum weight constraint
            min_weight=0.0,
            trading_cost_per_dollar=trading_costs_array   # No minimum weight constraint
        )
        
        # Mean-CVaR optimization
        cvar_weights, cvar_portfolio_return, cvar_value = mean_cvar_optimization(
            returns_df=window_returns,
            target_return=reference_geometric_daily_return+0.00005,
            confidence_level=0.95,
            allow_short_selling=False,  # No short selling
            max_weight=1.0,  # No maximum weight constraint
            min_weight=0.0,
            trading_cost_per_dollar=trading_costs_array   # No minimum weight constraint
        )
    else:
        # Mean-Variance optimization
        optimal_weights, portfolio_return, portfolio_volatility = mean_variance_optimization(
            returns_df=window_returns,
            target_return=reference_geometric_daily_return+(1 + 0.02)**(1/252)-1,
            allow_short_selling=False,  # No short selling
            max_weight=1.0,  # No maximum weight constraint
            min_weight=0.0, 
            prev_weights=optimal_weights,
            trading_cost_per_dollar=trading_costs_array   # No minimum weight constraint
        )
        
        # Mean-CVaR optimization
        cvar_weights, cvar_portfolio_return, cvar_value = mean_cvar_optimization(
            returns_df=window_returns,
            target_return=reference_geometric_daily_return+(1 + 0.02)**(1/252)-1,
            confidence_level=0.95,
            allow_short_selling=False,  # No short selling
            max_weight=1.0,  # No maximum weight constraint
            min_weight=0.0,
            prev_weights=cvar_weights,
            trading_cost_per_dollar=trading_costs_array   # No minimum weight constraint
        )
    

    count += 1
    
    # Store mean-variance optimal weights for this year
    weights_dict = {}
    for i, asset in enumerate(window_returns.columns):
        weights_dict[asset] = np.abs(optimal_weights[i])  # Use absolute values for display
    
    # Add to weights DataFrame
    optimal_weights_df[year] = pd.Series(weights_dict)
    
    # Store mean-CVaR weights for this year
    cvar_weights_dict = {}
    for i, asset in enumerate(window_returns.columns):
        cvar_weights_dict[asset] = np.abs(cvar_weights[i])  # Use absolute values for display
    
    # Add to CVaR weights DataFrame
    cvar_weights_df[year] = pd.Series(cvar_weights_dict)
    
    # Calculate out-of-sample returns for the year following the window
    next_year_start = window_end + pd.DateOffset(days=1)
    next_year_end = min(window_end + pd.DateOffset(years=1), returns.index[-1])
    
    # Get the out-of-sample data
    out_of_sample_returns = returns.loc[next_year_start:next_year_end]
    
    # Calculate mean-variance optimal portfolio returns out-of-sample (keep as pandas Series with date index)
    optimal_out_of_sample = pd.Series(
        np.sum(optimal_weights * out_of_sample_returns.values, axis=1),
        index=out_of_sample_returns.index,
        name='optimal_returns'
    )
    
    # Calculate mean-CVaR portfolio returns out-of-sample
    cvar_out_of_sample = pd.Series(
        np.sum(cvar_weights * out_of_sample_returns.values, axis=1),
        index=out_of_sample_returns.index,
        name='cvar_returns'
    )
    
    # Store returns in DataFrames (concatenate to existing data)
    if optimal_returns_df.empty:
        optimal_returns_df = pd.DataFrame(optimal_out_of_sample, columns=['optimal_returns'])
    else:
        optimal_returns_df = pd.concat([optimal_returns_df, pd.DataFrame(optimal_out_of_sample, columns=['optimal_returns'])])
    
    if cvar_returns_df.empty:
        cvar_returns_df = pd.DataFrame(cvar_out_of_sample, columns=['cvar_returns'])
    else:
        cvar_returns_df = pd.concat([cvar_returns_df, pd.DataFrame(cvar_out_of_sample, columns=['cvar_returns'])])


# Get the date range from our portfolio data
portfolio_index = optimal_returns_df.index.union(cvar_returns_df.index)
start_date = portfolio_index[0].strftime('%Y-%m-%d')
end_date = portfolio_index[-1].strftime('%Y-%m-%d')

# Fetch DTB3 (3-Month Treasury Bill) from FRED
risk_free_rate = pdr.get_data_fred('DTB3', start=start_date, end=end_date)
risk_free_rate = risk_free_rate['DTB3'].dropna()

# Convert annual rate to daily rate
risk_free_daily = (1 + risk_free_rate / 100) ** (1/252) - 1


# Align risk-free rate with portfolio data
risk_free_aligned = risk_free_daily.reindex(portfolio_index, method='ffill').fillna(0)

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
optimal_metrics = calculate_metrics(
    optimal_returns_df.iloc[:, 0], 
    risk_free_aligned.loc[optimal_returns_df.index], 
    'Mean-Variance Optimal'
)

cvar_metrics = calculate_metrics(
    cvar_returns_df.iloc[:, 0], 
    risk_free_aligned.loc[cvar_returns_df.index], 
    'Mean-CVaR Optimal'
)

# Create comparison table
comparison_data = [optimal_metrics, cvar_metrics]
comparison_df = pd.DataFrame(comparison_data)

print(f"Analysis Period: {portfolio_index[0].strftime('%Y-%m-%d')} to {portfolio_index[-1].strftime('%Y-%m-%d')}")
print(f"Total Observations: {len(portfolio_index)}")
print(f"Risk-Free Rate Source: FRED DTB3 (3-Month Treasury Bill)")
print()

"""# Display formatted table
print(f"{'Portfolio':<30} {'Annual Return':<15} {'Annual Std Dev':<15} {'Sharpe Ratio':<12} {'97.5% CVaR':<12}")
print("-" * 85)

for _, row in comparison_df.iterrows():
    print(f"{row['Portfolio']:<33} "
          f"{row['Annual Return']:<15.2%} "
          f"{row['Annual Std Dev']:<15.2%} "
          f"{row['Sharpe Ratio']:<12.2f} "
          f"{row['97.5% CVaR']:<12.2%}")

print("-" * 85)"""

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

# Create the stackplot with proper x-axis handling (shifted +1 year)
years = weights_for_plot.index.values + 1

plt.stackplot(years, 
                *[weights_for_plot[asset] for asset in weights_for_plot.columns],
                labels=weights_for_plot.columns,
                colors=asset_colors,
                alpha=0.8)

# Customize the plot
plt.title('Daily Mean-Variance Portfolio Over Time', fontsize=16, fontweight='bold')
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

# Create stackplot for CVaR portfolio weights
cvar_weights_for_plot = cvar_weights_df.T

plt.figure(figsize=(14, 8))

# Get the color list in the same order as the DataFrame columns
asset_colors = [colors.get(asset, '#000000') for asset in cvar_weights_for_plot.columns]

# Create the stackplot with proper x-axis handling (shifted +1 year)
years = cvar_weights_for_plot.index.values + 1

plt.stackplot(years, 
                *[cvar_weights_for_plot[asset] for asset in cvar_weights_for_plot.columns],
                labels=cvar_weights_for_plot.columns,
                colors=asset_colors,
                alpha=0.8)

# Customize the plot
plt.title('Daily Mean-CVaR Portfolio Over Time', fontsize=16, fontweight='bold')
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

# Calculate average weights for CVaR portfolio
cvar_avg_weights = cvar_weights_for_plot.mean()


# Calculate cumulative returns
optimal_cumulative = (1 + optimal_returns_df['optimal_returns']).cumprod()
cvar_cumulative = (1 + cvar_returns_df['cvar_returns']).cumprod()

"""
plt.figure(figsize=(14, 8))

# Don't think that I need this plot. Leaving it here just in case.


# Plot cumulative return series
plt.plot(optimal_cumulative.index, optimal_cumulative.values, 
            label='Mean-Variance Optimal Portfolio', 
            linewidth=2, color='#ff7f0e', alpha=0.8)
plt.plot(cvar_cumulative.index, cvar_cumulative.values, 
            label='Mean-CVaR Optimal Portfolio', 
            linewidth=2, color='#2ca02c', alpha=0.8)

plt.title('Cumulative Return Comparison: Optimal Portfolios', 
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
plt.show()
"""

# WEEKLY PORTFOLIO OPTIMIZATION

# Convert daily returns to weekly returns
weekly_returns = daily_to_weekly_returns(returns)

# Initialize DataFrames for weekly portfolios
weekly_optimal_returns_df = pd.DataFrame()
weekly_cvar_returns_df = pd.DataFrame()

# Initialize DataFrame to store optimal portfolio weights for each year (using weekly data)
weekly_optimal_weights_df = pd.DataFrame()
weekly_cvar_weights_df = pd.DataFrame()

# Find the start date for weekly portfolio calculation
weekly_series_start_dates = weekly_returns.apply(lambda x: x.first_valid_index())
weekly_most_recent_start = weekly_series_start_dates.max()

# Calculate start date (5 years after the most recent series start)
weekly_portfolio_start_date = weekly_most_recent_start + pd.DateOffset(years=5)

# Find the earliest date where all series have 5 years of data
weekly_earliest_start = weekly_series_start_dates.max()  # Most recent start among all series
weekly_five_years_later = weekly_earliest_start + pd.DateOffset(years=5)

weekly_latest_end = pd.Timestamp("2024-12-31")

# Generate year-end dates for rolling windows using weekly data
weekly_start_year = weekly_five_years_later.year
weekly_end_year = weekly_latest_end.year

# Rolling 5-year window loop for weekly returns
weekly_count = 0
weekly_optimal_weights = None
weekly_cvar_weights = None

for year in range(weekly_start_year, weekly_end_year + 1):
    # Define the 5-year window (5 years ending at year-end)
    weekly_window_end = pd.Timestamp(f"{year}-12-31")
    weekly_window_start = weekly_window_end - pd.DateOffset(years=5)

    # Extract the 5-year window of weekly returns
    weekly_window_returns = weekly_returns.loc[weekly_window_start:weekly_window_end]
    
    # Skip if we don't have enough data
    if len(weekly_window_returns) < 10:
        continue
    
    # Calculate portfolio returns for this window (60% ACWI + 40% AGG)
    weekly_window_portfolio_returns = (acwi_weight * weekly_window_returns['MSCI_ACWI_IMI'] + 
                                        agg_weight * weekly_window_returns['Bloomberg_US_Agg'])
    
    # Calculate geometric weekly return as target
    weekly_reference_geometric_return = (1 + weekly_window_portfolio_returns).prod() ** (1 / len(weekly_window_portfolio_returns)) - 1

    # Calculate mean-variance and mean-CVaR optimal portfolio weights using weekly returns
    if weekly_count == 0:
        # Mean-Variance optimization
        weekly_optimal_weights, weekly_port_ret, weekly_port_vol = mean_variance_optimization(
            returns_df=weekly_window_returns,
            target_return=weekly_reference_geometric_return+0.00005,
            allow_short_selling=False,
            max_weight=1.0,
            min_weight=0.0,
            trading_cost_per_dollar=trading_costs_array
        )
        
        # Mean-CVaR optimization
        weekly_cvar_weights, weekly_cvar_port_ret, weekly_cvar_value = mean_cvar_optimization(
            returns_df=weekly_window_returns,
            target_return=weekly_reference_geometric_return+0.00005,
            confidence_level=0.95,
            allow_short_selling=False,
            max_weight=1.0,
            min_weight=0.0,
            trading_cost_per_dollar=trading_costs_array
        )
    else:
        # Mean-Variance optimization
        weekly_optimal_weights, weekly_port_ret, weekly_port_vol = mean_variance_optimization(
            returns_df=weekly_window_returns,
            target_return=weekly_reference_geometric_return+(1 + 0.02)**(1/52)-1,
            allow_short_selling=False,
            max_weight=1.0,
            min_weight=0.0,
            prev_weights=weekly_optimal_weights,
            trading_cost_per_dollar=trading_costs_array
        )
        
        # Mean-CVaR optimization
        weekly_cvar_weights, weekly_cvar_port_ret, weekly_cvar_value = mean_cvar_optimization(
            returns_df=weekly_window_returns,
            target_return=weekly_reference_geometric_return+(1 + 0.02)**(1/52)-1,
            confidence_level=0.95,
            allow_short_selling=False,
            max_weight=1.0,
            min_weight=0.0,
            prev_weights=weekly_cvar_weights,
            trading_cost_per_dollar=trading_costs_array
        )

    weekly_count += 1
    
    # Store mean-variance optimal weights for this year
    weekly_weights_dict = {}
    for i, asset in enumerate(weekly_window_returns.columns):
        weekly_weights_dict[asset] = np.abs(weekly_optimal_weights[i])
    
    weekly_optimal_weights_df[year] = pd.Series(weekly_weights_dict)
    
    # Store mean-CVaR weights for this year
    weekly_cvar_weights_dict = {}
    for i, asset in enumerate(weekly_window_returns.columns):
        weekly_cvar_weights_dict[asset] = np.abs(weekly_cvar_weights[i])
    
    weekly_cvar_weights_df[year] = pd.Series(weekly_cvar_weights_dict)
    
    # Calculate out-of-sample returns for the year following the window
    weekly_next_year_start = weekly_window_end + pd.DateOffset(days=1)
    weekly_next_year_end = min(weekly_window_end + pd.DateOffset(years=1), weekly_returns.index[-1])
    
    # Get the out-of-sample data
    weekly_out_of_sample_returns = weekly_returns.loc[weekly_next_year_start:weekly_next_year_end]
    
    if len(weekly_out_of_sample_returns) > 0:
        # Calculate mean-variance optimal portfolio returns out-of-sample
        weekly_optimal_out_of_sample = pd.Series(
            np.sum(weekly_optimal_weights * weekly_out_of_sample_returns.values, axis=1),
            index=weekly_out_of_sample_returns.index,
            name='weekly_optimal_returns'
        )
        
        # Calculate mean-CVaR portfolio returns out-of-sample
        weekly_cvar_out_of_sample = pd.Series(
            np.sum(weekly_cvar_weights * weekly_out_of_sample_returns.values, axis=1),
            index=weekly_out_of_sample_returns.index,
            name='weekly_cvar_returns'
        )
        
        # Store returns in DataFrames
        if weekly_optimal_returns_df.empty:
            weekly_optimal_returns_df = pd.DataFrame(weekly_optimal_out_of_sample, columns=['weekly_optimal_returns'])
        else:
            weekly_optimal_returns_df = pd.concat([weekly_optimal_returns_df, pd.DataFrame(weekly_optimal_out_of_sample, columns=['weekly_optimal_returns'])])
        
        if weekly_cvar_returns_df.empty:
            weekly_cvar_returns_df = pd.DataFrame(weekly_cvar_out_of_sample, columns=['weekly_cvar_returns'])
        else:
            weekly_cvar_returns_df = pd.concat([weekly_cvar_returns_df, pd.DataFrame(weekly_cvar_out_of_sample, columns=['weekly_cvar_returns'])])

# Create stackplot for weekly mean-variance portfolio weights
weekly_weights_for_plot = weekly_optimal_weights_df.T

plt.figure(figsize=(14, 8))

# Get the color list in the same order as the DataFrame columns
asset_colors = [colors.get(asset, '#000000') for asset in weekly_weights_for_plot.columns]

# Create the stackplot with proper x-axis handling (shifted +1 year)
years = weekly_weights_for_plot.index.values + 1

plt.stackplot(years, 
                *[weekly_weights_for_plot[asset] for asset in weekly_weights_for_plot.columns],
                labels=weekly_weights_for_plot.columns,
                colors=asset_colors,
                alpha=0.8)

# Customize the plot
plt.title('Weekly Mean-Variance Portfolio Over Time', fontsize=16, fontweight='bold')
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

# Calculate average weights for weekly mean-variance portfolio
weekly_avg_weights = weekly_weights_for_plot.mean()

# Create stackplot for weekly mean-CVaR portfolio weights
weekly_cvar_weights_for_plot = weekly_cvar_weights_df.T

plt.figure(figsize=(14, 8))

# Get the color list in the same order as the DataFrame columns
asset_colors = [colors.get(asset, '#000000') for asset in weekly_cvar_weights_for_plot.columns]

# Create the stackplot with proper x-axis handling (shifted +1 year)
years = weekly_cvar_weights_for_plot.index.values + 1

plt.stackplot(years, 
                *[weekly_cvar_weights_for_plot[asset] for asset in weekly_cvar_weights_for_plot.columns],
                labels=weekly_cvar_weights_for_plot.columns,
                colors=asset_colors,
                alpha=0.8)

# Customize the plot
plt.title('Weekly Mean-CVaR Portfolio Over Time', fontsize=16, fontweight='bold')
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

# Calculate average weights for weekly mean-CVaR portfolio
weekly_cvar_avg_weights = weekly_cvar_weights_for_plot.mean()

# Consolidated average weight table
all_assets = (avg_weights.index
              .union(cvar_avg_weights.index)
              .union(weekly_avg_weights.index)
              .union(weekly_cvar_avg_weights.index))

weight_summary = pd.DataFrame({
    'Daily Mean-Variance': avg_weights,
    'Daily Mean-CVaR': cvar_avg_weights,
    'Weekly Mean-Variance': weekly_avg_weights,
    'Weekly Mean-CVaR': weekly_cvar_avg_weights,
}).reindex(all_assets).fillna(0)

print(f"\n" + "="*80)
print("AVERAGE WEIGHTS ACROSS ALL YEARS")
print("="*80)
print(f"{'Asset':<18} {'Daily MV':>12} {'Daily CVaR':>12} {'Weekly MV':>12} {'Weekly CVaR':>12}")
print("-" * 70)
for asset, row in weight_summary.iterrows():
    print(f"{asset:<18} "
          f"{row['Daily Mean-Variance']:>12.1%} "
          f"{row['Daily Mean-CVaR']:>12.1%} "
          f"{row['Weekly Mean-Variance']:>12.1%} "
          f"{row['Weekly Mean-CVaR']:>12.1%}")

# ============================================================================
# WEIGHT DIFFERENCE STACKPLOTS
# ============================================================================

print(f"\n" + "="*80)
print("PORTFOLIO WEIGHT DIFFERENCES: WEEKLY vs DAILY")
print("="*80)

# Calculate differences between weekly and daily mean-variance weights
# Both DataFrames should have the same years as columns
daily_weights_plot = optimal_weights_df.T
weekly_weights_plot = weekly_optimal_weights_df.T

# Get common years
common_years = sorted(set(daily_weights_plot.index) & set(weekly_weights_plot.index))

# Create DataFrames with only common years for comparison
daily_weights_common = daily_weights_plot.loc[common_years]
weekly_weights_common = weekly_weights_plot.loc[common_years]

# Calculate the difference (Weekly - Daily)
mv_weight_diff = weekly_weights_common - daily_weights_common

print(f"\nMean-Variance Weight Differences (Weekly - Daily):")

# Create bar chart for mean-variance weight differences
plt.figure(figsize=(14, 8))

# Separate positive and negative differences for stacking
mv_diff_positive = mv_weight_diff.copy()
mv_diff_positive[mv_diff_positive < 0] = 0

mv_diff_negative = mv_weight_diff.copy()
mv_diff_negative[mv_diff_negative > 0] = 0
mv_diff_negative = -mv_diff_negative  # Make negative differences positive for stacking

years = mv_weight_diff.index.values
display_years = years + 1
x_pos = np.arange(len(years))
width = 0.1

# Plot each asset as a separate bar group
for i, asset in enumerate(mv_weight_diff.columns):
    asset_color = colors.get(asset, '#000000')
    values = mv_weight_diff[asset].values
    
    # Plot bars - positive and negative values will be displayed correctly
    plt.bar(x_pos + i * width, values, width, label=asset, color=asset_color, alpha=0.8)

# Customize the plot
plt.title('Mean-Variance Portfolio: Weight Differences (Weekly - Daily Returns)', 
          fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Weight Difference', fontsize=12)

# Set x-axis
ax = plt.gca()
ax.set_xticks(x_pos + width * 3.5)  # Center the tick labels
ax.set_xticklabels(display_years, rotation=45)

# Add a horizontal line at zero
plt.axhline(y=0, color='black', linestyle='-', linewidth=1.0)

# Add legend
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

# Add grid
plt.grid(True, alpha=0.3, axis='y')

# Force tight layout
plt.tight_layout()

plt.show()

# Print summary statistics for mean-variance differences in table form
mv_stats = pd.DataFrame({
    'Mean': mv_weight_diff.mean(),
    'Max': mv_weight_diff.max(),
    'Min': mv_weight_diff.min(),
    'MAD': np.abs(mv_weight_diff).mean()
})

print("\nMean-Variance Weight Change Statistics:")
print(f"{'Asset':<18} {'Mean':>12} {'Max':>12} {'Min':>12} {'MAD':>12}")
print("-" * 78)
for asset, row in mv_stats.iterrows():
    print(f"{asset:<18} {row['Mean']:+12.2%} {row['Max']:+12.2%} {row['Min']:+12.2%} {row['MAD']:>12.2%}")

# Calculate differences between weekly and daily mean-CVaR weights
daily_cvar_weights_plot = cvar_weights_df.T
weekly_cvar_weights_plot = weekly_cvar_weights_df.T

# Get common years
common_years_cvar = sorted(set(daily_cvar_weights_plot.index) & set(weekly_cvar_weights_plot.index))

# Create DataFrames with only common years for comparison
daily_cvar_weights_common = daily_cvar_weights_plot.loc[common_years_cvar]
weekly_cvar_weights_common = weekly_cvar_weights_plot.loc[common_years_cvar]

# Calculate the difference (Weekly - Daily)
cvar_weight_diff = weekly_cvar_weights_common - daily_cvar_weights_common

print(f"\nMean-CVaR Weight Differences (Weekly - Daily):")

# Create bar chart for mean-CVaR weight differences
plt.figure(figsize=(14, 8))

years_cvar = cvar_weight_diff.index.values
display_years_cvar = years_cvar + 1
x_pos_cvar = np.arange(len(years_cvar))
width = 0.1

# Plot each asset as a separate bar group
for i, asset in enumerate(cvar_weight_diff.columns):
    asset_color = colors.get(asset, '#000000')
    values = cvar_weight_diff[asset].values
    
    # Plot bars - positive and negative values will be displayed correctly
    plt.bar(x_pos_cvar + i * width, values, width, label=asset, color=asset_color, alpha=0.8)

# Customize the plot
plt.title('Mean-CVaR Portfolio: Weight Differences (Weekly - Daily Returns)', 
          fontsize=16, fontweight='bold')
plt.xlabel('Year', fontsize=12)
plt.ylabel('Weight Difference', fontsize=12)

# Set x-axis
ax = plt.gca()
ax.set_xticks(x_pos_cvar + width * 3.5)  # Center the tick labels
ax.set_xticklabels(display_years_cvar, rotation=45)

# Add a horizontal line at zero
plt.axhline(y=0, color='black', linestyle='-', linewidth=1.0)

# Add legend
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)

# Add grid
plt.grid(True, alpha=0.3, axis='y')

# Force tight layout
plt.tight_layout()

plt.show()

# Print summary statistics for mean-CVaR differences in table form
cvar_stats = pd.DataFrame({
    'Mean': cvar_weight_diff.mean(),
    'Max': cvar_weight_diff.max(),
    'Min': cvar_weight_diff.min(),
    'MAD': np.abs(cvar_weight_diff).mean()
})

print("\nMean-CVaR Weight Change Statistics:")
print(f"{'Asset':<18} {'Mean':>12} {'Max':>12} {'Min':>12} {'MAD':>12}")
print("-" * 78)
for asset, row in cvar_stats.iterrows():
    print(f"{asset:<18} {row['Mean']:+12.2%} {row['Max']:+12.2%} {row['Min']:+12.2%} {row['MAD']:>12.2%}")

# ============================================================================
# SHARPE RATIO SUMMARY
# ============================================================================

def annualized_sharpe_ratio(returns_series, risk_free_series, periods_per_year):
    """Compute annualized Sharpe ratio with aligned risk-free series."""
    aligned_rf = risk_free_series.reindex(returns_series.index, method='ffill').fillna(0)
    excess_returns = returns_series - aligned_rf
    return (excess_returns.mean() * periods_per_year) / (excess_returns.std() * np.sqrt(periods_per_year))

# Build a weekly risk-free return series to match weekly portfolios
risk_free_weekly = daily_to_weekly_returns(risk_free_daily.to_frame('risk_free'))['risk_free']

def compute_sharpe(series, rf_series, periods_per_year, year=None):
    """Compute annualized Sharpe for full series or a specific calendar year."""
    subset = series if year is None else series[series.index.year == year]
    if subset.empty:
        return None
    return annualized_sharpe_ratio(subset, rf_series, periods_per_year)

rows = []

# Add overall Sharpe ratios as final row
rows.append({
    'year': 'All',
    'daily_mv': compute_sharpe(optimal_returns_df['optimal_returns'], risk_free_daily, 252),
    'daily_cvar': compute_sharpe(cvar_returns_df['cvar_returns'], risk_free_daily, 252),
    'weekly_mv': compute_sharpe(weekly_optimal_returns_df['weekly_optimal_returns'], risk_free_weekly, 52),
    'weekly_cvar': compute_sharpe(weekly_cvar_returns_df['weekly_cvar_returns'], risk_free_weekly, 52),
})

def fmt(val):
    return f"{val:.2f}" if val is not None else "NA"

print(f"\n" + "="*80)
print("ANNUALIZED SHARPE RATIOS (BY YEAR)")
print("="*80)
print(f"{'Year':<8} {'Daily MV':<12} {'Daily CVaR':<12} {'Weekly MV':<12} {'Weekly CVaR':<12}")
print("-" * 60)
for row in rows:
    print(f"{str(row['year']):<8} {fmt(row['daily_mv']):<12} {fmt(row['daily_cvar']):<12} "
          f"{fmt(row['weekly_mv']):<12} {fmt(row['weekly_cvar']):<12}")
