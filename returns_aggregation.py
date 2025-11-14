import pandas as pd
import numpy as np


def daily_to_weekly_returns(daily_returns_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate daily returns into weekly returns based on calendar weeks.
    
    This function takes a DataFrame of daily returns and aggregates them into 
    weekly returns, where each week corresponds to a calendar week (Monday-Sunday).
    The weekly return for each asset is computed as the compound return over all 
    trading days in that calendar week.
    
    Parameters:
    -----------
    daily_returns_df : pd.DataFrame
        DataFrame with daily returns, where:
        - Index: DatetimeIndex representing trading dates
        - Columns: Asset names/tickers
        - Values: Daily returns (as decimals, e.g., 0.01 for 1%)
    
    Returns:
    --------
    pd.DataFrame
        DataFrame with weekly returns, where:
        - Index: DatetimeIndex representing the end of each calendar week (Sunday)
        - Columns: Same asset names as input
        - Values: Compound weekly returns for each asset
    
    Example:
    --------
    >>> # Assuming daily_returns is a DataFrame of daily returns
    >>> weekly_returns = daily_to_weekly_returns(daily_returns)
    >>> weekly_returns.head()
    """
    
    # Create a copy to avoid modifying the original
    df = daily_returns_df.copy()
    
    # Use strftime to get ISO year and week for grouping
    # ISO calendar: Monday=1, Sunday=7 (so weeks end on Sunday)
    df_indexed = df.copy()
    
    # Create year-week identifiers
    year_week = df.index.to_series().dt.strftime('%Y-%W')
    
    # Group by year-week and get the last date of each week
    week_groups = df.groupby(year_week)
    week_ends_list = []
    weekly_returns_list = []
    
    for week_id, group_df in week_groups:
        # Compound the returns for this week
        # Weekly return = (1 + r1) * (1 + r2) * ... * (1 + rn) - 1
        weekly_return = (1 + group_df).prod(axis=0) - 1
        weekly_returns_list.append(weekly_return)
        
        # Store the end date of this week (last trading day in the group)
        week_ends_list.append(group_df.index[-1])
    
    # Create the weekly returns DataFrame
    weekly_df = pd.DataFrame(weekly_returns_list, index=week_ends_list)
    weekly_df.index.name = daily_returns_df.index.name
    
    # Sort by index to ensure chronological order
    weekly_df = weekly_df.sort_index()
    
    return weekly_df
