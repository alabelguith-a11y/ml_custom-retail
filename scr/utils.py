"""
Utility functions for retail customer preprocessing pipeline.
"""

import numpy as np
import pandas as pd


# ── Color constants used across plots ─────────────────────────────────────
BLUE = "#4C9BE8"
ORANGE = "#E8A44C"
RED = "#E85C4C"
CHURN_PALETTE = {0: BLUE, 1: RED}


def ip_is_private(ip_str: str) -> int:
    """
    Check if an IP address is private (RFC-1918).
    
    Private ranges:
    - 10.0.0.0 — 10.255.255.255
    - 172.16.0.0 — 172.31.255.255
    - 192.168.0.0 — 192.168.255.255
    
    Args:
        ip_str: IP address string (e.g., "192.168.1.1")
    
    Returns:
        1 if private, 0 if public, -1 if invalid
    """
    if not isinstance(ip_str, str):
        return -1
    
    parts = ip_str.strip().split('.')
    if len(parts) != 4:
        return -1
    
    try:
        a, b = int(parts[0]), int(parts[1])
    except ValueError:
        return -1
    
    # Check RFC-1918 ranges
    if a == 10:
        return 1
    if a == 172 and 16 <= b <= 31:
        return 1
    if a == 192 and b == 168:
        return 1
    
    return 0


def ip_first_octet(ip_str: str) -> float:
    """
    Extract first octet from IP address.
    
    Args:
        ip_str: IP address string
    
    Returns:
        First octet as integer, or NaN if invalid
    """
    if not isinstance(ip_str, str):
        return np.nan
    
    try:
        return int(ip_str.strip().split('.')[0])
    except (ValueError, IndexError):
        return np.nan


def detect_missing_values(df: pd.DataFrame, threshold: float = 50.0) -> pd.DataFrame:
    """
    Detect columns with missing values above threshold.
    
    Args:
        df: Input DataFrame
        threshold: Percentage threshold (default: 50%)
    
    Returns:
        DataFrame with missing value statistics
    """
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    
    missing_df = pd.DataFrame({
        'Manquants': missing,
        'Pourcentage (%)': missing_pct
    }).query('Manquants > 0').sort_values('Pourcentage (%)', ascending=False)
    
    return missing_df


def get_high_correlation_pairs(corr_matrix: pd.DataFrame, threshold: float = 0.85) -> list:
    """
    Extract pairs of highly correlated features.
    
    Args:
        corr_matrix: Correlation matrix
        threshold: Correlation threshold (default: 0.85)
    
    Returns:
        List of tuples (feature_a, feature_b, correlation)
    """
    corr_abs = corr_matrix.abs()
    upper_tri = corr_abs.where(
        np.triu(np.ones_like(corr_abs, dtype=bool), k=1)
    )
    
    high_corr_pairs = [
        (col, row, upper_tri.at[row, col])
        for col in upper_tri.columns
        for row in upper_tri.index
        if upper_tri.at[row, col] > threshold
    ]
    
    high_corr_pairs.sort(key=lambda x: x[2], reverse=True)
    
    return high_corr_pairs


def create_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create 10 derived features for enhanced ML signal.
    
    Args:
        df: Input DataFrame with raw features
    
    Returns:
        DataFrame with new engineered features added
    """
    df = df.copy()
    
    # 1. Spending intensity (£ per day since last purchase)
    df['MonetaryPerDay'] = df['MonetaryTotal'] / (df['Recency'] + 1)
    
    # 2. Average basket value
    df['AvgBasketValue'] = df['MonetaryTotal'] / df['Frequency'].replace(0, np.nan)
    
    # 3. Tenure-Recency ratio (high = inactive despite longevity)
    df['TenureRecencyRatio'] = df['Recency'] / (df['CustomerTenureDays'] + 1)
    
    # 4. Cancellation rate per frequency
    df['CancelPerFreq'] = df['CancelledTransactions'] / (df['Frequency'] + 1)
    
    # 5. Share of max order in total
    df['MaxOrderShare'] = df['MonetaryMax'] / (df['MonetaryTotal'].abs() + 1)
    
    # 6. Support burden (tickets per tenure day)
    df['SupportBurden'] = df['SupportTicketsCount'] / (df['CustomerTenureDays'] + 1)
    
    # 7. Product breadth per transaction
    df['ProdBreadthPerTrans'] = df['UniqueProducts'] / (df['Frequency'] + 1)
    
    # 8. Flag: recent customer (tenure < 90 days)
    df['IsNewCustomer'] = (df['CustomerTenureDays'] < 90).astype(int)
    
    # 9. Flag: high cancellation rate (> 10% of transactions)
    df['HighCancelFlag'] = (
        (df['CancelledTransactions'] / (df['TotalTransactions'] + 1)) > 0.10
    ).astype(int)
    
    # 10. Years active since registration (reference: 2011)
    df['RegYearsActive'] = 2011 - df['RegYear'].fillna(2010)
    
    return df


def print_preprocessing_summary(original_shape: tuple, final_shape: tuple, 
                                dropped_features: dict) -> None:
    """
    Print summary statistics of preprocessing.
    
    Args:
        original_shape: Original data shape (rows, cols)
        final_shape: Final data shape (rows, cols)
        dropped_features: Dict with feature drop categories and counts
    """
    print('=' * 60)
    print('  RÉSUMÉ PRÉTRAITEMENT')
    print('=' * 60)
    print(f'  Observations          : {original_shape[0]:,}')
    print(f'  Features initiales    : {original_shape[1]}')
    print(f'  Features finales      : {final_shape[1]}')
    print()
    print('  Suppressions :')
    for category, count in dropped_features.items():
        print(f'    {category:30s} : -{count}')
    print('=' * 60)
