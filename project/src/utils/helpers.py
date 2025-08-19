"""
Module containing helper functions and utilities.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
import os
import re


def safe_numeric_convert(value: any, default: float = np.nan) -> float:
    """
    Safely convert a value to numeric type.
    
    Args:
        value: Value to convert
        default: Default value to return if conversion fails
        
    Returns:
        Converted numeric value or default
    """
    if pd.isna(value):
        return default
    
    try:
        # Remove any non-numeric characters except decimal point
        if isinstance(value, str):
            # Remove percentage sign if present
            value = value.replace('%', '')
            
            # Try to extract numeric portion
            numeric_match = re.search(r'[-+]?\d*\.?\d+', value)
            if numeric_match:
                value = numeric_match.group(0)
        
        return float(value)
    except (ValueError, TypeError):
        return default


def clean_column_name(column: str) -> str:
    """
    Clean and standardize column names.
    
    Args:
        column: Original column name
        
    Returns:
        Cleaned column name
    """
    # Convert to lowercase
    cleaned = str(column).lower()
    
    # Replace spaces and special characters with underscore
    cleaned = re.sub(r'[^\w\s]', '', cleaned)
    cleaned = re.sub(r'\s+', '_', cleaned)
    
    # Remove duplicate underscores
    cleaned = re.sub(r'_+', '_', cleaned)
    
    # Remove leading/trailing underscores
    cleaned = cleaned.strip('_')
    
    return cleaned


def create_directory_if_not_exists(directory: str) -> None:
    """
    Create a directory if it doesn't exist.
    
    Args:
        directory: Directory path to create
    """
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)


def extract_years_from_data(df: pd.DataFrame, year_col: str = 'year') -> List[int]:
    """
    Extract unique years from a DataFrame.
    
    Args:
        df: DataFrame containing year data
        year_col: Name of the column containing years
        
    Returns:
        List of unique years sorted in ascending order
    """
    if df.empty or year_col not in df.columns:
        return []
    
    years = sorted(df[year_col].unique().tolist())
    
    return years


def extract_counties_from_data(df: pd.DataFrame, county_col: str = 'county') -> List[str]:
    """
    Extract unique counties from a DataFrame.
    
    Args:
        df: DataFrame containing county data
        county_col: Name of the column containing counties
        
    Returns:
        List of unique counties sorted alphabetically
    """
    if df.empty or county_col not in df.columns:
        return []
    
    counties = sorted(df[county_col].unique().tolist())
    
    return counties


def calculate_summary_statistics(df: pd.DataFrame, 
                               value_col: str,
                               groupby_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Calculate summary statistics for a numeric column, optionally grouped by other columns.
    
    Args:
        df: DataFrame containing data
        value_col: Column to calculate statistics for
        groupby_cols: Optional list of columns to group by
        
    Returns:
        DataFrame with summary statistics
    """
    if df.empty or value_col not in df.columns:
        return pd.DataFrame()
    
    # Ensure the value column is numeric
    df = df.copy()
    df[value_col] = pd.to_numeric(df[value_col], errors='coerce')
    
    # Group by specified columns if provided
    if groupby_cols:
        valid_groupby_cols = [col for col in groupby_cols if col in df.columns]
        if valid_groupby_cols:
            grouped = df.groupby(valid_groupby_cols)
            stats = grouped[value_col].agg([
                'count', 'mean', 'median', 'std', 'min', 'max'
            ]).reset_index()
            return stats
    
    # If no groupby columns or they're not in the DataFrame, calculate overall stats
    stats = pd.DataFrame({
        'count': [df[value_col].count()],
        'mean': [df[value_col].mean()],
        'median': [df[value_col].median()],
        'std': [df[value_col].std()],
        'min': [df[value_col].min()],
        'max': [df[value_col].max()]
    })
    
    return stats


def calculate_prevalence_change(df: pd.DataFrame,
                              county: str,
                              prevalence_col: str = 'hiv_prevalence',
                              start_year: Optional[int] = None,
                              end_year: Optional[int] = None) -> Dict[str, float]:
    """
    Calculate the change in prevalence for a specific county over a time period.
    
    Args:
        df: DataFrame containing prevalence data
        county: County to calculate change for
        prevalence_col: Column containing prevalence data
        start_year: Starting year (if None, uses earliest available)
        end_year: Ending year (if None, uses latest available)
        
    Returns:
        Dictionary with absolute and percentage change
    """
    if df.empty or 'county' not in df.columns or prevalence_col not in df.columns:
        return {'absolute_change': np.nan, 'percentage_change': np.nan}
    
    # Filter for the specified county
    county_df = df[df['county'] == county].copy()
    
    if county_df.empty:
        return {'absolute_change': np.nan, 'percentage_change': np.nan}
    
    # Convert prevalence to numeric
    county_df[prevalence_col] = pd.to_numeric(county_df[prevalence_col], errors='coerce')
    
    # Get available years
    available_years = sorted(county_df['year'].unique())
    
    if not available_years:
        return {'absolute_change': np.nan, 'percentage_change': np.nan}
    
    # Use provided years or defaults
    if start_year is None:
        start_year = available_years[0]
    
    if end_year is None:
        end_year = available_years[-1]
    
    # Get start and end values
    start_value = county_df[county_df['year'] == start_year][prevalence_col].values
    end_value = county_df[county_df['year'] == end_year][prevalence_col].values
    
    if len(start_value) == 0 or len(end_value) == 0:
        return {'absolute_change': np.nan, 'percentage_change': np.nan}
    
    start_value = start_value[0]
    end_value = end_value[0]
    
    # Calculate changes
    absolute_change = end_value - start_value
    
    if start_value != 0:
        percentage_change = (absolute_change / start_value) * 100
    else:
        percentage_change = np.nan
    
    return {
        'start_year': start_year,
        'end_year': end_year,
        'start_value': start_value,
        'end_value': end_value,
        'absolute_change': absolute_change,
        'percentage_change': percentage_change
    }


if __name__ == "__main__":
    # Example usage
    from datetime import datetime
    
    # Print current date and time
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"Current time: {current_time}")
    
    # Ensure data directories exist
    create_directory_if_not_exists("data/raw")
    create_directory_if_not_exists("data/processed")
    create_directory_if_not_exists("data/visualizations")
    create_directory_if_not_exists("data/models")
    
    print("Created necessary directories for the project.")