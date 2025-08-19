"""
Module for feature engineering and transformation.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union


class FeatureEngineer:
    """Class for creating and transforming features for analysis and modeling."""
    
    def __init__(self):
        """Initialize the FeatureEngineer."""
        pass
    
    def create_time_features(self, df: pd.DataFrame, date_col: str = 'year') -> pd.DataFrame:
        """
        Create time-based features for time series analysis.
        
        Args:
            df: DataFrame containing time data
            date_col: Name of the column containing date/year information
            
        Returns:
            DataFrame with additional time-based features
        """
        if df.empty or date_col not in df.columns:
            return df
            
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # If date_col is just the year (as integer)
        if pd.api.types.is_integer_dtype(result_df[date_col]):
            # Calculate years since 2000 (arbitrary reference point)
            result_df['years_since_2000'] = result_df[date_col] - 2000
            
            # Create decade feature
            result_df['decade'] = (result_df[date_col] // 10) * 10
            
            # Flag for pre/post antiretroviral therapy era (approx 1996)
            result_df['post_art_era'] = result_df[date_col] > 1996
            
            # Flag for pre/post 90-90-90 UNAIDS targets (introduced in 2014)
            result_df['post_90_90_90_target'] = result_df[date_col] >= 2014
        
        return result_df
    
    def create_geographical_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create features based on geographical information.
        
        Args:
            df: DataFrame containing county information
            
        Returns:
            DataFrame with additional geographical features
        """
        if df.empty or 'county' not in df.columns:
            return df
            
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Define region mappings for Kenya
        region_mapping = {
            # Coast
            'Mombasa': 'Coast',
            'Kwale': 'Coast',
            'Kilifi': 'Coast',
            'Tana River': 'Coast',
            'Lamu': 'Coast',
            'Taita Taveta': 'Coast',
            
            # North Eastern
            'Garissa': 'North Eastern',
            'Wajir': 'North Eastern',
            'Mandera': 'North Eastern',
            
            # Eastern
            'Marsabit': 'Eastern',
            'Isiolo': 'Eastern',
            'Meru': 'Eastern',
            'Tharaka Nithi': 'Eastern',
            'Embu': 'Eastern',
            'Kitui': 'Eastern',
            'Machakos': 'Eastern',
            'Makueni': 'Eastern',
            
            # Central
            'Nyandarua': 'Central',
            'Nyeri': 'Central',
            'Kirinyaga': 'Central',
            'Murang\'a': 'Central',
            'Kiambu': 'Central',
            
            # Rift Valley
            'Turkana': 'Rift Valley',
            'West Pokot': 'Rift Valley',
            'Samburu': 'Rift Valley',
            'Trans Nzoia': 'Rift Valley',
            'Uasin Gishu': 'Rift Valley',
            'Elgeyo Marakwet': 'Rift Valley',
            'Nandi': 'Rift Valley',
            'Baringo': 'Rift Valley',
            'Laikipia': 'Rift Valley',
            'Nakuru': 'Rift Valley',
            'Narok': 'Rift Valley',
            'Kajiado': 'Rift Valley',
            'Kericho': 'Rift Valley',
            'Bomet': 'Rift Valley',
            
            # Western
            'Kakamega': 'Western',
            'Vihiga': 'Western',
            'Bungoma': 'Western',
            'Busia': 'Western',
            
            # Nyanza
            'Siaya': 'Nyanza',
            'Kisumu': 'Nyanza',
            'Homa Bay': 'Nyanza',
            'Migori': 'Nyanza',
            'Kisii': 'Nyanza',
            'Nyamira': 'Nyanza',
            
            # Nairobi
            'Nairobi': 'Nairobi'
        }
        
        # Add region feature
        result_df['region'] = result_df['county'].map(region_mapping)
        
        # Define urban/rural classification (simplified example)
        urban_counties = ['Nairobi', 'Mombasa', 'Kisumu', 'Nakuru', 'Eldoret', 'Thika']
        result_df['urban_rural'] = result_df['county'].apply(
            lambda x: 'Urban' if x in urban_counties else 'Rural'
        )
        
        # Add high prevalence flag based on historical data
        high_prevalence_counties = [
            'Homa Bay', 'Kisumu', 'Siaya', 'Migori', 'Nyamira', 
            'Kisii', 'Nairobi', 'Mombasa'
        ]
        result_df['high_prevalence_area'] = result_df['county'].isin(high_prevalence_counties)
        
        return result_df
    
    def create_rate_change_features(self, df: pd.DataFrame, rate_cols: List[str]) -> pd.DataFrame:
        """
        Create features that capture changes in rates over time.
        
        Args:
            df: DataFrame containing rate data over time
            rate_cols: List of column names containing rate data
            
        Returns:
            DataFrame with additional rate change features
        """
        if df.empty or 'year' not in df.columns or 'county' not in df.columns:
            return df
            
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Sort by county and year
        result_df = result_df.sort_values(['county', 'year'])
        
        # Calculate year-over-year changes for each rate column
        for col in rate_cols:
            if col in result_df.columns:
                result_df[f'{col}_yoy_change'] = result_df.groupby('county')[col].diff()
                
                # Calculate percentage change
                result_df[f'{col}_yoy_pct_change'] = result_df.groupby('county')[col].pct_change() * 100
                
                # Calculate 3-year moving average
                result_df[f'{col}_3yr_avg'] = result_df.groupby('county')[col].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
                
                # Calculate if the current value is above or below the 3-year trend
                result_df[f'{col}_above_trend'] = result_df[col] > result_df[f'{col}_3yr_avg']
        
        return result_df
    
    def create_indicators_pivot(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Pivot the indicators dataset to have indicators as columns.
        
        Args:
            df: DataFrame containing indicators in long format
            
        Returns:
            DataFrame with indicators pivoted to columns
        """
        if df.empty or not all(col in df.columns for col in ['county', 'year', 'indicator', 'value']):
            return df
            
        # Create a copy to avoid modifying the original
        result_df = df.copy()
        
        # Pivot the indicators
        pivoted = result_df.pivot_table(
            index=['county', 'year'],
            columns='indicator',
            values='value',
            aggfunc='first'
        ).reset_index()
        
        # Rename columns to prevent spaces and special characters
        pivoted.columns = [col.lower().replace(' ', '_').replace('-', '_').replace('/', '_').replace('%', 'pct')
                          if isinstance(col, str) else col for col in pivoted.columns]
        
        return pivoted
    
    def process_features(self, df: pd.DataFrame, is_indicators: bool = False) -> pd.DataFrame:
        """
        Process all features for a dataset.
        
        Args:
            df: DataFrame to process
            is_indicators: Whether the DataFrame is the indicators dataset in long format
            
        Returns:
            Processed DataFrame with all engineered features
        """
        if df.empty:
            return df
            
        # If it's the indicators dataset, pivot it first
        if is_indicators and all(col in df.columns for col in ['county', 'year', 'indicator', 'value']):
            df = self.create_indicators_pivot(df)
        
        # Create time features
        df = self.create_time_features(df)
        
        # Create geographical features
        df = self.create_geographical_features(df)
        
        # Create rate change features for relevant columns
        rate_cols = [col for col in df.columns if any(term in col.lower() for term in 
                                                    ['prevalence', 'rate', 'coverage'])]
        if rate_cols:
            df = self.create_rate_change_features(df, rate_cols)
        
        return df


if __name__ == "__main__":
    # This code runs when the script is executed directly
    import os
    from src.data.data_loader import DataLoader
    from src.data.data_cleaner import DataCleaner
    
    # Load and clean the data
    loader = DataLoader()
    raw_datasets = loader.load_all_datasets()
    
    cleaner = DataCleaner()
    cleaned_datasets = cleaner.clean_all_datasets(raw_datasets)
    
    # Feature engineering
    engineer = FeatureEngineer()
    
    # Process each dataset
    for name, df in cleaned_datasets.items():
        if not df.empty:
            is_indicators = name == 'indicators'
            processed_df = engineer.process_features(df, is_indicators)
            
            # Save the processed dataset
            output_path = f"data/processed/{name}_processed.csv"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            processed_df.to_csv(output_path, index=False)
            print(f"Saved processed {name} data to {output_path}")