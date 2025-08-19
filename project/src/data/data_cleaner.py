"""
Module for cleaning and preprocessing the datasets.
"""
import os
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union


class DataCleaner:
    """Class for cleaning and preprocessing datasets."""
    
    def __init__(self):
        """Initialize the DataCleaner."""
        self.kenya_counties = [
            'Baringo', 'Bomet', 'Bungoma', 'Busia', 'Elgeyo Marakwet', 
            'Embu', 'Garissa', 'Homa Bay', 'Isiolo', 'Kajiado', 
            'Kakamega', 'Kericho', 'Kiambu', 'Kilifi', 'Kirinyaga', 
            'Kisii', 'Kisumu', 'Kitui', 'Kwale', 'Laikipia', 
            'Lamu', 'Machakos', 'Makueni', 'Mandera', 'Marsabit', 
            'Meru', 'Migori', 'Mombasa', 'Murang\'a', 'Nairobi', 
            'Nakuru', 'Nandi', 'Narok', 'Nyamira', 'Nyandarua', 
            'Nyeri', 'Samburu', 'Siaya', 'Taita Taveta', 'Tana River', 
            'Tharaka Nithi', 'Trans Nzoia', 'Turkana', 'Uasin Gishu', 
            'Vihiga', 'Wajir', 'West Pokot'
        ]
    
    def clean_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the indicators dataset.
        
        Args:
            df: Raw indicators DataFrame
            
        Returns:
            Cleaned indicators DataFrame
        """
        if df.empty:
            return df
            
        # Make a copy to avoid modifying the original
        cleaned_df = df.copy()
        
        # Standardize column names
        cleaned_df.columns = cleaned_df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Standardize county names
        if 'county' in cleaned_df.columns:
            cleaned_df['county'] = cleaned_df['county'].str.strip().str.title()
            
            # Correct any inconsistent county names
            county_mapping = {
                # Add any county name corrections here, for example:
                'Nairobi County': 'Nairobi',
                'Mombasa County': 'Mombasa',
                # Add more as needed
            }
            
            cleaned_df['county'] = cleaned_df['county'].replace(county_mapping)
            
            # Filter to only include valid Kenya counties
            cleaned_df = cleaned_df[cleaned_df['county'].isin(self.kenya_counties)]
        
        # Convert year to numeric
        if 'year' in cleaned_df.columns:
            cleaned_df['year'] = pd.to_numeric(cleaned_df['year'], errors='coerce')
            cleaned_df = cleaned_df.dropna(subset=['year'])
            cleaned_df['year'] = cleaned_df['year'].astype(int)
        
        # Convert value to numeric
        if 'value' in cleaned_df.columns:
            cleaned_df['value'] = pd.to_numeric(cleaned_df['value'], errors='coerce')
        
        # Clean indicator names if present
        if 'indicator' in cleaned_df.columns:
            cleaned_df['indicator'] = cleaned_df['indicator'].str.strip()
        
        return cleaned_df
    
    def clean_prevalence(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the prevalence dataset.
        
        Args:
            df: Raw prevalence DataFrame
            
        Returns:
            Cleaned prevalence DataFrame
        """
        if df.empty:
            return df
            
        # Make a copy to avoid modifying the original
        cleaned_df = df.copy()
        
        # Standardize column names
        cleaned_df.columns = cleaned_df.columns.str.strip().str.lower().str.replace(' ', '_').str.replace('(%)', '')
        
        # Convert percentage columns to float values
        percentage_columns = ['hiv_prevalence', 'art_coverage']
        for col in percentage_columns:
            if col in cleaned_df.columns:
                # Remove % sign if present
                if cleaned_df[col].dtype == 'object':
                    cleaned_df[col] = cleaned_df[col].astype(str).str.replace('%', '')
                    
                # Convert to float
                cleaned_df[col] = pd.to_numeric(cleaned_df[col], errors='coerce')
                
                # Normalize values > 1 (assume they're already in percentage form)
                mask = cleaned_df[col] > 1
                cleaned_df.loc[mask, col] = cleaned_df.loc[mask, col] / 100
        
        # Standardize county names
        if 'county' in cleaned_df.columns:
            cleaned_df['county'] = cleaned_df['county'].str.strip().str.title()
            
            # Filter to only include valid Kenya counties
            cleaned_df = cleaned_df[cleaned_df['county'].isin(self.kenya_counties)]
        
        # Convert year to numeric
        if 'year' in cleaned_df.columns:
            cleaned_df['year'] = pd.to_numeric(cleaned_df['year'], errors='coerce')
            cleaned_df = cleaned_df.dropna(subset=['year'])
            cleaned_df['year'] = cleaned_df['year'].astype(int)
        
        # Convert STI rate to numeric
        if 'sti_rate' in cleaned_df.columns:
            cleaned_df['sti_rate'] = pd.to_numeric(cleaned_df['sti_rate'], errors='coerce')
        
        return cleaned_df
    
    def clean_unaids_facts(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the UNAIDS facts dataset.
        
        Args:
            df: Raw UNAIDS facts DataFrame
            
        Returns:
            Cleaned UNAIDS facts DataFrame
        """
        if df.empty:
            return df
            
        # Make a copy to avoid modifying the original
        cleaned_df = df.copy()
        
        # Standardize column names
        cleaned_df.columns = cleaned_df.columns.str.strip().str.lower().str.replace(' ', '_')
        
        # Convert year to numeric
        if 'year' in cleaned_df.columns:
            cleaned_df['year'] = pd.to_numeric(cleaned_df['year'], errors='coerce')
            cleaned_df = cleaned_df.dropna(subset=['year'])
            cleaned_df['year'] = cleaned_df['year'].astype(int)
        
        # Clean text fields
        text_columns = ['key_facts', 'highlights']
        for col in text_columns:
            if col in cleaned_df.columns:
                cleaned_df[col] = cleaned_df[col].fillna('').astype(str).str.strip()
        
        return cleaned_df
    
    def merge_datasets(self, datasets: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Merge the indicators and prevalence datasets.
        
        Args:
            datasets: Dictionary of DataFrames with keys 'indicators', 'prevalence'
            
        Returns:
            Merged DataFrame containing data from both sources
        """
        indicators_df = datasets.get('indicators', pd.DataFrame())
        prevalence_df = datasets.get('prevalence', pd.DataFrame())
        
        if indicators_df.empty or prevalence_df.empty:
            return pd.DataFrame()
        
        # Clean the datasets first
        indicators_clean = self.clean_indicators(indicators_df)
        prevalence_clean = self.clean_prevalence(prevalence_df)
        
        # Merge on county and year
        merged_df = pd.merge(
            prevalence_clean,
            indicators_clean,
            on=['county', 'year'],
            how='outer'
        )
        
        return merged_df
    
    def clean_all_datasets(self, datasets: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        """
        Clean all datasets.
        
        Args:
            datasets: Dictionary of raw DataFrames
            
        Returns:
            Dictionary of cleaned DataFrames
        """
        return {
            'indicators': self.clean_indicators(datasets.get('indicators', pd.DataFrame())),
            'prevalence': self.clean_prevalence(datasets.get('prevalence', pd.DataFrame())),
            'unaids_facts': self.clean_unaids_facts(datasets.get('unaids_facts', pd.DataFrame())),
            'merged': self.merge_datasets(datasets)
        }


if __name__ == "__main__":
    # This code runs when the script is executed directly
    from data_loader import DataLoader
    
    # Load the data
    loader = DataLoader()
    raw_datasets = loader.load_all_datasets()
    
    # Clean the data
    cleaner = DataCleaner()
    cleaned_datasets = cleaner.clean_all_datasets(raw_datasets)
    
    # Save the cleaned datasets
    for name, df in cleaned_datasets.items():
        if not df.empty:
            output_path = f"data/processed/{name}_cleaned.csv"
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            df.to_csv(output_path, index=False)
            print(f"Saved cleaned {name} data to {output_path}")