"""
Module for loading raw datasets from various sources.
"""
import os
import pandas as pd
from typing import Dict, Optional, Union


class DataLoader:
    """Class for loading datasets from files."""
    
    def __init__(self, data_dir: str = "data/raw"):
        """
        Initialize the DataLoader.
        
        Args:
            data_dir: Directory containing the raw data files
        """
        self.data_dir = data_dir
        
    def load_indicators(self) -> pd.DataFrame:
        """
        Load the indicators dataset.
        
        Returns:
            DataFrame containing HIV/STI indicators by county and year
        """
        file_path = os.path.join(self.data_dir, "indicators.csv")
        try:
            df = pd.read_csv(file_path)
            print(f"Loaded indicators data with shape: {df.shape}")
            return df
        except FileNotFoundError:
            print(f"Error: File {file_path} not found.")
            return pd.DataFrame()
            
    def load_prevalence(self) -> pd.DataFrame:
        """
        Load the prevalence dataset.
        
        Returns:
            DataFrame containing HIV and STI prevalence rates by county and year
        """
        file_path = os.path.join(self.data_dir, "prevalence.csv")
        try:
            df = pd.read_csv(file_path)
            print(f"Loaded prevalence data with shape: {df.shape}")
            return df
        except FileNotFoundError:
            print(f"Error: File {file_path} not found.")
            return pd.DataFrame()
            
    def load_unaids_facts(self) -> pd.DataFrame:
        """
        Load the UNAIDS fact sheets dataset.
        
        Returns:
            DataFrame containing extracted summaries from UNAIDS fact sheets
        """
        file_path = os.path.join(self.data_dir, "unaids_facts.csv")
        try:
            df = pd.read_csv(file_path)
            print(f"Loaded UNAIDS facts data with shape: {df.shape}")
            return df
        except FileNotFoundError:
            print(f"Error: File {file_path} not found.")
            return pd.DataFrame()
    
    def load_all_datasets(self) -> Dict[str, pd.DataFrame]:
        """
        Load all datasets.
        
        Returns:
            Dictionary of DataFrames with keys 'indicators', 'prevalence', 'unaids_facts'
        """
        return {
            'indicators': self.load_indicators(),
            'prevalence': self.load_prevalence(),
            'unaids_facts': self.load_unaids_facts()
        }


if __name__ == "__main__":
    # Example usage
    loader = DataLoader()
    datasets = loader.load_all_datasets()
    
    # Print dataset info
    for name, df in datasets.items():
        if not df.empty:
            print(f"\nDataset: {name}")
            print(f"Shape: {df.shape}")
            print(f"Columns: {df.columns.tolist()}")
            print(f"Sample data:\n{df.head(2)}")