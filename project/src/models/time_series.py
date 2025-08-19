"""
Module for time series forecasting of HIV/STI trends.
"""
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from prophet import Prophet
from prophet.diagnostics import cross_validation, performance_metrics


class TimeSeriesForecaster:
    """Class for forecasting HIV/STI trends over time."""
    
    def __init__(self):
        """Initialize the TimeSeriesForecaster."""
        pass
    
    def prepare_prophet_data(self, df: pd.DataFrame, 
                            county: str,
                            target_col: str,
                            date_col: str = 'year') -> pd.DataFrame:
        """
        Prepare data for Prophet forecasting.
        
        Args:
            df: DataFrame containing time series data
            county: County to filter for
            target_col: Column to forecast
            date_col: Column containing date information
            
        Returns:
            DataFrame formatted for Prophet
        """
        if df.empty or county not in df['county'].values:
            return pd.DataFrame()
            
        # Filter for the specified county
        county_df = df[df['county'] == county].copy()
        
        # Select required columns and rename for Prophet
        prophet_df = county_df[[date_col, target_col]].rename(
            columns={date_col: 'ds', target_col: 'y'}
        )
        
        # If ds is just a year, convert to datetime
        if prophet_df['ds'].dtype == 'int64':
            prophet_df['ds'] = pd.to_datetime(prophet_df['ds'], format='%Y')
        
        return prophet_df
    
    def fit_prophet_model(self, data: pd.DataFrame,
                        future_periods: int = 5,
                        yearly_seasonality: bool = True,
                        growth: str = 'linear') -> Tuple[Prophet, pd.DataFrame]:
        """
        Fit Prophet model and generate forecasts.
        
        Args:
            data: DataFrame formatted for Prophet (with 'ds' and 'y' columns)
            future_periods: Number of years to forecast
            yearly_seasonality: Whether to include yearly seasonality
            growth: Growth pattern ('linear' or 'logistic')
            
        Returns:
            Tuple of (trained Prophet model, forecast DataFrame)
        """
        if data.empty:
            return None, pd.DataFrame()
        
        # Initialize Prophet model
        model = Prophet(
            yearly_seasonality=yearly_seasonality,
            growth=growth,
            # Set bounded growth for prevalence rates
            interval_width=0.95,  # 95% prediction intervals
        )
        
        # Add a capacity (upper bound) for logistic growth if needed
        if growth == 'logistic':
            data['cap'] = 1.0  # Assuming target is a prevalence rate (0-1)
            
        # Fit the model
        model.fit(data)
        
        # Create future dataframe
        last_date = data['ds'].max()
        if isinstance(last_date, pd.Timestamp):
            future = model.make_future_dataframe(
                periods=future_periods,
                freq='Y'  # Annual frequency
            )
        else:
            # If we're working with just years, create a manual future df
            max_year = data['ds'].max().year if isinstance(data['ds'].max(), pd.Timestamp) else int(data['ds'].max())
            future_years = pd.DataFrame({
                'ds': pd.date_range(start=f'{max_year+1}-01-01', periods=future_periods, freq='Y')
            })
            future = pd.concat([data[['ds']], future_years])
        
        # Add capacity for logistic growth
        if growth == 'logistic':
            future['cap'] = 1.0
            
        # Make forecast
        forecast = model.predict(future)
        
        return model, forecast
    
    def forecast_county_rates(self, df: pd.DataFrame,
                            target_cols: List[str],
                            counties: Optional[List[str]] = None,
                            future_periods: int = 5) -> Dict[str, Dict[str, pd.DataFrame]]:
        """
        Generate forecasts for multiple counties and target columns.
        
        Args:
            df: DataFrame containing time series data
            target_cols: List of columns to forecast
            counties: List of counties to forecast for (if None, uses all counties)
            future_periods: Number of years to forecast
            
        Returns:
            Nested dictionary of forecasts: {county: {target: forecast_df}}
        """
        if df.empty or 'county' not in df.columns:
            return {}
            
        # Use all counties if none specified
        if counties is None:
            counties = df['county'].unique().tolist()
            
        results = {}
        
        for county in counties:
            county_results = {}
            
            for target in target_cols:
                if target in df.columns:
                    # Prepare data for this county and target
                    prophet_data = self.prepare_prophet_data(df, county, target)
                    
                    if not prophet_data.empty and len(prophet_data) >= 3:  # Need at least 3 points
                        # Determine growth type based on target
                        growth = 'logistic' if 'prevalence' in target.lower() or 'rate' in target.lower() else 'linear'
                        
                        # Fit model and generate forecast
                        _, forecast = self.fit_prophet_model(
                            prophet_data,
                            future_periods=future_periods,
                            growth=growth
                        )
                        
                        # Add county information to forecast
                        if not forecast.empty:
                            forecast['county'] = county
                            forecast['target'] = target
                            county_results[target] = forecast
            
            if county_results:
                results[county] = county_results
        
        return results
    
    def evaluate_forecast_accuracy(self, df: pd.DataFrame,
                                target_col: str,
                                county: str,
                                cv_periods: int = 3) -> pd.DataFrame:
        """
        Evaluate forecast accuracy using cross-validation.
        
        Args:
            df: DataFrame containing time series data
            target_col: Column to forecast
            county: County to filter for
            cv_periods: Number of periods to use for cross-validation
            
        Returns:
            DataFrame containing performance metrics
        """
        # Prepare data
        prophet_data = self.prepare_prophet_data(df, county, target_col)
        
        if prophet_data.empty or len(prophet_data) <= cv_periods:
            return pd.DataFrame()
            
        # Define growth type
        growth = 'logistic' if 'prevalence' in target_col.lower() or 'rate' in target_col.lower() else 'linear'
        
        # Initialize and fit model
        model = Prophet(
            yearly_seasonality=True,
            growth=growth,
            interval_width=0.95
        )
        
        # Add capacity for logistic growth
        if growth == 'logistic':
            prophet_data['cap'] = 1.0
            
        model.fit(prophet_data)
        
        # Perform cross-validation
        df_cv = cross_validation(
            model,
            initial='3 years',
            period='1 year',
            horizon='1 year',
            parallel='processes'
        )
        
        # Calculate performance metrics
        df_performance = performance_metrics(df_cv)
        
        # Add county and target information
        df_performance['county'] = county
        df_performance['target'] = target_col
        
        return df_performance


if __name__ == "__main__":
    # This code runs when the script is executed directly
    import os
    from src.data.data_loader import DataLoader
    from src.data.data_cleaner import DataCleaner
    
    # Load and clean data
    loader = DataLoader()
    cleaner = DataCleaner()
    
    raw_datasets = loader.load_all_datasets()
    cleaned_datasets = cleaner.clean_all_datasets(raw_datasets)
    
    # Get the prevalence dataset
    prevalence_df = cleaned_datasets.get('prevalence', pd.DataFrame())
    
    if not prevalence_df.empty:
        # Initialize forecaster
        forecaster = TimeSeriesForecaster()
        
        # Target columns to forecast
        targets = ['hiv_prevalence', 'sti_rate']
        
        # Sample counties
        sample_counties = ['Nairobi', 'Mombasa', 'Kisumu']
        
        # Generate forecasts
        forecasts = forecaster.forecast_county_rates(
            prevalence_df,
            target_cols=targets,
            counties=sample_counties,
            future_periods=5
        )
        
        # Save forecasts
        for county, county_forecasts in forecasts.items():
            for target, forecast_df in county_forecasts.items():
                output_path = f"data/processed/forecasts_{county}_{target}.csv"
                os.makedirs(os.path.dirname(output_path), exist_ok=True)
                forecast_df.to_csv(output_path, index=False)
                print(f"Saved {target} forecast for {county} to {output_path}")