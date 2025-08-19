"""
Module for creating plots and charts for visualization.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Union
import os


class PlotGenerator:
    """Class for generating plots and visualizations."""
    
    def __init__(self, 
                theme: str = 'teal',
                figsize: Tuple[int, int] = (10, 6),
                dpi: int = 100):
        """
        Initialize the PlotGenerator.
        
        Args:
            theme: Color theme ('teal', 'purple', or 'red')
            figsize: Default figure size
            dpi: Default figure resolution
        """
        self.figsize = figsize
        self.dpi = dpi
        
        # Set up color palettes
        self.themes = {
            'teal': {
                'primary': '#20B2AA',  # Light Sea Green
                'secondary': '#5F9EA0',  # Cadet Blue
                'tertiary': '#008080',  # Teal
                'highlight': '#00CED1',  # Dark Turquoise
                'background': '#F0F8FF',  # Alice Blue
            },
            'purple': {
                'primary': '#9370DB',  # Medium Purple
                'secondary': '#8A2BE2',  # Blue Violet
                'tertiary': '#483D8B',  # Dark Slate Blue
                'highlight': '#7B68EE',  # Medium Slate Blue
                'background': '#F8F4FF',  # Light Lavender
            },
            'red': {
                'primary': '#FF6347',  # Tomato
                'secondary': '#CD5C5C',  # Indian Red
                'tertiary': '#B22222',  # Fire Brick
                'highlight': '#FF7F50',  # Coral
                'background': '#FFF5F5',  # Light Pink
            }
        }
        
        # Set the theme
        self.set_theme(theme)
        
    def set_theme(self, theme: str) -> None:
        """
        Set the color theme for plots.
        
        Args:
            theme: Color theme name
        """
        if theme in self.themes:
            self.theme = self.themes[theme]
        else:
            # Default to teal theme
            self.theme = self.themes['teal']
            
        # Set the matplotlib style
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Set custom style elements
        plt.rcParams['figure.figsize'] = self.figsize
        plt.rcParams['figure.dpi'] = self.dpi
        plt.rcParams['axes.facecolor'] = self.theme['background']
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.labelcolor'] = '#333333'
        plt.rcParams['axes.titlecolor'] = '#333333'
        plt.rcParams['font.family'] = 'sans-serif'
        plt.rcParams['font.sans-serif'] = ['Arial', 'Helvetica', 'DejaVu Sans']
    
    def plot_trend_by_county(self, 
                            df: pd.DataFrame,
                            target_col: str,
                            counties: List[str],
                            title: Optional[str] = None,
                            ylabel: Optional[str] = None,
                            save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot trends over time for selected counties.
        
        Args:
            df: DataFrame containing time series data
            target_col: Column to plot
            counties: List of counties to include
            title: Plot title
            ylabel: Y-axis label
            save_path: Path to save the figure
            
        Returns:
            Matplotlib Figure object
        """
        if df.empty or 'year' not in df.columns or target_col not in df.columns:
            return None
            
        # Filter for the specified counties
        county_df = df[df['county'].isin(counties)].copy()
        
        if county_df.empty:
            return None
            
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Plot each county
        palette = sns.color_palette("viridis", n_colors=len(counties))
        
        for i, county in enumerate(counties):
            county_data = county_df[county_df['county'] == county]
            if not county_data.empty:
                ax.plot(
                    county_data['year'], 
                    county_data[target_col],
                    marker='o',
                    linestyle='-',
                    linewidth=2,
                    markersize=8,
                    label=county,
                    color=palette[i]
                )
        
        # Set title and labels
        if title:
            ax.set_title(title, fontsize=16, pad=20)
        else:
            ax.set_title(f"{target_col.replace('_', ' ').title()} Trends by County", fontsize=16, pad=20)
            
        ax.set_xlabel("Year", fontsize=12, labelpad=10)
        
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=12, labelpad=10)
        else:
            ax.set_ylabel(target_col.replace('_', ' ').title(), fontsize=12, labelpad=10)
        
        # Customize the plot
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add legend
        ax.legend(title="County", title_fontsize=12, fontsize=10, 
                 loc='best', frameon=True, framealpha=0.9)
        
        # Add a subtle background color
        fig.patch.set_facecolor(self.theme['background'])
        ax.set_facecolor(self.theme['background'])
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure if a path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig
    
    def plot_forecast(self,
                     forecast_df: pd.DataFrame,
                     county: str,
                     target: str,
                     title: Optional[str] = None,
                     ylabel: Optional[str] = None,
                     save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot time series forecast from Prophet.
        
        Args:
            forecast_df: Prophet forecast DataFrame
            county: County name for the title
            target: Target variable name
            title: Plot title
            ylabel: Y-axis label
            save_path: Path to save the figure
            
        Returns:
            Matplotlib Figure object
        """
        if forecast_df.empty or 'ds' not in forecast_df.columns:
            return None
            
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Plot actual values
        ax.plot(
            forecast_df['ds'], 
            forecast_df['yhat'],
            color=self.theme['primary'],
            linestyle='-',
            linewidth=2,
            label='Forecast'
        )
        
        # Plot prediction intervals
        ax.fill_between(
            forecast_df['ds'],
            forecast_df['yhat_lower'],
            forecast_df['yhat_upper'],
            color=self.theme['primary'],
            alpha=0.2,
            label='95% Confidence Interval'
        )
        
        # If 'y' (actual values) is in the dataframe, plot them too
        if 'y' in forecast_df.columns:
            # Filter to only show actual values (not NaN)
            actuals = forecast_df[~forecast_df['y'].isna()]
            ax.scatter(
                actuals['ds'],
                actuals['y'],
                color=self.theme['highlight'],
                s=50,
                zorder=5,
                label='Actual Values'
            )
        
        # Set title and labels
        if title:
            ax.set_title(title, fontsize=16, pad=20)
        else:
            ax.set_title(f"{target.replace('_', ' ').title()} Forecast for {county}", fontsize=16, pad=20)
            
        ax.set_xlabel("Year", fontsize=12, labelpad=10)
        
        if ylabel:
            ax.set_ylabel(ylabel, fontsize=12, labelpad=10)
        else:
            ax.set_ylabel(target.replace('_', ' ').title(), fontsize=12, labelpad=10)
        
        # Customize the plot
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add a vertical line to separate historical and future periods
        if 'y' in forecast_df.columns:
            last_actual = forecast_df[~forecast_df['y'].isna()]['ds'].max()
            if not pd.isna(last_actual):
                ax.axvline(
                    x=last_actual,
                    color='gray',
                    linestyle='--',
                    linewidth=1.5,
                    alpha=0.7,
                    label='Forecast Start'
                )
        
        # Add legend
        ax.legend(loc='best', frameon=True, framealpha=0.9, fontsize=10)
        
        # Add a subtle background color
        fig.patch.set_facecolor(self.theme['background'])
        ax.set_facecolor(self.theme['background'])
        
        # Format x-axis dates
        plt.xticks(rotation=45)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure if a path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig
    
    def plot_correlation_heatmap(self,
                                df: pd.DataFrame,
                                columns: Optional[List[str]] = None,
                                title: Optional[str] = None,
                                save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot correlation heatmap for selected columns.
        
        Args:
            df: DataFrame containing data
            columns: List of columns to include in correlation
            title: Plot title
            save_path: Path to save the figure
            
        Returns:
            Matplotlib Figure object
        """
        if df.empty:
            return None
            
        # Select columns for correlation
        if columns is None:
            # Select numeric columns only
            columns = df.select_dtypes(include=[np.number]).columns.tolist()
        else:
            # Filter to ensure all columns exist and are numeric
            columns = [col for col in columns if col in df.columns 
                      and np.issubdtype(df[col].dtype, np.number)]
        
        if not columns:
            return None
            
        # Calculate correlation matrix
        corr_matrix = df[columns].corr()
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Plot heatmap
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        sns.heatmap(
            corr_matrix,
            mask=mask,
            cmap=cmap,
            vmax=1,
            vmin=-1,
            center=0,
            square=True,
            linewidths=.5,
            cbar_kws={"shrink": .8},
            annot=True,
            fmt=".2f",
            ax=ax
        )
        
        # Set title
        if title:
            ax.set_title(title, fontsize=16, pad=20)
        else:
            ax.set_title("Correlation Heatmap", fontsize=16, pad=20)
        
        # Make column labels more readable
        plt.xticks(rotation=45, ha='right', fontsize=10)
        plt.yticks(fontsize=10)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure if a path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig
    
    def plot_county_comparison(self,
                              df: pd.DataFrame,
                              target_col: str,
                              year: int,
                              top_n: int = 10,
                              title: Optional[str] = None,
                              xlabel: Optional[str] = None,
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot horizontal bar chart comparing counties for a specific year.
        
        Args:
            df: DataFrame containing data
            target_col: Column to compare
            year: Year to filter for
            top_n: Number of top counties to show
            title: Plot title
            xlabel: X-axis label
            save_path: Path to save the figure
            
        Returns:
            Matplotlib Figure object
        """
        if df.empty or 'county' not in df.columns or target_col not in df.columns:
            return None
            
        # Filter for the specified year
        year_df = df[df['year'] == year].copy()
        
        if year_df.empty:
            return None
            
        # Sort and get top counties
        top_counties = year_df.sort_values(target_col, ascending=False).head(top_n)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Plot horizontal bar chart
        bars = ax.barh(
            top_counties['county'],
            top_counties[target_col],
            color=self.theme['primary'],
            alpha=0.8,
            height=0.6
        )
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + 0.01,
                bar.get_y() + bar.get_height()/2,
                f'{width:.2f}',
                ha='left',
                va='center',
                fontsize=10
            )
        
        # Set title and labels
        if title:
            ax.set_title(title, fontsize=16, pad=20)
        else:
            ax.set_title(f"Top {top_n} Counties by {target_col.replace('_', ' ').title()} in {year}", fontsize=16, pad=20)
            
        if xlabel:
            ax.set_xlabel(xlabel, fontsize=12, labelpad=10)
        else:
            ax.set_xlabel(target_col.replace('_', ' ').title(), fontsize=12, labelpad=10)
            
        ax.set_ylabel("County", fontsize=12, labelpad=10)
        
        # Customize the plot
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add a subtle background color
        fig.patch.set_facecolor(self.theme['background'])
        ax.set_facecolor(self.theme['background'])
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure if a path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig
    
    def plot_feature_importance(self,
                              importance_df: pd.DataFrame,
                              title: Optional[str] = None,
                              save_path: Optional[str] = None) -> plt.Figure:
        """
        Plot feature importance from a model.
        
        Args:
            importance_df: DataFrame with feature importance (columns: 'feature', 'importance')
            title: Plot title
            save_path: Path to save the figure
            
        Returns:
            Matplotlib Figure object
        """
        if importance_df.empty or 'feature' not in importance_df.columns or 'importance' not in importance_df.columns:
            return None
            
        # Sort by importance
        sorted_df = importance_df.sort_values('importance', ascending=True).tail(15)
        
        # Create figure
        fig, ax = plt.subplots(figsize=self.figsize, dpi=self.dpi)
        
        # Plot horizontal bar chart
        bars = ax.barh(
            sorted_df['feature'],
            sorted_df['importance'],
            color=self.theme['tertiary'],
            alpha=0.8,
            height=0.6
        )
        
        # Add value labels
        for bar in bars:
            width = bar.get_width()
            ax.text(
                width + 0.01,
                bar.get_y() + bar.get_height()/2,
                f'{width:.3f}',
                ha='left',
                va='center',
                fontsize=10
            )
        
        # Set title and labels
        if title:
            ax.set_title(title, fontsize=16, pad=20)
        else:
            ax.set_title("Feature Importance", fontsize=16, pad=20)
            
        ax.set_xlabel("Importance", fontsize=12, labelpad=10)
        ax.set_ylabel("Feature", fontsize=12, labelpad=10)
        
        # Customize the plot
        ax.grid(True, axis='x', alpha=0.3, linestyle='--')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        
        # Add a subtle background color
        fig.patch.set_facecolor(self.theme['background'])
        ax.set_facecolor(self.theme['background'])
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the figure if a path is provided
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig


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
        # Initialize plot generator
        plotter = PlotGenerator(theme='teal')
        
        # Example: Plot HIV prevalence trends for selected counties
        top_counties = ['Nairobi', 'Mombasa', 'Kisumu']
        fig = plotter.plot_trend_by_county(
            prevalence_df,
            'hiv_prevalence',
            top_counties,
            title="HIV Prevalence Trends in Major Counties",
            ylabel="HIV Prevalence (%)",
            save_path="data/visualizations/hiv_prevalence_trends.png"
        )
        
        # Example: Plot county comparison for a specific year
        fig = plotter.plot_county_comparison(
            prevalence_df,
            'hiv_prevalence',
            2020,  # Use a year that exists in your data
            top_n=10,
            title="Top 10 Counties by HIV Prevalence in 2020",
            xlabel="HIV Prevalence (%)",
            save_path="data/visualizations/county_comparison_2020.png"
        )