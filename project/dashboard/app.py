"""
Main Streamlit dashboard application for AfyaScope Kenya project.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import folium
from streamlit_folium import folium_static
import sys
import os
from datetime import datetime

# Add the project root to the path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.data.data_loader import DataLoader
from src.data.data_cleaner import DataCleaner
from src.features.feature_engineering import FeatureEngineer
from src.models.time_series import TimeSeriesForecaster
from src.models.classification import RiskClassifier
from src.visualization.maps import MapGenerator
from src.visualization.plots import PlotGenerator
from src.utils.helpers import calculate_summary_statistics, extract_counties_from_data, extract_years_from_data, calculate_prevalence_change

# Page configuration
st.set_page_config(
    page_title="AfyaScope Kenya: HIV & STI Trends and Predictions",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Apply custom CSS for styling
st.markdown("""
<style>
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        padding-left: 2rem;
        padding-right: 2rem;
    }
    h1, h2, h3 {
        color: #20B2AA;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 24px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: transparent;
        border-radius: 4px 4px 0px 0px;
        color: #20B2AA;
        padding-left: 16px;
        padding-right: 16px;
    }
    .stTabs [aria-selected="true"] {
        background-color: rgba(32, 178, 170, 0.1);
        border-bottom: 2px solid #20B2AA;
    }
    [data-testid="stSidebar"] {
        background-color: #F0F8FF;
    }
    [data-testid="stMetric"] {
        background-color: rgba(32, 178, 170, 0.1);
        padding: 10px;
        border-radius: 5px;
    }
    .css-12w0qpk {
        background-color: #F8F4FF;
        border-radius: 5px;
        padding: 1rem;
    }
    .metric-card {
        background-color: white;
        border-radius: 5px;
        padding: 15px;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
    }
    .insight-card {
        background-color: white;
        border-radius: 5px;
        padding: 15px;
        margin-bottom: 1rem;
        box-shadow: 0 0 10px rgba(0,0,0,0.1);
    }
    .high-alert {
        color: #B22222;
        font-weight: bold;
    }
    .medium-alert {
        color: #FF6347;
        font-weight: bold;
    }
    .positive-trend {
        color: #008080;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state variables if they don't exist
if 'loaded_data' not in st.session_state:
    st.session_state.loaded_data = False
if 'datasets' not in st.session_state:
    st.session_state.datasets = {}
if 'cleaned_datasets' not in st.session_state:
    st.session_state.cleaned_datasets = {}
if 'processed_datasets' not in st.session_state:
    st.session_state.processed_datasets = {}
if 'counties' not in st.session_state:
    st.session_state.counties = []
if 'years' not in st.session_state:
    st.session_state.years = []


def load_data():
    """Load, clean and process all datasets."""
    with st.spinner("Loading data..."):
        # Initialize the data loader
        loader = DataLoader()
        
        # Load raw datasets
        st.session_state.datasets = loader.load_all_datasets()
        
        # Check if data was loaded successfully
        if all(df.empty for df in st.session_state.datasets.values()):
            st.error("Failed to load data. Please check that the data files exist in the data/raw directory.")
            return False
        
        # Initialize the data cleaner
        cleaner = DataCleaner()
        
        # Clean the datasets
        st.session_state.cleaned_datasets = cleaner.clean_all_datasets(st.session_state.datasets)
        
        # Initialize the feature engineer
        engineer = FeatureEngineer()
        
        # Process each dataset
        st.session_state.processed_datasets = {}
        for name, df in st.session_state.cleaned_datasets.items():
            if not df.empty:
                is_indicators = name == 'indicators'
                st.session_state.processed_datasets[name] = engineer.process_features(df, is_indicators)
        
        # Extract counties and years from the prevalence dataset
        prevalence_df = st.session_state.cleaned_datasets.get('prevalence', pd.DataFrame())
        if not prevalence_df.empty:
            st.session_state.counties = extract_counties_from_data(prevalence_df)
            st.session_state.years = extract_years_from_data(prevalence_df)
            
            # Set the data as loaded
            st.session_state.loaded_data = True
            return True
        else:
            st.error("Failed to process prevalence data.")
            return False


def display_overview():
    """Display the overview dashboard."""
    st.title("AfyaScope Kenya: HIV & STI Overview")
    
    # Display key facts from the latest year in UNAIDS facts
    unaids_df = st.session_state.cleaned_datasets.get('unaids_facts', pd.DataFrame())
    if not unaids_df.empty:
        latest_year = unaids_df['year'].max()
        latest_facts = unaids_df[unaids_df['year'] == latest_year]
        
        if not latest_facts.empty:
            st.subheader(f"Key Facts ({latest_year})")
            
            with st.container():
                st.markdown(f"""
                <div class="insight-card">
                    <h4>Highlights</h4>
                    <p>{latest_facts['highlights'].values[0]}</p>
                </div>
                """, unsafe_allow_html=True)
                
                st.markdown(f"""
                <div class="insight-card">
                    <h4>Key Facts</h4>
                    <p>{latest_facts['key_facts'].values[0]}</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Display summary statistics
    prevalence_df = st.session_state.cleaned_datasets.get('prevalence', pd.DataFrame())
    if not prevalence_df.empty:
        latest_year = prevalence_df['year'].max()
        latest_data = prevalence_df[prevalence_df['year'] == latest_year]
        
        st.subheader(f"Summary Statistics ({latest_year})")
        
        col1, col2, col3 = st.columns(3)
        
        # HIV Prevalence
        hiv_stats = calculate_summary_statistics(latest_data, 'hiv_prevalence')
        if not hiv_stats.empty:
            with col1:
                st.metric("Average HIV Prevalence", f"{hiv_stats['mean'].values[0]:.2%}")
                st.markdown(f"""
                <div class="metric-card">
                    <p>Min: {hiv_stats['min'].values[0]:.2%}</p>
                    <p>Max: {hiv_stats['max'].values[0]:.2%}</p>
                    <p>Standard Deviation: {hiv_stats['std'].values[0]:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # STI Rate
        sti_stats = calculate_summary_statistics(latest_data, 'sti_rate')
        if not sti_stats.empty:
            with col2:
                st.metric("Average STI Rate", f"{sti_stats['mean'].values[0]:.2f}")
                st.markdown(f"""
                <div class="metric-card">
                    <p>Min: {sti_stats['min'].values[0]:.2f}</p>
                    <p>Max: {sti_stats['max'].values[0]:.2f}</p>
                    <p>Standard Deviation: {sti_stats['std'].values[0]:.2f}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # ART Coverage
        art_stats = calculate_summary_statistics(latest_data, 'art_coverage')
        if not art_stats.empty and 'art_coverage' in latest_data.columns:
            with col3:
                st.metric("Average ART Coverage", f"{art_stats['mean'].values[0]:.2%}")
                st.markdown(f"""
                <div class="metric-card">
                    <p>Min: {art_stats['min'].values[0]:.2%}</p>
                    <p>Max: {art_stats['max'].values[0]:.2%}</p>
                    <p>Standard Deviation: {art_stats['std'].values[0]:.2%}</p>
                </div>
                """, unsafe_allow_html=True)
        
        # Create a choropleth map of HIV prevalence
        st.subheader(f"HIV Prevalence by County ({latest_year})")
        
        map_gen = MapGenerator()
        folium_map = map_gen.create_choropleth_map(
            prevalence_df,
            'hiv_prevalence',
            latest_year,
            f"HIV Prevalence by County ({latest_year})"
        )
        
        # Display the map
        folium_static(folium_map, width=800, height=500)
        
        # Add a note about the map
        st.markdown("""
        *Note: The map shows HIV prevalence by county. Darker colors indicate higher prevalence rates.*
        """)
        
        # Display trends over time for selected counties
        st.subheader("HIV Prevalence Trends")
        
        # Get top 5 counties by HIV prevalence
        top_counties = latest_data.sort_values('hiv_prevalence', ascending=False).head(5)['county'].tolist()
        
        # Allow user to select counties for comparison
        selected_counties = st.multiselect(
            "Select counties to compare:",
            options=st.session_state.counties,
            default=top_counties[:3]
        )
        
        if selected_counties:
            # Initialize plot generator
            plotter = PlotGenerator(theme='teal')
            
            # Create and display the plot
            fig = plotter.plot_trend_by_county(
                prevalence_df,
                'hiv_prevalence',
                selected_counties,
                title="HIV Prevalence Trends by County",
                ylabel="HIV Prevalence (%)"
            )
            
            if fig:
                st.pyplot(fig)
                plt.close(fig)
            
            # Show insights about the trends
            st.subheader("Key Insights")
            
            insights_col1, insights_col2 = st.columns(2)
            
            with insights_col1:
                st.markdown("<h4>Prevalence Change</h4>", unsafe_allow_html=True)
                
                for county in selected_counties:
                    change_data = calculate_prevalence_change(
                        prevalence_df,
                        county,
                        'hiv_prevalence',
                        None,
                        None
                    )
                    
                    if not all(pd.isna(val) for val in change_data.values()):
                        change_class = "positive-trend" if change_data['absolute_change'] < 0 else "medium-alert"
                        
                        st.markdown(f"""
                        <div class="insight-card">
                            <h5>{county}</h5>
                            <p>From {change_data['start_year']} to {change_data['end_year']}:</p>
                            <p>Initial: {change_data['start_value']:.2%}</p>
                            <p>Current: {change_data['end_value']:.2%}</p>
                            <p>Change: <span class="{change_class}">{change_data['absolute_change']:.2%}</span></p>
                            <p>Percentage Change: <span class="{change_class}">{change_data['percentage_change']:.2f}%</span></p>
                        </div>
                        """, unsafe_allow_html=True)
            
            with insights_col2:
                st.markdown("<h4>Regional Patterns</h4>", unsafe_allow_html=True)
                
                # Get region information from the feature-engineered data
                processed_df = st.session_state.processed_datasets.get('prevalence', pd.DataFrame())
                
                if not processed_df.empty and 'region' in processed_df.columns:
                    # Calculate regional averages
                    region_stats = processed_df[processed_df['year'] == latest_year].groupby('region')['hiv_prevalence'].agg(['mean', 'std']).reset_index()
                    
                    # Sort by mean prevalence
                    region_stats = region_stats.sort_values('mean', ascending=False)
                    
                    for _, row in region_stats.iterrows():
                        alert_class = "high-alert" if row['mean'] > 0.1 else "medium-alert" if row['mean'] > 0.05 else ""
                        
                        st.markdown(f"""
                        <div class="insight-card">
                            <h5>{row['region']}</h5>
                            <p>Average Prevalence: <span class="{alert_class}">{row['mean']:.2%}</span></p>
                            <p>Standard Deviation: {row['std']:.2%}</p>
                        </div>
                        """, unsafe_allow_html=True)


def display_county_analysis():
    """Display county-level analysis dashboard."""
    st.title("County-Level Analysis")
    
    # Get the data
    prevalence_df = st.session_state.cleaned_datasets.get('prevalence', pd.DataFrame())
    processed_df = st.session_state.processed_datasets.get('prevalence', pd.DataFrame())
    
    if prevalence_df.empty:
        st.error("Prevalence data is not available.")
        return
    
    # County selection
    selected_county = st.selectbox(
        "Select a county:",
        options=st.session_state.counties
    )
    
    if selected_county:
        # Filter data for the selected county
        county_data = prevalence_df[prevalence_df['county'] == selected_county]
        
        if county_data.empty:
            st.warning(f"No data available for {selected_county}.")
            return
        
        # Display county information
        st.subheader(f"{selected_county} County Overview")
        
        latest_year = county_data['year'].max()
        latest_data = county_data[county_data['year'] == latest_year]
        
        col1, col2, col3 = st.columns(3)
        
        # HIV Prevalence
        if 'hiv_prevalence' in latest_data.columns:
            with col1:
                st.metric(
                    "HIV Prevalence",
                    f"{latest_data['hiv_prevalence'].values[0]:.2%}"
                )
        
        # STI Rate
        if 'sti_rate' in latest_data.columns:
            with col2:
                st.metric(
                    "STI Rate",
                    f"{latest_data['sti_rate'].values[0]:.2f}"
                )
        
        # ART Coverage
        if 'art_coverage' in latest_data.columns:
            with col3:
                st.metric(
                    "ART Coverage",
                    f"{latest_data['art_coverage'].values[0]:.2%}"
                )
        
        # Display trends over time
        st.subheader("Trends Over Time")
        
        trend_tab1, trend_tab2 = st.tabs(["HIV Prevalence", "STI Rate"])
        
        with trend_tab1:
            plotter = PlotGenerator(theme='teal')
            fig = plotter.plot_trend_by_county(
                prevalence_df,
                'hiv_prevalence',
                [selected_county],
                title=f"HIV Prevalence Trend in {selected_county}",
                ylabel="HIV Prevalence (%)"
            )
            
            if fig:
                st.pyplot(fig)
                plt.close(fig)
                
                # Calculate change
                change_data = calculate_prevalence_change(
                    prevalence_df,
                    selected_county,
                    'hiv_prevalence',
                    None,
                    None
                )
                
                if not all(pd.isna(val) for val in change_data.values()):
                    change_direction = "decreased" if change_data['absolute_change'] < 0 else "increased"
                    
                    st.markdown(f"""
                    <div class="insight-card">
                        <h4>HIV Prevalence Change</h4>
                        <p>From {change_data['start_year']} to {change_data['end_year']}, HIV prevalence in {selected_county} has {change_direction} by {abs(change_data['absolute_change']):.2%} (from {change_data['start_value']:.2%} to {change_data['end_value']:.2%}).</p>
                        <p>This represents a {abs(change_data['percentage_change']):.2f}% {change_direction} over this period.</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        with trend_tab2:
            fig = plotter.plot_trend_by_county(
                prevalence_df,
                'sti_rate',
                [selected_county],
                title=f"STI Rate Trend in {selected_county}",
                ylabel="STI Rate"
            )
            
            if fig:
                st.pyplot(fig)
                plt.close(fig)
                
                # Calculate change
                change_data = calculate_prevalence_change(
                    prevalence_df,
                    selected_county,
                    'sti_rate',
                    None,
                    None
                )
                
                if not all(pd.isna(val) for val in change_data.values()):
                    change_direction = "decreased" if change_data['absolute_change'] < 0 else "increased"
                    
                    st.markdown(f"""
                    <div class="insight-card">
                        <h4>STI Rate Change</h4>
                        <p>From {change_data['start_year']} to {change_data['end_year']}, STI rate in {selected_county} has {change_direction} by {abs(change_data['absolute_change']):.2f} (from {change_data['start_value']:.2f} to {change_data['end_value']:.2f}).</p>
                        <p>This represents a {abs(change_data['percentage_change']):.2f}% {change_direction} over this period.</p>
                    </div>
                    """, unsafe_allow_html=True)
        
        # Compare with other counties
        st.subheader("Comparison with Other Counties")
        
        comparison_year = st.select_slider(
            "Select year for comparison:",
            options=sorted(county_data['year'].unique())
        )
        
        if comparison_year:
            year_data = prevalence_df[prevalence_df['year'] == comparison_year]
            
            # Create a horizontal bar chart comparing counties
            fig = plotter.plot_county_comparison(
                year_data,
                'hiv_prevalence',
                comparison_year,
                top_n=10,
                title=f"Top 10 Counties by HIV Prevalence in {comparison_year}",
                xlabel="HIV Prevalence (%)"
            )
            
            if fig:
                st.pyplot(fig)
                plt.close(fig)
            
            # Show where the selected county ranks
            all_counties = year_data.sort_values('hiv_prevalence', ascending=False)
            county_rank = all_counties[all_counties['county'] == selected_county].index[0] + 1
            total_counties = len(all_counties)
            
            st.markdown(f"""
            <div class="insight-card">
                <h4>County Ranking</h4>
                <p>{selected_county} ranks <strong>#{county_rank}</strong> out of {total_counties} counties in terms of HIV prevalence in {comparison_year}.</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Show additional indicators if available
            indicators_df = st.session_state.cleaned_datasets.get('indicators', pd.DataFrame())
            
            if not indicators_df.empty:
                county_indicators = indicators_df[(indicators_df['county'] == selected_county) & (indicators_df['year'] == comparison_year)]
                
                if not county_indicators.empty:
                    st.subheader("Additional Health Indicators")
                    
                    # Show top 5 indicators
                    top_indicators = county_indicators.sort_values('value', ascending=False).head(5)
                    
                    if not top_indicators.empty:
                        for _, row in top_indicators.iterrows():
                            st.markdown(f"""
                            <div class="metric-card">
                                <h5>{row['indicator']}</h5>
                                <p>{row['value']:.2f}</p>
                            </div>
                            """, unsafe_allow_html=True)


def display_predictions():
    """Display predictions dashboard."""
    st.title("HIV & STI Predictions")
    
    # Get the data
    prevalence_df = st.session_state.cleaned_datasets.get('prevalence', pd.DataFrame())
    
    if prevalence_df.empty:
        st.error("Prevalence data is not available.")
        return
    
    # County selection
    selected_county = st.selectbox(
        "Select a county:",
        options=st.session_state.counties
    )
    
    if selected_county:
        # Initialize the forecaster
        forecaster = TimeSeriesForecaster()
        
        # Create tabs for different prediction types
        pred_tab1, pred_tab2 = st.tabs(["HIV Prevalence Forecast", "STI Rate Forecast"])
        
        with pred_tab1:
            st.subheader(f"HIV Prevalence Forecast for {selected_county}")
            
            # Prepare Prophet data
            prophet_data = forecaster.prepare_prophet_data(
                prevalence_df,
                selected_county,
                'hiv_prevalence'
            )
            
            if prophet_data.empty:
                st.warning(f"Not enough data to generate forecast for {selected_county}.")
            else:
                # Generate forecast
                _, forecast = forecaster.fit_prophet_model(
                    prophet_data,
                    future_periods=5,
                    yearly_seasonality=True,
                    growth='logistic'
                )
                
                if forecast.empty:
                    st.warning("Failed to generate forecast.")
                else:
                    # Plot the forecast
                    plotter = PlotGenerator(theme='teal')
                    fig = plotter.plot_forecast(
                        forecast,
                        selected_county,
                        'hiv_prevalence',
                        title=f"HIV Prevalence Forecast for {selected_county}",
                        ylabel="HIV Prevalence (%)"
                    )
                    
                    if fig:
                        st.pyplot(fig)
                        plt.close(fig)
                    
                    # Extract and display key insights
                    last_actual = prophet_data['y'].iloc[-1]
                    forecast_end = forecast[forecast['ds'] == forecast['ds'].max()]
                    last_forecast = forecast_end['yhat'].values[0]
                    
                    change = last_forecast - last_actual
                    percent_change = (change / last_actual) * 100
                    
                    trend_class = "positive-trend" if change < 0 else "medium-alert"
                    trend_direction = "decrease" if change < 0 else "increase"
                    
                    st.markdown(f"""
                    <div class="insight-card">
                        <h4>Forecast Insights</h4>
                        <p>Current HIV Prevalence: {last_actual:.2%}</p>
                        <p>Forecasted HIV Prevalence (5 years): {last_forecast:.2%}</p>
                        <p>Projected Change: <span class="{trend_class}">{change:.2%} ({trend_direction})</span></p>
                        <p>Percentage Change: <span class="{trend_class}">{percent_change:.2f}%</span></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show forecast table
                    with st.expander("Show Forecast Details"):
                        # Get only future years (exclude historical)
                        future_forecast = forecast[forecast['ds'] > prophet_data['ds'].max()]
                        
                        # Format the forecast table
                        forecast_table = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                        forecast_table.columns = ['Year', 'Forecast', 'Lower Bound', 'Upper Bound']
                        forecast_table['Year'] = forecast_table['Year'].dt.year
                        
                        # Format as percentages
                        forecast_table['Forecast'] = forecast_table['Forecast'].apply(lambda x: f"{x:.2%}")
                        forecast_table['Lower Bound'] = forecast_table['Lower Bound'].apply(lambda x: f"{x:.2%}")
                        forecast_table['Upper Bound'] = forecast_table['Upper Bound'].apply(lambda x: f"{x:.2%}")
                        
                        st.table(forecast_table)
        
        with pred_tab2:
            st.subheader(f"STI Rate Forecast for {selected_county}")
            
            # Prepare Prophet data
            prophet_data = forecaster.prepare_prophet_data(
                prevalence_df,
                selected_county,
                'sti_rate'
            )
            
            if prophet_data.empty:
                st.warning(f"Not enough data to generate forecast for {selected_county}.")
            else:
                # Generate forecast
                _, forecast = forecaster.fit_prophet_model(
                    prophet_data,
                    future_periods=5,
                    yearly_seasonality=True,
                    growth='linear'
                )
                
                if forecast.empty:
                    st.warning("Failed to generate forecast.")
                else:
                    # Plot the forecast
                    plotter = PlotGenerator(theme='purple')
                    fig = plotter.plot_forecast(
                        forecast,
                        selected_county,
                        'sti_rate',
                        title=f"STI Rate Forecast for {selected_county}",
                        ylabel="STI Rate"
                    )
                    
                    if fig:
                        st.pyplot(fig)
                        plt.close(fig)
                    
                    # Extract and display key insights
                    last_actual = prophet_data['y'].iloc[-1]
                    forecast_end = forecast[forecast['ds'] == forecast['ds'].max()]
                    last_forecast = forecast_end['yhat'].values[0]
                    
                    change = last_forecast - last_actual
                    percent_change = (change / last_actual) * 100
                    
                    trend_class = "positive-trend" if change < 0 else "medium-alert"
                    trend_direction = "decrease" if change < 0 else "increase"
                    
                    st.markdown(f"""
                    <div class="insight-card">
                        <h4>Forecast Insights</h4>
                        <p>Current STI Rate: {last_actual:.2f}</p>
                        <p>Forecasted STI Rate (5 years): {last_forecast:.2f}</p>
                        <p>Projected Change: <span class="{trend_class}">{change:.2f} ({trend_direction})</span></p>
                        <p>Percentage Change: <span class="{trend_class}">{percent_change:.2f}%</span></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    # Show forecast table
                    with st.expander("Show Forecast Details"):
                        # Get only future years (exclude historical)
                        future_forecast = forecast[forecast['ds'] > prophet_data['ds'].max()]
                        
                        # Format the forecast table
                        forecast_table = future_forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].copy()
                        forecast_table.columns = ['Year', 'Forecast', 'Lower Bound', 'Upper Bound']
                        forecast_table['Year'] = forecast_table['Year'].dt.year
                        
                        # Format as numbers
                        forecast_table['Forecast'] = forecast_table['Forecast'].apply(lambda x: f"{x:.2f}")
                        forecast_table['Lower Bound'] = forecast_table['Lower Bound'].apply(lambda x: f"{x:.2f}")
                        forecast_table['Upper Bound'] = forecast_table['Upper Bound'].apply(lambda x: f"{x:.2f}")
                        
                        st.table(forecast_table)
        
        # Risk Classification
        st.subheader("County Risk Classification")
        
        # Get processed data
        processed_df = st.session_state.processed_datasets.get('prevalence', pd.DataFrame())
        
        if not processed_df.empty:
            # Initialize the risk classifier
            classifier = RiskClassifier(threshold=0.05)  # 5% HIV prevalence threshold
            
            # Prepare the data
            X, y = classifier.prepare_data(processed_df)
            
            if not X.empty and not y.empty:
                with st.spinner("Training risk classification model..."):
                    # Train the model
                    classifier.train_model(X, y)
                    
                    # Make predictions
                    risk_predictions = classifier.predict_risk(processed_df)
                    
                    # Calculate metrics
                    metrics = classifier.evaluate_model(X, y)
                    
                    if metrics:
                        # Display model performance
                        st.markdown(f"""
                        <div class="insight-card">
                            <h4>Risk Classification Model Performance</h4>
                            <p>ROC AUC: {metrics['roc_auc']:.4f}</p>
                            <p>Accuracy: {metrics['classification_report']['accuracy']:.4f}</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Display feature importance
                        st.subheader("Risk Factors")
                        
                        fig = plotter.plot_feature_importance(
                            metrics['feature_importance'],
                            title="Most Important Risk Factors"
                        )
                        
                        if fig:
                            st.pyplot(fig)
                            plt.close(fig)
                        
                        # Show county risk prediction
                        county_risk = risk_predictions[risk_predictions['county'] == selected_county]
                        
                        if not county_risk.empty:
                            risk_level = county_risk['risk_level'].values[0]
                            risk_prob = county_risk['risk_probability'].values[0]
                            
                            risk_class = "high-alert" if risk_level == "High Risk" else ""
                            
                            st.markdown(f"""
                            <div class="insight-card">
                                <h4>Risk Assessment for {selected_county}</h4>
                                <p>Risk Level: <span class="{risk_class}">{risk_level}</span></p>
                                <p>Risk Probability: {risk_prob:.2f}</p>
                                <p>This assessment is based on multiple factors including HIV prevalence, STI rates, healthcare access, and regional patterns.</p>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            # Create a map of risk categories
                            map_gen = MapGenerator()
                            risk_cat_col = 'risk_category'
                            year_data = risk_predictions[risk_predictions['year'] == risk_predictions['year'].max()]
                            
                            if not year_data.empty and risk_cat_col in year_data.columns:
                                st.subheader("Risk Classification Map")
                                
                                folium_map = map_gen.create_choropleth_map(
                                    year_data,
                                    risk_cat_col,
                                    year_data['year'].max(),
                                    "HIV Risk Classification by County"
                                )
                                
                                folium_static(folium_map, width=800, height=500)
                                
                                st.markdown("""
                                *Note: The map shows risk classification by county. Darker colors indicate higher risk areas.*
                                """)


def display_insights():
    """Display insights dashboard."""
    st.title("Key Insights and Recommendations")
    
    # Get the data
    prevalence_df = st.session_state.cleaned_datasets.get('prevalence', pd.DataFrame())
    indicators_df = st.session_state.cleaned_datasets.get('indicators', pd.DataFrame())
    processed_df = st.session_state.processed_datasets.get('prevalence', pd.DataFrame())
    
    if prevalence_df.empty:
        st.error("Prevalence data is not available.")
        return
    
    # Calculate national trends
    latest_year = prevalence_df['year'].max()
    earliest_year = prevalence_df['year'].min()
    
    national_trends = prevalence_df.groupby('year')['hiv_prevalence'].mean().reset_index()
    
    if not national_trends.empty:
        # Calculate overall change
        initial_prevalence = national_trends[national_trends['year'] == earliest_year]['hiv_prevalence'].values[0]
        current_prevalence = national_trends[national_trends['year'] == latest_year]['hiv_prevalence'].values[0]
        
        absolute_change = current_prevalence - initial_prevalence
        percent_change = (absolute_change / initial_prevalence) * 100
        
        change_direction = "decreased" if absolute_change < 0 else "increased"
        trend_class = "positive-trend" if absolute_change < 0 else "medium-alert"
        
        # National trends
        st.subheader("National HIV Trends")
        
        st.markdown(f"""
        <div class="insight-card">
            <h4>Overall HIV Prevalence Trend ({earliest_year} - {latest_year})</h4>
            <p>National HIV prevalence has <span class="{trend_class}">{change_direction} by {abs(absolute_change):.2%}</span> over the past {latest_year - earliest_year} years.</p>
            <p>From {initial_prevalence:.2%} in {earliest_year} to {current_prevalence:.2%} in {latest_year}.</p>
            <p>This represents a <span class="{trend_class}">{abs(percent_change):.2f}% {change_direction}</span> over this period.</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Plot national trend
        plotter = PlotGenerator(theme='teal')
        fig = plt.figure(figsize=(10, 6))
        plt.plot(national_trends['year'], national_trends['hiv_prevalence'], marker='o', linewidth=2)
        plt.title('National HIV Prevalence Trend')
        plt.xlabel('Year')
        plt.ylabel('HIV Prevalence (%)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        st.pyplot(fig)
        plt.close(fig)
    
    # High burden counties
    st.subheader("Priority Focus Areas")
    
    latest_data = prevalence_df[prevalence_df['year'] == latest_year]
    
    if not latest_data.empty:
        # Get top 5 high prevalence counties
        high_burden = latest_data.sort_values('hiv_prevalence', ascending=False).head(5)
        
        st.markdown("<h4>Highest HIV Burden Counties</h4>", unsafe_allow_html=True)
        
        burden_col1, burden_col2 = st.columns(2)
        
        with burden_col1:
            for _, row in high_burden.iterrows():
                st.markdown(f"""
                <div class="insight-card">
                    <h5>{row['county']}</h5>
                    <p>HIV Prevalence: <span class="high-alert">{row['hiv_prevalence']:.2%}</span></p>
                    <p>STI Rate: {row['sti_rate']:.2f}</p>
                    {f"<p>ART Coverage: {row['art_coverage']:.2%}</p>" if 'art_coverage' in row else ""}
                </div>
                """, unsafe_allow_html=True)
        
        with burden_col2:
            # Create a pie chart of the top counties
            fig = plt.figure(figsize=(8, 8))
            plt.pie(high_burden['hiv_prevalence'], labels=high_burden['county'], autopct='%1.1f%%',
                  shadow=True, startangle=90, colors=plt.cm.Reds(np.linspace(0.5, 0.8, len(high_burden))))
            plt.axis('equal')
            plt.title('Share of HIV Burden in Top 5 Counties')
            
            st.pyplot(fig)
            plt.close(fig)
    
    # Regional patterns
    if not processed_df.empty and 'region' in processed_df.columns:
        st.subheader("Regional Patterns")
        
        latest_processed = processed_df[processed_df['year'] == latest_year]
        
        # Calculate regional averages
        region_stats = latest_processed.groupby('region')['hiv_prevalence'].agg(['mean', 'std', 'count']).reset_index()
        region_stats = region_stats.sort_values('mean', ascending=False)
        
        # Plot regional patterns
        fig = plt.figure(figsize=(10, 6))
        bars = plt.bar(region_stats['region'], region_stats['mean'], yerr=region_stats['std'], 
                    capsize=5, alpha=0.8, color=plt.cm.viridis(np.linspace(0, 0.8, len(region_stats))))
        plt.title('HIV Prevalence by Region')
        plt.xlabel('Region')
        plt.ylabel('HIV Prevalence (%)')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        st.pyplot(fig)
        plt.close(fig)
        
        # Display regional insights
        high_region = region_stats.iloc[0]['region']
        low_region = region_stats.iloc[-1]['region']
        
        st.markdown(f"""
        <div class="insight-card">
            <h4>Regional Disparities</h4>
            <p>The <span class="high-alert">{high_region}</span> region has the highest average HIV prevalence at {region_stats.iloc[0]['mean']:.2%}.</p>
            <p>The <span class="positive-trend">{low_region}</span> region has the lowest average HIV prevalence at {region_stats.iloc[-1]['mean']:.2%}.</p>
            <p>This represents a {(region_stats.iloc[0]['mean'] - region_stats.iloc[-1]['mean']):.2%} difference between the highest and lowest regions.</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Key recommendations
    st.subheader("Recommendations")
    
    st.markdown("""
    <div class="insight-card">
        <h4>Targeted Interventions</h4>
        <ol>
            <li>Focus resources on high-burden counties with prevalence rates above 10%</li>
            <li>Implement comprehensive prevention programs in counties showing increasing trends</li>
            <li>Strengthen ART coverage in counties with low coverage rates</li>
            <li>Scale up STI screening and treatment services in areas with high STI rates</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-card">
        <h4>Policy Recommendations</h4>
        <ol>
            <li>Develop region-specific policies addressing unique challenges and risk factors</li>
            <li>Increase funding for HIV/STI programs in high-burden counties</li>
            <li>Improve data collection and monitoring systems for better tracking of trends</li>
            <li>Promote integration of HIV and STI services into primary healthcare</li>
            <li>Enhance community engagement and education programs in high-risk areas</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-card">
        <h4>Future Research Priorities</h4>
        <ol>
            <li>Investigate social and economic factors driving regional disparities</li>
            <li>Study the effectiveness of current intervention programs</li>
            <li>Explore the relationship between HIV and other sexually transmitted infections</li>
            <li>Evaluate the impact of healthcare accessibility on prevalence rates</li>
            <li>Research behavioral factors contributing to transmission in high-prevalence areas</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)


def main():
    """Main function to run the Streamlit app."""
    # Add a logo and title in the sidebar
    st.sidebar.image("https://img.icons8.com/color/96/000000/heart-health.png", width=80)
    st.sidebar.title("AfyaScope Kenya")
    st.sidebar.subheader("HIV & STI Trends and Predictions")
    
    # Load data button
    if not st.session_state.loaded_data:
        if st.sidebar.button("Load Data"):
            success = load_data()
            if success:
                st.sidebar.success("Data loaded successfully!")
            else:
                st.sidebar.error("Failed to load data.")
    
    # Show navigation menu only if data is loaded
    if st.session_state.loaded_data:
        # Navigation menu
        pages = {
            "Overview": display_overview,
            "County Analysis": display_county_analysis,
            "Predictions": display_predictions,
            "Insights & Recommendations": display_insights
        }
        
        selection = st.sidebar.radio("Navigation", list(pages.keys()))
        
        # Display the selected page
        pages[selection]()
        
        # Add information about the latest update
        st.sidebar.markdown("---")
        st.sidebar.info(f"Last updated: {datetime.now().strftime('%Y-%m-%d')}")
    else:
        # Display welcome message if data is not loaded
        st.title("Welcome to AfyaScope Kenya")
        
        st.markdown("""
        <div style="text-align: center; padding: 2rem 0;">
            <img src="https://img.icons8.com/color/240/000000/heart-health.png" width="120">
            <h2 style="margin-top: 1rem;">HIV & STI Trends and Predictions</h2>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        This dashboard provides analysis and visualization of HIV/AIDS and STI trends in Kenya. It includes:
        
        - Overview of the HIV/STI situation in Kenya
        - County-level analysis and comparisons
        - Temporal trends and patterns
        - Future predictions based on historical data
        - Key insights from UNAIDS reports
        
        To begin, please click the "Load Data" button in the sidebar.
        """)
        
        st.info("Note: This application requires the following data files in the 'data/raw' directory: indicators.csv, prevalence.csv, and unaids_facts.csv.")


if __name__ == "__main__":
    main()