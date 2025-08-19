"""
Predictions page for the AfyaScope Kenya Streamlit dashboard.

This page provides forecasts and risk predictions for HIV/STI trends.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from streamlit_folium import folium_static

# Add the project root to the path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.models.time_series import TimeSeriesForecaster
from src.models.classification import RiskClassifier
from src.visualization.maps import MapGenerator
from src.visualization.plots import PlotGenerator


def app():
    """Run the predictions page."""
    st.title("HIV & STI Predictions")
    
    # Check if data is loaded
    if not st.session_state.get("loaded_data", False):
        st.warning("Please load data from the main page first.")
        return
    
    # Get the data
    prevalence_df = st.session_state.cleaned_datasets.get('prevalence', pd.DataFrame())
    processed_df = st.session_state.processed_datasets.get('prevalence', pd.DataFrame())
    
    if prevalence_df.empty:
        st.error("Prevalence data is not available.")
        return
    
    # Create tabs for different types of predictions
    tab1, tab2 = st.tabs(["County Forecasts", "Risk Classification"])
    
    with tab1:
        st.subheader("County-Level Forecasts")
        
        # County selection
        selected_county = st.selectbox(
            "Select a county:",
            options=st.session_state.counties,
            key="forecast_county"
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
    
    with tab2:
        st.subheader("Risk Classification")
        
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
                        
                        plotter = PlotGenerator(theme='teal')
                        fig = plotter.plot_feature_importance(
                            metrics['feature_importance'],
                            title="Most Important Risk Factors"
                        )
                        
                        if fig:
                            st.pyplot(fig)
                            plt.close(fig)
                        
                        # County selection for risk assessment
                        selected_county = st.selectbox(
                            "Select a county for risk assessment:",
                            options=st.session_state.counties,
                            key="risk_county"
                        )
                        
                        if selected_county:
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
                        st.subheader("Risk Classification Map")
                        
                        # Set the year for the map
                        latest_year = processed_df['year'].max()
                        map_year = st.select_slider(
                            "Select year for risk map:",
                            options=sorted(processed_df['year'].unique()),
                            value=latest_year
                        )
                        
                        if map_year:
                            # Filter data for the selected year
                            year_data = risk_predictions[risk_predictions['year'] == map_year]
                            
                            if not year_data.empty:
                                map_gen = MapGenerator()
                                risk_cat_col = 'risk_category'
                                
                                if risk_cat_col in year_data.columns:
                                    folium_map = map_gen.create_choropleth_map(
                                        year_data,
                                        risk_cat_col,
                                        map_year,
                                        f"HIV Risk Classification by County ({map_year})"
                                    )
                                    
                                    folium_static(folium_map, width=800, height=500)
                                    
                                    st.markdown("""
                                    *Note: The map shows risk classification by county. Darker colors indicate higher risk areas.*
                                    """)
                                    
                                    # Count high-risk vs low-risk counties
                                    high_risk = year_data[year_data[risk_cat_col] == 1].shape[0]
                                    low_risk = year_data[year_data[risk_cat_col] == 0].shape[0]
                                    
                                    st.markdown(f"""
                                    <div class="insight-card">
                                        <h4>Risk Distribution ({map_year})</h4>
                                        <p><span class="high-alert">High Risk Counties:</span> {high_risk}</p>
                                        <p><span class="positive-trend">Low Risk Counties:</span> {low_risk}</p>
                                    </div>
                                    """, unsafe_allow_html=True)


if __name__ == "__main__":
    app()