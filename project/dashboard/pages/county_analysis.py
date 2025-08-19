"""
County analysis page for the AfyaScope Kenya Streamlit dashboard.

This page provides detailed information about HIV/STI trends for specific counties.
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add the project root to the path so we can import from src
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

from src.visualization.plots import PlotGenerator
from src.utils.helpers import calculate_prevalence_change


def app():
    """Run the county analysis page."""
    st.title("County-Level Analysis")
    
    # Check if data is loaded
    if not st.session_state.get("loaded_data", False):
        st.warning("Please load data from the main page first.")
        return
    
    # Get the data
    prevalence_df = st.session_state.cleaned_datasets.get('prevalence', pd.DataFrame())
    indicators_df = st.session_state.cleaned_datasets.get('indicators', pd.DataFrame())
    
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
        
        # Additional indicators if available
        if not indicators_df.empty:
            st.subheader("Additional Health Indicators")
            
            # Filter indicators for this county
            county_indicators = indicators_df[indicators_df['county'] == selected_county]
            
            if not county_indicators.empty:
                # Let user select year
                indicator_year = st.selectbox(
                    "Select year for indicators:",
                    options=sorted(county_indicators['year'].unique()),
                    key="indicator_year"
                )
                
                # Filter by selected year
                year_indicators = county_indicators[county_indicators['year'] == indicator_year]
                
                if not year_indicators.empty:
                    # Display indicators
                    col1, col2 = st.columns(2)
                    
                    # Sort indicators by value
                    sorted_indicators = year_indicators.sort_values('value', ascending=False)
                    
                    # Split into two columns
                    half_point = len(sorted_indicators) // 2
                    
                    with col1:
                        for _, row in sorted_indicators.iloc[:half_point].iterrows():
                            st.markdown(f"""
                            <div class="metric-card">
                                <h5>{row['indicator']}</h5>
                                <p>{row['value']:.2f}</p>
                            </div>
                            """, unsafe_allow_html=True)
                    
                    with col2:
                        for _, row in sorted_indicators.iloc[half_point:].iterrows():
                            st.markdown(f"""
                            <div class="metric-card">
                                <h5>{row['indicator']}</h5>
                                <p>{row['value']:.2f}</p>
                            </div>
                            """, unsafe_allow_html=True)


if __name__ == "__main__":
    app()