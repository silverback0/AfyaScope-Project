"""
Overview page for the AfyaScope Kenya Streamlit dashboard.

This page provides a high-level summary of HIV/STI trends in Kenya.
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

from src.visualization.maps import MapGenerator
from src.visualization.plots import PlotGenerator
from src.utils.helpers import calculate_summary_statistics, calculate_prevalence_change


def app():
    """Run the overview page."""
    st.title("National HIV/STI Overview")
    
    # Check if data is loaded
    if not st.session_state.get("loaded_data", False):
        st.warning("Please load data from the main page first.")
        return
    
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
        
        # National trends over time
        st.subheader("National HIV Prevalence Trend")
        
        # Calculate national averages by year
        national_trend = prevalence_df.groupby('year')['hiv_prevalence'].mean().reset_index()
        
        # Plot the trend
        fig = plt.figure(figsize=(10, 6))
        plt.plot(national_trend['year'], national_trend['hiv_prevalence'], marker='o', linewidth=2, color='#20B2AA')
        plt.title('National HIV Prevalence Trend')
        plt.xlabel('Year')
        plt.ylabel('HIV Prevalence (%)')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        st.pyplot(fig)
        plt.close(fig)
        
        # Calculate change over time
        if len(national_trend) >= 2:
            start_year = national_trend['year'].min()
            end_year = national_trend['year'].max()
            
            start_val = national_trend[national_trend['year'] == start_year]['hiv_prevalence'].values[0]
            end_val = national_trend[national_trend['year'] == end_year]['hiv_prevalence'].values[0]
            
            change = end_val - start_val
            percent_change = (change / start_val) * 100
            
            trend_direction = "decreased" if change < 0 else "increased"
            trend_class = "positive-trend" if change < 0 else "medium-alert"
            
            st.markdown(f"""
            <div class="insight-card">
                <h4>National HIV Prevalence Change</h4>
                <p>From {start_year} to {end_year}, the national HIV prevalence has <span class="{trend_class}">{trend_direction} by {abs(change):.2%}</span>.</p>
                <p>This represents a <span class="{trend_class}">{abs(percent_change):.2f}% {trend_direction}</span> over this period.</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Regional comparison
        processed_df = st.session_state.processed_datasets.get('prevalence', pd.DataFrame())
        
        if not processed_df.empty and 'region' in processed_df.columns:
            st.subheader("Regional Comparison")
            
            latest_processed = processed_df[processed_df['year'] == latest_year]
            
            # Calculate regional averages
            region_stats = latest_processed.groupby('region')['hiv_prevalence'].agg(['mean', 'std', 'count']).reset_index()
            region_stats = region_stats.sort_values('mean', ascending=False)
            
            # Plot regional comparison
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
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Highest Prevalence Regions")
                
                for _, row in region_stats.head(3).iterrows():
                    st.markdown(f"""
                    <div class="metric-card">
                        <h5>{row['region']}</h5>
                        <p>Average Prevalence: <span class="high-alert">{row['mean']:.2%}</span></p>
                        <p>Number of Counties: {int(row['count'])}</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.subheader("Lowest Prevalence Regions")
                
                for _, row in region_stats.tail(3).iloc[::-1].iterrows():
                    st.markdown(f"""
                    <div class="metric-card">
                        <h5>{row['region']}</h5>
                        <p>Average Prevalence: <span class="positive-trend">{row['mean']:.2%}</span></p>
                        <p>Number of Counties: {int(row['count'])}</p>
                    </div>
                    """, unsafe_allow_html=True)


if __name__ == "__main__":
    app()