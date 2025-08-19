"""
Insights and recommendations page for the AfyaScope Kenya Streamlit dashboard.

This page provides key insights and recommendations based on the data analysis.
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


def app():
    """Run the insights and recommendations page."""
    st.title("Key Insights and Recommendations")
    
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
        fig = plt.figure(figsize=(10, 6))
        plt.plot(national_trends['year'], national_trends['hiv_prevalence'], marker='o', linewidth=2, color='#20B2AA')
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
    
    # Call to action
    st.subheader("Call to Action")
    
    st.markdown("""
    <div class="insight-card">
        <h4>Key Stakeholders</h4>
        <p>The following stakeholders should collaborate to address the HIV/STI challenges in Kenya:</p>
        <ul>
            <li><strong>Government agencies</strong>: Ministry of Health, county health departments</li>
            <li><strong>Healthcare providers</strong>: Hospitals, clinics, community health workers</li>
            <li><strong>NGOs and international organizations</strong>: UNAIDS, WHO, local NGOs</li>
            <li><strong>Community leaders</strong>: Religious leaders, chiefs, youth leaders</li>
            <li><strong>Academic and research institutions</strong>: Universities, research centers</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("""
    <div class="insight-card">
        <h4>Next Steps</h4>
        <ol>
            <li>Disseminate these findings to key stakeholders</li>
            <li>Develop county-specific action plans focusing on high-burden areas</li>
            <li>Allocate resources based on prevalence rates and trends</li>
            <li>Implement monitoring and evaluation frameworks to track progress</li>
            <li>Conduct regular data collection to keep information current</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    app()