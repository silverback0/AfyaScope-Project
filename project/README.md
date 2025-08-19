# AfyaScope Kenya: HIV & STI Trends and Predictions

## Overview
AfyaScope Kenya is a data science project that analyzes and predicts HIV/AIDS and STI trends in Kenya using real-world data. The project aims to present the results in an understandable way for the general public, healthcare workers, and policymakers.

## Features
- Data cleaning and integration of multiple HIV/STI datasets
- Exploratory data analysis with visualizations
- Geospatial mapping of prevalence rates across counties
- Time series forecasting of HIV/STI trends
- Interactive Streamlit dashboard for data exploration

## Project Structure
```
/afyascope-kenya/
├── data/                      # Data files
│   ├── raw/                   # Original datasets
│   └── processed/             # Cleaned and processed datasets
├── notebooks/                 # Jupyter notebooks for exploration
├── src/                       # Source code
│   ├── data/                  # Data loading and processing
│   ├── features/              # Feature engineering
│   ├── models/                # Prediction models
│   ├── visualization/         # Visualization components
│   └── utils/                 # Helper functions
├── dashboard/                 # Streamlit dashboard
│   ├── app.py                 # Main dashboard file
│   ├── pages/                 # Dashboard pages
│   └── components/            # Reusable dashboard components
├── tests/                     # Unit tests
├── README.md                  # Project documentation
└── requirements.txt           # Package dependencies
```

## Prerequisites
- Python 3.8 or higher must be installed on your system
- Ensure Python is added to your system's PATH
- Node.js 18 or higher for the web interface

## Setup Instructions
1. Clone this repository
2. Install Python if not already installed:
   - Download from [python.org](https://www.python.org/downloads/)
   - During installation, check "Add Python to PATH"
3. Create a virtual environment:
   ```bash
   python -m venv venv
   ```
4. Activate the virtual environment:
   - Windows: `venv\Scripts\activate`
   - Mac/Linux: `source venv/bin/activate`
5. Upgrade pip to the latest version:
   ```bash
   python -m pip install --upgrade pip
   ```
6. Install dependencies:
   ```bash
   python -m pip install -r requirements.txt
   ```
7. Place the data files in the `data/raw/` directory
8. Run the dashboard:
   ```bash
   python -m streamlit run dashboard/app.py
   ```

## Data Sources
The project uses three primary datasets:
- `indicators.csv`: HIV/STI indicators by county and year
- `prevalence.csv`: HIV and STI prevalence rates by county and year
- `unaids_facts.csv`: Extracted summaries from UNAIDS fact sheets

## Dashboard
The interactive dashboard provides:
- Overview of HIV/STI situation in Kenya
- County-level analysis and comparisons
- Temporal trends and patterns
- Future predictions based on historical data
- Key insights from UNAIDS reports

## License
[MIT License](LICENSE)