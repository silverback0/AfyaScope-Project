import requests
import pandas as pd
import os 

# Indicators of interest
INDICATORS = {
    "PLHIV": "HIV_0000000001",         # People living with HIV
    "Prevalence_15_49": "HIV_0000000003",
    "New_HIV_Infections": "HIV_0000000002",
    "AIDS_Deaths": "HIV_0000000006",
    "ART_Coverage": "HIV_0000000009"
}

OUTPUT_PATH = os.path.join("data", "raw", "hiv_kenya.csv")

def fetch_indicator(indicator_code, indicator_name):
    """Fetch one indicator from WHO Athena API"""
    url = f"https://ghoapi.azureedge.net/api/{indicator_code}"
    print(f"Fetching: {url}")  # debug
    response = requests.get(url)
    print(f"Status Code: {response.status_code}")  # debug
    print(f"Response Text (first 200 chars): {response.text[:200]}")  # debug
    data = response.json()
    df = pd.json_normalize(data['value'])
    # Filter for Kenya only
    df = df[df['SpatialDim'] == 'KEN']
    df['Indicator'] = indicator_name
    return df[['TimeDim', 'Value', 'Indicator']]

def clean_values(value):
    """Extract numeric value from string with confidence interval"""
    # Example input: "1 300 000 [1 200 000 - 1 500 000]"
    return int(value.split('[')[0].replace(" ", "").replace(",", ""))

def main():
    all_data = []
    for name, code in INDICATORS.items():
        df = fetch_indicator(code, name)
        # Clean Value column
        df['Value'] = df['Value'].apply(clean_values)
        all_data.append(df)
    
    final_df = pd.concat(all_data)
    final_df.rename(columns={"TimeDim": "Year"}, inplace=True)
    
    # Ensure output directory exists
    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    final_df.to_csv(OUTPUT_PATH, index=False)
    print(f"Data saved to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
    print("Fetching WHO HIV data for Kenya...")