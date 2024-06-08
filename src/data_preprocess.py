import pandas as pd
import requests
from bs4 import BeautifulSoup
from paths import RAW_DATA_DIR

def scrape_fbref_la_liga_data(start_year, end_year, output_path):
    """
    Function to scrape La Liga match data from FBRef and save it to a CSV file.
    
    Parameters:
    - start_year: Starting year for scraping (int).
    - end_year: Ending year for scraping (int).
    - output_path: Path to save the CSV file (str).
    
    Returns:
    - None
    """
    dates = pd.date_range(start=f'1/1/{start_year}', end=f'12/31/{end_year}', freq='YS')
    all_dfs = []

    for date in dates:
        season_start = date.year
        season_end = date.year + 1
        season = f'{season_start}-{season_end}'
        print(f"Scraping season: {season}")
        
        # Get La Liga data
        try:
            url = f'https://fbref.com/en/comps/12/{season}/schedule/{season}-La-Liga-Scores-and-Fixtures'
            df = pd.read_html(url, attrs={"id": f"sched_{season}_12_1"})[0]
            all_dfs.append(df)
        except Exception as e:
            print(f"Error scraping La Liga for season {season}: {e}")

    # Concatenate and clean data
    df = pd.concat(all_dfs, ignore_index=True)
    df = df.dropna(subset=['Wk'])
    df.drop(columns=['Match Report', 'Notes'], inplace=True)

    # Save data to CSV
    df.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")
