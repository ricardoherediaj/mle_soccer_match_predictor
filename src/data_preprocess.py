import pandas as pd
import numpy as np
import requests
from bs4 import BeautifulSoup
from janitor import clean_names
from paths import RAW_DATA_DIR
import argparse

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
    
    # Data cleaning
    df = clean_names(df)
    df.rename(columns = {'wk':'week', 'xg':'xG', 'xg_1':'xG_1'}, inplace=True)
    df['week'] = df['week'].astype('Int64')
    df['date'] = pd.to_datetime(df['date'])
    
    # Separate categorical and numerical data
    cat = df.select_dtypes(exclude='number').copy()
    num = df.select_dtypes(include='number').copy()

    # Clean categorical data
    cat.drop(columns='time', inplace=True)

    # Clean numerical data
    num.drop(columns='attendance', inplace=True)

    # Concatenate cleaned data
    data_quality = pd.concat([num, cat], axis=1)
    data_quality = data_quality[['week', 'date', 'day', 'home', 'score', 'away', 'xG', 'xG_1', 'venue', 'referee']]

    # Save cleaned data to CSV
    data_quality.to_csv(output_path, index=False)
    print(f"Data saved to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Scrape and clean La Liga data.')
    parser.add_argument('--start_year', type=int, required=True, help='Starting year for scraping data.')
    parser.add_argument('--end_year', type=int, required=True, help='Ending year for scraping data.')
    parser.add_argument('--output_path', type=str, required=True, help='Output path for the cleaned data.')

    args = parser.parse_args()

    scrape_fbref_la_liga_data(args.start_year, args.end_year, args.output_path)