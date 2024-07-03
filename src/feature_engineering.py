import os
import pandas as pd
from datetime import datetime
import numpy as np
from dotenv import load_dotenv
from paths import TRANSFORMED_DATA_DIR, PARENT_DIR

# API key from .env
load_dotenv(PARENT_DIR / '.env')
HOPSWORKS_API_KEY = os.environ['HOPSWORKS_API_KEY']
HOPSWORKS_PROJECT_NAME = 'rheredia8'

# Load preprocessed data
data_quality = pd.read_csv(TRANSFORMED_DATA_DIR / 'data_quality.csv')

# Feature engineering

# Divide 'score' column into 'home_goals' and 'away_goals'
data_quality[['home_goals', 'away_goals']] = data_quality['score'].str.split('–', expand=True).astype(float)

# Create 'season_start' column to identify the season
data_quality['date'] = pd.to_datetime(data_quality['date'])
data_quality['season_start'] = data_quality['date'].apply(lambda x: x.year - 1 if x.month < 8 else x.year)

# Create target variable 'result' for match prediction 
def determine_result(row):
    if row['home_goals'] > row['away_goals']:
        return 'Home win'
    elif row['home_goals'] < row['away_goals']:
        return 'Away win'
    else:
        return 'Draw'

data_quality['result'] = data_quality.apply(determine_result, axis=1)

# One Hot Encoding column 'day'
data_quality['day'] = data_quality['date'].dt.day_name()
data_quality = pd.get_dummies(data_quality, columns=['day'])

# Calculate rolling averages
data_quality = data_quality.sort_values(['date'])

# Calculate rolling averages for goals
for x in data_quality.home.unique():
    temp_df = data_quality[(data_quality['home'] == x) | (data_quality['away'] == x)]
    temp_df = temp_df.sort_values(['date'])
    
    temp_df['goal_value_to_calculate'] = temp_df.apply(lambda y: y['home_goals'] if y['home'] == x else y['away_goals'], axis=1)
    temp_df['rolling_avg_goals'] = temp_df['goal_value_to_calculate'].rolling(window=5, closed="left", min_periods=1).mean()
    
    for index, row in temp_df.iterrows():
        if row['home'] == x:
            data_quality.at[index, 'home_rolling_avg_goals'] = row['rolling_avg_goals']
        else:
            data_quality.at[index, 'away_rolling_avg_goals'] = row['rolling_avg_goals']

# Calculate rolling averages for xG
for x in data_quality.home.unique():
    temp_df = data_quality[(data_quality['home'] == x) | (data_quality['away'] == x)]
    temp_df = temp_df.sort_values(['date'])
    
    temp_df['xG_value_to_calculate'] = temp_df.apply(lambda y: y['xG'] if y['home'] == x else y['xG_1'], axis=1)
    temp_df['rolling_avg_xG'] = temp_df['xG_value_to_calculate'].rolling(window=5, closed="left", min_periods=1).mean()
    
    for index, row in temp_df.iterrows():
        if row['home'] == x:
            data_quality.at[index, 'home_rolling_avg_xG'] = row['rolling_avg_xG']
        else:
            data_quality.at[index, 'away_rolling_avg_xG'] = row['rolling_avg_xG']

# Cleean rows with nulls rolling averages
data_quality = data_quality.dropna(subset=['home_rolling_avg_goals', 'away_rolling_avg_goals', 'home_rolling_avg_xG', 'away_rolling_avg_xG'])

# Save transformed features
data_transformed = data_quality
output_path_data_transformed = TRANSFORMED_DATA_DIR / 'data_transformed_test.csv'
data_transformed.to_csv(output_path_data_transformed, index=False)
print(f"Data saved to {output_path_data_transformed}")

# Upsert features to Hopsworks
import hopsworks

project = hopsworks.login(
    project=HOPSWORKS_PROJECT_NAME,
    api_key_value=HOPSWORKS_API_KEY
)

feature_store = project.get_feature_store()

FEATURE_GROUP_NAME = 'la_liga_features'
FEATURE_GROUP_VERSION = 1

feature_group = feature_store.get_or_create_feature_group(
    name=FEATURE_GROUP_NAME,
    version=FEATURE_GROUP_VERSION,
    description="La Liga features from matches",
    primary_key=['date', 'home', 'away'],
    event_time='date',
)

feature_group.insert(data_transformed, write_options={"wait_for_job": False})