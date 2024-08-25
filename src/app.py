import streamlit as st
import pandas as pd
import hopsworks
import xgboost as xgb
import pickle
import os
from dotenv import load_dotenv
from comet_ml import API

# Load the environment variables
load_dotenv()

# Title of the app
st.title('La Liga Match Predictions')

# Subtitle or description
st.write("This application allows you to visualize match predictions for the 2023-2024 season.")

# Fetch Hopsworks API key from .env file
HOPSWORKS_API_KEY = os.getenv('HOPSWORKS_API_KEY')
COMET_API_KEY = os.getenv('COMET_API_KEY')

# Connect to Hopsworks to fetch the features
project = hopsworks.login(api_key_value=HOPSWORKS_API_KEY)
feature_store = project.get_feature_store()

FEATURE_GROUP_NAME = 'la_liga_features'
FEATURE_GROUP_VERSION = 1
feature_group = feature_store.get_feature_group(
    name=FEATURE_GROUP_NAME,
    version=FEATURE_GROUP_VERSION
)

# Load transformed data from the Feature Store
data_transformed = feature_group.read()

# Display a table with the data
st.write("2023-2024 season data:")
st.dataframe(data_transformed.head())

# Connect to CometML to retrieve the model
api = API(api_key=COMET_API_KEY)
workspace = "rheredia8"
project_name = "mle-soccer-project"
model_name = "xgboost_la_liga_model"

# Retrieve the model from CometML
model_registry = api.get_registry_model(workspace, project_name, model_name)
model_version = model_registry.get_latest_version()
model_file_path = model_version.download()

# Load the model from the retrieved artifact
model = xgb.Booster(model_file_path)

# Make predictions with the model
st.write("Making predictions...")
features = [column for column in data_transformed.columns if column not in ['date', 'xG', 'xG_1', 'home', 'away', 'referee', 'venue', 'score', 'result', 'home_goals', 'away_goals', 'season_start']]
X = data_transformed[features]
dmatrix = xgb.DMatrix(X)
predictions = model.predict(dmatrix)

# Display the predictions
st.write("Predicted match results:")
st.write(predictions)