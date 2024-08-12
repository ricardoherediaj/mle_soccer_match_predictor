import os
import pandas as pd
import comet_ml
import hopsworks
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix
import xgboost as xgb
import pickle  # Import pickle for saving model in .pkl format
from dotenv import load_dotenv
from paths import PARENT_DIR
from comet_ml import Experiment

# Load API keys from .env
load_dotenv(PARENT_DIR / '.env')
HOPSWORKS_API_KEY = os.environ['HOPSWORKS_API_KEY']
HOPSWORKS_PROJECT_NAME = 'rheredia8'
COMET_API_KEY = os.environ['COMET_API_KEY']
COMET_PROJECT_NAME = 'mle-soccer-project'
COMET_WORKSPACE = 'rheredia8'

# Create Comet experiment
experiment = Experiment(
    api_key=COMET_API_KEY,
    project_name=COMET_PROJECT_NAME,
    workspace=COMET_WORKSPACE,
)

# Load transformed features from Hopsworks Feature Store 
project = hopsworks.login(
    project=HOPSWORKS_PROJECT_NAME,
    api_key_value=HOPSWORKS_API_KEY
)

feature_store = project.get_feature_store()

# Read feature group
FEATURE_GROUP_NAME = 'la_liga_features'
FEATURE_GROUP_VERSION = 1

feature_group = feature_store.get_feature_group(
    name=FEATURE_GROUP_NAME,
    version=FEATURE_GROUP_VERSION
)

data_transformed = feature_group.read()

# Divide data into training and test
train_data = data_transformed[data_transformed['season_start'] <= 2022]
test_data = data_transformed[data_transformed['season_start'] == 2023]

features = [column for column in data_transformed.columns if column not in ['date', 'xG', 'xG_1', 'home', 'away', 'referee', 'venue', 'score', 'result', 'home_goals', 'away_goals', 'season_start']]

X_train = train_data[features]
y_train = train_data['result']
X_test = test_data[features]
y_test = test_data['result']

# Encode categorical variable
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

# Train XGBoost model
params = {
    'n_estimators': 100,
    'max_depth': 5,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'objective': 'multi:softprob',
    'eval_metric': 'mlogloss'
}

model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train_encoded)

# Evaluate model
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test_encoded, predictions)
conf_matrix = confusion_matrix(y_test_encoded, predictions)
print(f'XGBoost Accuracy: {accuracy:.4f}')
print('XGBoost Confusion Matrix:')
print(conf_matrix)

# Model registry in Comet
experiment.log_model("xgboost_la_liga_model", model)
experiment.log_confusion_matrix(matrix=conf_matrix)
experiment.log_metric("accuracy", accuracy)

# Save model in local as JSON
model_directory = '/Users/ricardoheredia/Desktop/mle-soccer-project/models'
model_json_path = os.path.join(model_directory, 'xgboost_la_liga_model.json')
model.save_model(model_json_path)
print(f"Model saved as JSON to {model_json_path}")

# Save model in local as .pkl
model_pkl_path = os.path.join(model_directory, 'xgboost_la_liga_model.pkl')
with open(model_pkl_path, 'wb') as file:
    pickle.dump(model, file)
print(f"Model saved as PKL to {model_pkl_path}")

# Upload model to Comet
experiment.log_asset(file_data=model_json_path, file_name='xgboost_la_liga_model.json')

