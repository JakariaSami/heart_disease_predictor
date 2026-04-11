import numpy as np
import pandas as pd
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os

# Feature definitions
CONTINUOUS_FEATURES = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
CATEGORICAL_FEATURES = ['sex', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal']

def load_data():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    X = pd.read_csv(os.path.join(base_dir, "data/raw/features.csv"))
    y = pd.read_csv(os.path.join(base_dir, "data/raw/targets.csv")).values.ravel()
    y = (y > 0).astype(int)
    return X, y

def build_preprocessor(): 
  continuous_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
  ])

  categorical_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
  ])

  preprocessor = ColumnTransformer([
    ('continuous', continuous_pipeline, CONTINUOUS_FEATURES), 
    ('categorical', categorical_pipeline, CATEGORICAL_FEATURES)
  ])

  return preprocessor

def get_processed_data(test_size=0.2, random_state=42):
  X, y = load_data()
  X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state, stratify=y
  )
  
  return X_train, X_test, y_train, y_test


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = get_processed_data()
    print(f"Train: {X_train.shape}, Test: {X_test.shape}")
    print(f"Train positive rate: {y_train.mean():.2f}")
    print(f"Test positive rate: {y_test.mean():.2f}")