import mlflow
import mlflow.sklearn
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import joblib
import os

from data import load_data, build_preprocessor, get_processed_data, CONTINUOUS_FEATURES, CATEGORICAL_FEATURES

def evaluate(model, X_test, y_test):
  y_pred = model.predict(X_test)
  y_proba = model.predict_proba(X_test)[:, 1]
  return {
    "accuracy": accuracy_score(y_test, y_pred),
    "f1": f1_score(y_test, y_pred),
    "roc_auc": roc_auc_score(y_test, y_proba),
  }


def train_model(model_name, classifier, X_train, X_test, y_train, y_test, use_smote=True):
    preprocessor = build_preprocessor()

    steps = [('preprocessor', preprocessor)]
    if use_smote:
        steps.append(('smote', SMOTE(random_state=42)))
    steps.append(('classifier', classifier))

    # will use imbalanced-learn pipeline if SMOTE is included
    PipelineClass = ImbPipeline if use_smote else Pipeline
    pipeline = PipelineClass(steps)

    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model", model_name)
        mlflow.log_param("use_smote", use_smote)

        pipeline.fit(X_train, y_train)
        metrics = evaluate(pipeline, X_test, y_test)

        for k, v in metrics.items():
            mlflow.log_metric(k, v)

        mlflow.sklearn.log_model(pipeline, artifact_path="model")

        print(f"\n{model_name}")
        print(f"  Accuracy : {metrics['accuracy']:.4f}")
        print(f"  F1       : {metrics['f1']:.4f}")
        print(f"  ROC-AUC  : {metrics['roc_auc']:.4f}")

    return pipeline, metrics


def save_final_model(pipeline, metrics, model_name): 
    import json

    # Always resolves to project_root/models/ no matter where it's ran from
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    models_dir = os.path.join(base_dir, "models")
    os.makedirs(models_dir, exist_ok=True)

    joblib.dump(pipeline, os.path.join(models_dir, "best_model.pkl"))

    metadata = {
        "model_name": model_name,
        "metrics": metrics, 
        "features": {
            "continuous": CONTINUOUS_FEATURES,
            "categorical": CATEGORICAL_FEATURES
        }
    }
    with open(os.path.join(models_dir, "model_metadata.json"), "w") as f:
        json.dump(metadata, f, indent=2)

    # Save transformed training data for SHAP background
    X_train, _, y_train, _ = get_processed_data()
    X_train_transformed = pipeline.named_steps['preprocessor'].transform(X_train)
    np.save(os.path.join(models_dir, "X_train_transformed.npy"), X_train_transformed)
    
    print("Model and metadata saved to models/")


def run_experiments():
    X_train, X_test, y_train, y_test = get_processed_data()

    # Creating the 'mlruns' folder in the parent directory
    mlflow.set_tracking_uri("file:../mlruns")
    mlflow.set_experiment("heart-disease-prediction")

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42),
        "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
        "GradientBoosting": GradientBoostingClassifier(n_estimators=200, random_state=42),
        "XGBoost": XGBClassifier(n_estimators=200, eval_metric='logloss',
        random_state=42, verbosity=0),
    }

    results = {}
    best_pipeline = None
    best_auc = 0

    for name, clf in models.items():
        pipeline, metrics = train_model(
            name, clf, X_train, X_test, y_train, y_test
        )
        results[name] = metrics
        if metrics['roc_auc'] > best_auc:
            best_auc = metrics['roc_auc']
            best_pipeline = pipeline
            best_name = name

    print(f"\nBest model: {best_name} (ROC-AUC: {best_auc:.4f})")

    save_final_model(best_pipeline, results[best_name], best_name)
    return results


if __name__ == "__main__":
  results = run_experiments()