import argparse
import json
import os
import pickle
from time import strftime, gmtime

import numpy as np
import polars as pl

from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score

import xgboost as xgb

# ------------------------------
# Argument parsing
# ------------------------------
parser = argparse.ArgumentParser(
        prog = "F1 Lap Time Prediction, ML Project",
        description = "Project for the ML Project / Data Science course (2024/2025 M1 IDD)")

# General arguments
parser.add_argument("--dataset_path", type=str, help="Path to dataset")
parser.add_argument("--save_dir", type=str, default="", help="Directory where model, logs, and config will be saved")


# ML model selection
parser.add_argument(
    "--ml_method",
    type=str,
    default=None,
    choices=["LinearRegression", "KNN", "DecisionTree", "RandomForest", "GradientBoosting", "Bagging", "XGBoost"],
    help="ML model to run (if not provided, all models will be tested)"
)

# Cross-validation
parser.add_argument("--cv_nsplits", type=int, default=7, help="Number of cross-validation splits") # cv = 7 gave the best results during training

# Hyperparameters (used depending on model)
parser.add_argument("--n_neighbors", type=int, default=5, help="Number of neighbors (KNN)")
parser.add_argument("--weights", type=str, default="distance", choices=["uniform", "distance"], help="Weight function used in prediction (KNN)")
parser.add_argument("--min_samples_split", type=int, default=2, help="Min samples split (Decision Tree, RandomForest)")
parser.add_argument("--max_depth", type=int, default=None, help="Maximum tree depth (Decision Tree, Random Forest, Gradient Boosting, XGBoost)")
parser.add_argument("--max_samples", type=float, default=1.0, help="Max samples (Bagging)")
parser.add_argument("--n_estimators", type=int, default=200, help="Number of estimators (Bagging, Random Forest, Gradient Boosting, XGBoost)")
parser.add_argument("--learning_rate", type=float, default=0.2, help="Learning rate (Gradient Boosting, XGBoost)")

args = parser.parse_args()

# ------------------------------
# Output folder setup
# ------------------------------
dir_name = strftime("%Y-%m-%d_%H-%M-%S", gmtime())
output_dir = os.path.join(args.save_dir, dir_name)
os.makedirs(output_dir, exist_ok=True)

path_model = os.path.join(output_dir, "model.pkl")
path_config = os.path.join(output_dir, "config.json")
path_log = os.path.join(output_dir, "logs.json")

# Store config
with open(path_config, "w") as f:
    json.dump(vars(args), f)

# ------------------------------
# Load dataset
# ------------------------------
df = pl.read_csv(args.dataset_path)

# One-Hot Encoding
df = df.to_dummies(columns=["Compound", "Team", "GrandPrix"])

# Columns to scale (numerical features)
cols_to_scale = [
    'Time', 'LapTime', 'LapNumber', 'Stint', 'Sector1Time', 'Sector2Time', 'Sector3Time',
    'Sector1SessionTime', 'Sector2SessionTime', 'Sector3SessionTime', 'SpeedI1', 'SpeedI2',
    'SpeedFL', 'SpeedST', 'IsPersonalBest', 'TyreLife', 'FreshTyre', 'LapStartTime', 'Position'
]

# Apply MinMaxScaler
minmax_scaler = MinMaxScaler()
df[cols_to_scale] = minmax_scaler.fit_transform(df[cols_to_scale])

# Apply StandardScaler
std_scaler = StandardScaler()
df[cols_to_scale] = std_scaler.fit_transform(df[cols_to_scale])

# Features
features = ['SpeedI1', 'SpeedI2', 'SpeedFL', 'SpeedST', 'LapNumber', 'TyreLife', 'LapStartTime', 'Stint', 'Position']

# Target
target = 'LapTime'

X = df[features]
y = df[target]

# First split: 80% train+val, 20% test
X_train_val, X_test, y_train_val, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Second split: 75% train, 25% validation (from train_val)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=0.25, random_state=42
)  # This gives 60% train, 20% val, 20% test

# ------------------------------
# Create model
# ------------------------------
models = {
    "LinearRegression": LinearRegression(),

    "KNN": KNeighborsRegressor(
        n_neighbors=args.n_neighbors,
        weights=args.weights
    ),

    "DecisionTree": DecisionTreeRegressor(
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split
    ),

    "Bagging": BaggingRegressor(
        n_estimators=args.n_estimators,
        max_samples=args.max_samples
    ),

    "RandomForest": RandomForestRegressor(
        n_estimators=100,
        max_depth=args.max_depth,
        min_samples_split=args.min_samples_split
    ),

    "GradientBoosting": GradientBoostingRegressor(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=5
    ),

    "XGBoost": xgb.XGBRegressor(
        n_estimators=args.n_estimators,
        learning_rate=args.learning_rate,
        max_depth=5
    )
}

# ------------------------------
# Select models to run
# ------------------------------
if args.ml_method:
    models_to_run = {args.ml_method: models[args.ml_method]}
else:
    models_to_run = models

# ------------------------------
# Cross-validation on train set only
# ------------------------------
def perform_cross_validation(model, X, y, cv_splits):
    cv = KFold(n_splits=cv_splits, shuffle=True, random_state=42)
    neg_mse_scores = cross_val_score(model, X, y, cv=cv, scoring="neg_mean_squared_error")
    mse_scores = -neg_mse_scores
    r2_scores = cross_val_score(model, X, y, cv=cv, scoring="r2")
    return mse_scores, r2_scores

# ------------------------------
# Train, evaluate and save models
# ------------------------------
results = {}


for name, model in models_to_run.items():
    print(f"Training and evaluating {name}...")

    model.fit(X_train, y_train)

    cv_mse_scores, cv_r2_scores = perform_cross_validation(model, X_train, y_train, args.cv_nsplits)

    y_val_pred = model.predict(X_val)
    val_mse = mean_squared_error(y_val, y_val_pred)
    val_r2 = r2_score(y_val, y_val_pred)

    y_test_pred = model.predict(X_test)
    test_mse = mean_squared_error(y_test, y_test_pred)
    test_r2 = r2_score(y_test, y_test_pred)

    model_path = path_model.replace(".pkl", f"_{name}.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)

    results[name] = {
        "train_cv_mse_mean": float(cv_mse_scores.mean()),
        "train_cv_mse_std": float(cv_mse_scores.std()),
        "train_cv_rmse_mean": float(np.sqrt(cv_mse_scores.mean())),
        "train_cv_r2_mean": float(cv_r2_scores.mean()),
        "train_cv_r2_std": float(cv_r2_scores.std()),
        "val_mse": val_mse,
        "val_rmse": np.sqrt(val_mse).item(),
        "val_r2": val_r2,
        "test_mse": test_mse,
        "test_rmse": np.sqrt(test_mse).item(),
        "test_r2": test_r2
    }

# Save all results to log file
with open(path_log, "w") as f:
    json.dump(results, f, indent=4)

print(f"Results saved in {path_log}")
print(f"Models saved in {output_dir}")