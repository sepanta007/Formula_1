# 🏎️ Formula 1 Lap Time Prediction

This project aims to predict lap times in Formula 1 races using various machine learning models. It is part of the Machine Learning Project / Data Science course.

* [Project Structure](#-project-structure)
* [Usage](#-usage)
* [Results and Outputs](#-results-and-outputs)
* [Analysis](#-analysis)
* [Requirements](#-requirements)
* [License](#-license)

---

## 📂 Project Structure

```
.
├── f1_prediction.ipynb                    # Jupyter notebook with full analysis and model comparison
├── grand_prix_laps_data_2024.csv          # Raw dataset CSV file
├── grand_prix_laps_data_2024_clean.csv    # Cleaned and preprocessed dataset
├── main.py                                # Main Python script to run training and evaluation of models
├── preparation.py                         # Script to extract and prepare data from the FastF1 API
└── F1_Prediction.pdf                      # Full detailed report of the project

```

---

## 🚀 Usage

### Example of running the main training script

```bash
python main.py --dataset_path ./grand_prix_laps_data_2024_clean.csv --save_dir results --ml_method RandomForest --cv_nsplits 7 --n_estimators 100
```

### Arguments

- `--dataset_path`: Path to the dataset CSV file (required).
- `--save_dir`: Directory where models, logs, and configs will be saved (default: current directory).
- `--ml_method`: Specify the ML model to run. Choices are:
  - `LinearRegression`
  - `KNN`
  - `DecisionTree`
  - `RandomForest`
  - `GradientBoosting`
  - `Bagging`
  - `XGBoost`
  - If not specified, all models are trained and evaluated.
- `--cv_nsplits`: Number of cross-validation splits (default: 7).
- `--n_neighbors`: Number of neighbors (for KNN, default: 5).
- `--weights`: Weight function for KNN (`uniform` or `distance`, default: `distance`).
- `--min_samples_split`: Minimum samples to split a node (DecisionTree, RandomForest, default: 2).
- `--max_depth`: Maximum depth of trees (DecisionTree, RandomForest, GradientBoosting, XGBoost, default: `None`).
- `--max_samples`: Maximum samples for Bagging (default: 1.0).
- `--n_estimators`: Number of estimators for Bagging, RandomForest, GradientBoosting, XGBoost (default: 200).
- `--learning_rate`: Learning rate for GradientBoosting and XGBoost (default: 0.2).

---

## 📈 Results and Outputs

- Models are saved as `.pkl` files in the specified `save_dir` under a timestamped folder.
- Logs containing training, validation, and test metrics are saved as `logs.json`.
- Configuration used for the run is saved as `config.json`.

---

## 📊 Analysis

Detailed exploratory data analysis and model comparisons can be found in `f1_prediction.ipynb`.

The final report is available in `F1_Prediction.pdf`.

---

## 🧰 Requirements

- Python 3.8+
- Packages:
  - fastf1
  - pandas
  - seaborn
  - numpy
  - matplotlib
  - scikit-learn
  - xgboost
  - polars
  - argparse
  - json
  - os
  - pickle

---

## 📜 License

This project is released under the [MIT](https://github.com/sepanta007/Formula_1/blob/master/LICENSE) License. 