"""
Heart Attack Risk Prediction - Model Training Script
Dataset: JoseMBM CDC 2022 Heart Attack Risk Predictors (heart_2022_no_nans.csv)
Model: Decision Tree Regressor with Cross-Validation and Hyperparameter Tuning
"""

import os
import json
import time
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, GridSearchCV, cross_validate, train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "heart_2022_no_nans.csv")
MODEL_DIR = os.path.join(os.path.dirname(__file__), "model")
MODEL_PATH = os.path.join(MODEL_DIR, "heart_model.pkl")
META_PATH = os.path.join(MODEL_DIR, "model_meta.json")

# Selected feature set: clinically significant and lifestyle-related predictors
CATEGORICAL_FEATURES = [
    "AgeCategory",
    "Sex",
    "GeneralHealth",
    "LastCheckupTime",
    "PhysicalActivities",
    "SmokerStatus",
    "HadAngina",
    "HadStroke",
    "HadAsthma",
    "HadCOPD",
    "HadDepressiveDisorder",
    "HadKidneyDisease",
    "HadArthritis",
    "HadDiabetes",
    "DifficultyWalking",
    "ChestScan",
    "AlcoholDrinkers",
]

NUMERICAL_FEATURES = [
    "BMI",
    "PhysicalHealthDays",
    "MentalHealthDays",
    "SleepHours",
]

ALL_FEATURES = CATEGORICAL_FEATURES + NUMERICAL_FEATURES
TARGET_COL = "HadHeartAttack"


def load_and_preprocess_data(csv_path: str):
    print(f"Loading dataset from: {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Dataset loaded. Total shape: {df.shape}")

    # Map target: HadHeartAttack 'Yes' -> 1.0, 'No' -> 0.0
    y = (df[TARGET_COL].str.strip().str.lower() == "yes").astype(float)
    X = df[ALL_FEATURES].copy()

    # Data integrity: ensure numerical types
    for col in NUMERICAL_FEATURES:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(X[col].median())

    # Ensure categorical types are string
    for col in CATEGORICAL_FEATURES:
        X[col] = X[col].astype(str).str.strip()

    print(f"Target distribution - Heart Attack Cases: {int(y.sum())} ({y.mean()*100:.2f}%), Non-cases: {int((1-y).sum())}")
    return X, y, df


def build_pipeline():
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", NUMERICAL_FEATURES),
            ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("regressor", DecisionTreeRegressor(random_state=42)),
        ]
    )
    return pipeline


def tune_and_train():
    start_time = time.time()
    os.makedirs(MODEL_DIR, exist_ok=True)

    X, y, df_full = load_and_preprocess_data(DATA_PATH)

    # Train / Test split for unbiased final test set evaluation
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=(y > 0)
    )
    print(f"Training split: {X_train.shape[0]} samples, Testing split: {X_test.shape[0]} samples")

    # Hyperparameter Grid Search with 5-Fold Cross Validation
    print("\n--- Running Hyperparameter Tuning & Cross Validation (GridSearchCV) ---")
    param_grid = {
        "regressor__max_depth": [5, 6, 8, 10],
        "regressor__min_samples_split": [20, 50, 100],
        "regressor__min_samples_leaf": [20, 50, 100],
        "regressor__criterion": ["squared_error", "friedman_mse"],
    }

    # Use a representative sample of 50,000 for fast, thorough grid search
    tune_sample_size = min(50000, len(X_train))
    X_tune = X_train.sample(n=tune_sample_size, random_state=42)
    y_tune = y_train.loc[X_tune.index]

    base_pipe = build_pipeline()
    cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        estimator=base_pipe,
        param_grid=param_grid,
        cv=cv_strategy,
        scoring="neg_mean_squared_error",
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )

    tune_start = time.time()
    grid_search.fit(X_tune, y_tune)
    tune_duration = time.time() - tune_start
    print(f"Hyperparameter tuning completed in {tune_duration:.2f}s")
    print(f"Best Hyperparameters: {grid_search.best_params_}")
    print(f"Best CV Neg MSE: {grid_search.best_score_:.6f} (CV RMSE: {np.sqrt(-grid_search.best_score_):.4f})")

    # Re-evaluate best model using 5-fold CV across the entire training set
    best_params = grid_search.best_params_
    best_pipe = Pipeline(
        steps=[
            ("preprocessor", build_pipeline().named_steps["preprocessor"]),
            (
                "regressor",
                DecisionTreeRegressor(
                    max_depth=best_params["regressor__max_depth"],
                    min_samples_split=best_params["regressor__min_samples_split"],
                    min_samples_leaf=best_params["regressor__min_samples_leaf"],
                    criterion=best_params["regressor__criterion"],
                    random_state=42,
                ),
            ),
        ]
    )

    print("\n--- Evaluating 5-Fold Cross Validation on Full Training Set ---")
    cv_results = cross_validate(
        best_pipe,
        X_train,
        y_train,
        cv=cv_strategy,
        scoring={
            "mse": "neg_mean_squared_error",
            "mae": "neg_mean_absolute_error",
            "r2": "r2",
        },
        return_train_score=False,
        n_jobs=-1,
    )

    cv_mse_scores = [-val for val in cv_results["test_mse"]]
    cv_mae_scores = [-val for val in cv_results["test_mae"]]
    cv_r2_scores = list(cv_results["test_r2"])

    print(f"5-Fold CV Mean MSE: {np.mean(cv_mse_scores):.6f} (+/- {np.std(cv_mse_scores):.6f})")
    print(f"5-Fold CV Mean RMSE: {np.mean([np.sqrt(s) for s in cv_mse_scores]):.4f}")
    print(f"5-Fold CV Mean MAE: {np.mean(cv_mae_scores):.4f}")
    print(f"5-Fold CV Mean R²: {np.mean(cv_r2_scores):.4f}")

    # Fit final pipeline on all training data
    print("\n--- Fitting Final Pipeline on Full Training Set ---")
    best_pipe.fit(X_train, y_train)

    # Test set evaluation
    test_preds = best_pipe.predict(X_test)
    test_mse = mean_squared_error(y_test, test_preds)
    test_rmse = float(np.sqrt(test_mse))
    test_mae = float(mean_absolute_error(y_test, test_preds))
    test_r2 = float(r2_score(y_test, test_preds))

    print("\n--- Test Set Evaluation (Holdout 20%, 49,205 samples) ---")
    print(f"Test MSE: {test_mse:.6f}")
    print(f"Test RMSE: {test_rmse:.4f}")
    print(f"Test MAE: {test_mae:.4f}")
    print(f"Test R²: {test_r2:.4f}")
    print(f"Prediction min: {test_preds.min()*100:.2f}%, max: {test_preds.max()*100:.2f}%, mean: {test_preds.mean()*100:.2f}%")

    # Extract Feature Importances
    ohe = best_pipe.named_steps["preprocessor"].named_transformers_["cat"]
    cat_feature_names = ohe.get_feature_names_out(CATEGORICAL_FEATURES).tolist()
    all_feature_names = NUMERICAL_FEATURES + cat_feature_names
    regressor = best_pipe.named_steps["regressor"]
    importances = regressor.feature_importances_

    feature_imp_list = [
        {"feature": name, "importance": float(imp)}
        for name, imp in zip(all_feature_names, importances)
        if imp > 0.0005
    ]
    feature_imp_list.sort(key=lambda x: x["importance"], reverse=True)

    # Aggregate by base feature
    base_importances = {}
    for item in feature_imp_list:
        base_col = item["feature"].split("_")[0] if "_" in item["feature"] else item["feature"]
        base_importances[base_col] = base_importances.get(base_col, 0.0) + item["importance"]
    base_imp_list = [{"feature": k, "importance": round(v * 100, 2)} for k, v in base_importances.items()]
    base_imp_list.sort(key=lambda x: x["importance"], reverse=True)

    # Collect unique options for UI dropdowns
    options = {}
    for col in CATEGORICAL_FEATURES:
        options[col] = sorted(df_full[col].dropna().unique().tolist())

    metadata = {
        "model_type": "Decision Tree Regressor",
        "dataset": "CDC 2022 Heart Attack Risk Predictors (JoseMBM Kaggle)",
        "total_records": len(df_full),
        "train_records": len(X_train),
        "test_records": len(X_test),
        "target": "HadHeartAttack (Risk Score: 0.0 to 1.0 / 0% to 100%)",
        "positive_rate_percent": round(float(y.mean() * 100), 2),
        "best_hyperparameters": {
            "max_depth": int(best_params["regressor__max_depth"]),
            "min_samples_split": int(best_params["regressor__min_samples_split"]),
            "min_samples_leaf": int(best_params["regressor__min_samples_leaf"]),
            "criterion": str(best_params["regressor__criterion"]),
        },
        "cv_results": {
            "folds": 5,
            "cv_mse_mean": float(np.mean(cv_mse_scores)),
            "cv_rmse_mean": float(np.mean([np.sqrt(s) for s in cv_mse_scores])),
            "cv_mae_mean": float(np.mean(cv_mae_scores)),
            "cv_r2_mean": float(np.mean(cv_r2_scores)),
            "cv_folds_mse": [float(s) for s in cv_mse_scores],
            "cv_folds_r2": [float(s) for s in cv_r2_scores],
        },
        "test_metrics": {
            "mse": float(test_mse),
            "rmse": float(test_rmse),
            "mae": float(test_mae),
            "r2": float(test_r2),
            "min_predicted_risk": float(test_preds.min()),
            "max_predicted_risk": float(test_preds.max()),
            "mean_predicted_risk": float(test_preds.mean()),
        },
        "top_features": base_imp_list[:10],
        "detailed_features": feature_imp_list[:20],
        "categorical_features": CATEGORICAL_FEATURES,
        "numerical_features": NUMERICAL_FEATURES,
        "options": options,
        "trained_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "total_elapsed_seconds": round(time.time() - start_time, 2),
    }

    # Save model and metadata
    print(f"\nSaving model pipeline to {MODEL_PATH}...")
    joblib.dump(best_pipe, MODEL_PATH)

    print(f"Saving metadata to {META_PATH}...")
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print("--- Training and Evaluation Successfully Completed! ---")
    return best_pipe, metadata


if __name__ == "__main__":
    tune_and_train()
