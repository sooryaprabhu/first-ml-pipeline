import pytest
import pandas as pd
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from load_data import load_data
from preprocess import preprocess_data
from train import train_model
from evaluate import evaluate_model


# ── TEST 1 ────────────────────────────────────────────────────────────
def test_load_data_shape():
    df = load_data("data/housing.csv")
    assert df.shape[0] == 20640, "Should have 20640 rows"
    assert df.shape[1] == 9, "Should have 9 columns"


# ── TEST 2 ────────────────────────────────────────────────────────────
def test_load_data_returns_dataframe():
    df = load_data("data/housing.csv")
    assert isinstance(df, pd.DataFrame), "Should return a DataFrame"


# ── TEST 3 ────────────────────────────────────────────────────────────
def test_no_missing_values():
    df = load_data("data/housing.csv")
    assert df.isnull().sum().sum() == 0, "Should have no missing values"


# ── TEST 4 ────────────────────────────────────────────────────────────
def test_target_column_exists():
    df = load_data("data/housing.csv")
    assert "MedHouseVal" in df.columns, "Target column should exist"


# ── TEST 5 ────────────────────────────────────────────────────────────
def test_preprocess_shapes():
    df = load_data("data/housing.csv")
    X_scaled, y, scaler = preprocess_data(df)
    assert X_scaled.shape[1] == 8, "Should have 8 feature columns"
    assert len(y) == len(X_scaled), "X and y should have same length"


# ── TEST 6 ────────────────────────────────────────────────────────────
def test_scaling_mean():
    df = load_data("data/housing.csv")
    X_scaled, y, scaler = preprocess_data(df)
    means = X_scaled.mean().round(5)
    for col in X_scaled.columns:
        assert abs(means[col]) < 0.01, f"{col} mean should be ~0"


# ── TEST 7 ────────────────────────────────────────────────────────────
def test_target_positive():
    df = load_data("data/housing.csv")
    X_scaled, y, scaler = preprocess_data(df)
    assert (y > 0).all(), "All house prices should be positive"


# ── TEST 8 — MODEL PERFORMANCE ────────────────────────────────────────
# This is the ML specific test
# Checks model accuracy is above minimum threshold
# If someone breaks the pipeline → R2 drops → CI fails
def test_model_performance():
    df = load_data("data/housing.csv")
    X_scaled, y, scaler = preprocess_data(df)
    model, X_train, X_test, y_train, y_test = train_model(X_scaled, y)
    y_pred = evaluate_model(model, X_test, y_test)

    from sklearn.metrics import r2_score, mean_absolute_error
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    # R2 must be above 0.50
    # If it drops below → something is broken
    assert r2 > 0.50, f"R2 too low: {r2:.4f} — model performance degraded"

    # MAE must be below $80,000
    # If it goes above → predictions are too inaccurate
    assert mae < 0.80, f"MAE too high: ${mae*100000:,.0f} — model performance degraded"

    print(f"Model performance OK — R2: {r2:.4f}, MAE: ${mae*100000:,.0f}")
