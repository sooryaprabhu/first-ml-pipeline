import pytest
import pandas as pd
import sys
import os

# This line lets Python find our src/ folder
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from load_data import load_data
from preprocess import preprocess_data

# ── TEST 1 ────────────────────────────────────────────────────────────
# Test that load_data returns a DataFrame with correct shape
def test_load_data_shape():
    df = load_data("data/housing.csv")
    # We know this dataset has 20,640 rows and 9 columns
    assert df.shape[0] == 20640, "Should have 20640 rows"
    assert df.shape[1] == 9,     "Should have 9 columns"

# ── TEST 2 ────────────────────────────────────────────────────────────
# Test that load_data returns a pandas DataFrame
def test_load_data_returns_dataframe():
    df = load_data("data/housing.csv")
    assert isinstance(df, pd.DataFrame), "Should return a DataFrame"

# ── TEST 3 ────────────────────────────────────────────────────────────
# Test that there are no missing values
def test_no_missing_values():
    df = load_data("data/housing.csv")
    assert df.isnull().sum().sum() == 0, "Should have no missing values"

# ── TEST 4 ────────────────────────────────────────────────────────────
# Test that MedHouseVal column exists (our target)
def test_target_column_exists():
    df = load_data("data/housing.csv")
    assert "MedHouseVal" in df.columns, "Target column should exist"

# ── TEST 5 ────────────────────────────────────────────────────────────
# Test that preprocess returns correct shapes
def test_preprocess_shapes():
    df = load_data("data/housing.csv")
    X_scaled, y, scaler = preprocess_data(df)
    # X should have 8 columns (9 minus the target)
    assert X_scaled.shape[1] == 8, "Should have 8 feature columns"
    # y should have same number of rows as X
    assert len(y) == len(X_scaled), "X and y should have same length"

# ── TEST 6 ────────────────────────────────────────────────────────────
# Test that scaling worked — mean should be close to 0
def test_scaling_mean():
    df = load_data("data/housing.csv")
    X_scaled, y, scaler = preprocess_data(df)
    # After scaling mean of every column should be ~0
    means = X_scaled.mean().round(5)
    for col in X_scaled.columns:
        assert abs(means[col]) < 0.01, f"{col} mean should be ~0"

# ── TEST 7 ────────────────────────────────────────────────────────────
# Test that target values are positive (house prices cant be negative)
def test_target_positive():
    df = load_data("data/housing.csv")
    X_scaled, y, scaler = preprocess_data(df)
    assert (y > 0).all(), "All house prices should be positive"
