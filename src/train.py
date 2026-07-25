import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

def train_model(X_scaled, y):
    # Split data — 80% train, 20% test
    # We can't test on data the model already saw — that's cheating
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    print(f"Training set size: {X_train.shape[0]} rows")
    print(f"Testing set size:  {X_test.shape[0]} rows")

    # Train a Linear Regression model
    # .fit() is where the actual learning happens
    model = LinearRegression()
    model.fit(X_train, y_train)

    print("\nModel trained successfully!")
    print("\nModel coefficients (weights per feature):")
    for feature, coef in zip(X_scaled.columns, model.coef_):
        print(f"  {feature:15s}: {coef:.4f}")

    return model, X_train, X_test, y_train, y_test

if __name__ == "__main__":
    from load_data import load_data
    from preprocess import preprocess_data
    df = load_data("data/housing.csv")
    X_scaled, y, scaler = preprocess_data(df)
    model, X_train, X_test, y_train, y_test = train_model(X_scaled, y)
