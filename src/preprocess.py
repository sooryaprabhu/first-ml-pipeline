import pandas as pd
from sklearn.preprocessing import StandardScaler


def preprocess_data(df):
    # X = features, y = target
    X = df.drop(columns=["MedHouseVal"])
    y = df["MedHouseVal"]

    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")

    # Check for missing values
    if X.isnull().sum().sum() == 0:
        print("\nNo missing values — we're good!")
    else:
        X = X.fillna(X.median())

    # Scale all features to same range
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)

    print("\nAfter scaling — first 5 rows:")
    print(X_scaled.head())

    return X_scaled, y, scaler


if __name__ == "__main__":
    from load_data import load_data
    df = load_data("data/housing.csv")
    X_scaled, y, scaler = preprocess_data(df)
