import pandas as pd


def load_data(filepath):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(filepath)

    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # First 5 rows
    print("\nFirst 5 rows:")
    print(df.head())

    # All column names
    print("\nColumn names:")
    print(df.columns.tolist())

    # Count missing values per column
    print("\nMissing values per column:")
    print(df.isnull().sum())

    return df


if __name__ == "__main__":
    df = load_data("data/housing.csv")
