import pandas as pd

def load_data(filepath):
    # Read the CSV file into a DataFrame (like loading a spreadsheet)
    df = pd.read_csv(filepath)

    print(f"Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # First 5 rows — quick look at what the data actually looks like
    print("\nFirst 5 rows:")
    print(df.head())

    # All column names — these are the features we'll use to predict price
    print("\nColumn names:")
    print(df.columns.tolist())

    # Count missing values per column — missing values break ML models
    print("\nMissing values per column:")
    print(df.isnull().sum())

    return df

if __name__ == "__main__":
    df = load_data("data/housing.csv")
