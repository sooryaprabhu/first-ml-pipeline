import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(model, X_test, y_test):
    # Make predictions on houses the model has never seen
    y_pred = model.predict(X_test)

    # How far off are we on average?
    mae = mean_absolute_error(y_test, y_pred)

    # Same but punishes big mistakes harder
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))

    # How much of price variation does our model explain?
    r2 = r2_score(y_test, y_pred)

    print("── Model Evaluation ──────────────────────────")
    print(f"MAE  (avg error):        ${mae * 100000:,.0f}")
    print(f"RMSE (punishes big err): ${rmse * 100000:,.0f}")
    print(f"R2   (explanation):       {r2:.4f}")
    print("──────────────────────────────────────────────")

    return y_pred


if __name__ == "__main__":
    from load_data import load_data
    from preprocess import preprocess_data
    from train import train_model
    df = load_data("data/housing.csv")
    X_scaled, y, scaler = preprocess_data(df)
    model, X_train, X_test, y_train, y_test = train_model(X_scaled, y)
    y_pred = evaluate_model(model, X_test, y_test)
