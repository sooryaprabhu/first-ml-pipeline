import joblib
import os


def save_model(model, scaler):
    # Create models folder if it doesn't exist
    os.makedirs("models", exist_ok=True)

    # Save the trained model to disk
    joblib.dump(model, "models/house_price_model.pkl")
    print("Model saved to models/house_price_model.pkl")

    # Save the scaler too
    joblib.dump(scaler, "models/scaler.pkl")
    print("Scaler saved to models/scaler.pkl")


def load_model():
    model = joblib.load("models/house_price_model.pkl")
    scaler = joblib.load("models/scaler.pkl")
    print("Model and scaler loaded from disk!")
    return model, scaler


if __name__ == "__main__":
    from load_data import load_data
    from preprocess import preprocess_data
    from train import train_model
    df = load_data("data/housing.csv")
    X_scaled, y, scaler = preprocess_data(df)
    model, X_train, X_test, y_train, y_test = train_model(X_scaled, y)
    save_model(model, scaler)
    print("\nReloading from disk to verify...")
    model_loaded, scaler_loaded = load_model()
    print(f"Model type: {type(model_loaded)}")
