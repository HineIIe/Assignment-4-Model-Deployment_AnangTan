import os
import pickle

from sklearn.model_selection import train_test_split

from src.ingest import load_data
from src.preprocess import preprocess_data, build_preprocessor
from src.train import tune_logistic_regression, train_final_model
from src.evaluate import evaluate_model



RANDOM_STATE = 42


def run_pipeline():

    print("========== STARTING PIPELINE ==========")

    # =========================
    # 1 Load Data
    # =========================

    df = load_data("data/train.csv")

    # =========================
    # 2 Preprocess Data
    # =========================

    X, y, cat_features, num_features = preprocess_data(df)

    print("\nFeatures prepared!")
    print(f"Feature shape: {X.shape}")

    # =========================
    # 3 Train / Validation Split
    # =========================

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        stratify=y,
        random_state=RANDOM_STATE
    )

    print("\nData split completed")
    print(f"Train size: {X_train.shape}")
    print(f"Validation size: {X_val.shape}")

    # =========================
    # 4 Build Preprocessor
    # =========================

    preprocessor = build_preprocessor(cat_features, num_features)

    X_train_processed = preprocessor.fit_transform(X_train)
    X_val_processed = preprocessor.transform(X_val)

    print("\nPreprocessing completed!")

    # =========================
    # 5 Hyperparameter Tuning
    # =========================

    best_params = tune_logistic_regression(X_train_processed, y_train)

    # =========================
    # 6 Train Final Model
    # =========================

    model = train_final_model(X_train_processed, y_train, best_params)

    # =========================
    # 7 Evaluate Model
    # =========================

    evaluate_model(model, X_val_processed, y_val)

    # =========================
    # 8 Save Model and Preprocessor
    # =========================

    os.makedirs("models", exist_ok=True)

    with open("models/model.pkl", "wb") as f:
        pickle.dump(model, f)

    with open("models/preprocessor.pkl", "wb") as f:
        pickle.dump(preprocessor, f)

    print("\nModel and Preprocessor saved successfully!")

    print("========== PIPELINE FINISHED ==========")


if __name__ == "__main__":
    run_pipeline()