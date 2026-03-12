import numpy as np

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score
)


def evaluate_model(model, X_val, y_val):
    

    # Predictions
    y_pred = model.predict(X_val)

    # Probabilities (needed for ROC-AUC)
    y_prob = model.predict_proba(X_val)[:, 1]

    # Metrics
    acc = accuracy_score(y_val, y_pred)
    roc = roc_auc_score(y_val, y_prob)

    print("\n===== MODEL EVALUATION =====")

    print(f"\nAccuracy: {acc:.4f}")
    print(f"ROC-AUC: {roc:.4f}")

    print("\nClassification Report:")
    print(classification_report(y_val, y_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(y_val, y_pred))

    return {
        "accuracy": acc,
        "roc_auc": roc
    }