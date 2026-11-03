
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score
)


def calculate_metrics(y_true, y_pred, y_probability):
    """Calculate classification metrics for fraud detection."""

    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(
            precision_score(y_true, y_pred, zero_division=0)
        ),
        "recall": float(
            recall_score(y_true, y_pred, zero_division=0)
        ),
        "f1_score": float(
            f1_score(y_true, y_pred, zero_division=0)
        ),
        "average_precision": float(
            average_precision_score(y_true, y_probability)
        ),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred
        ).tolist()
    }


def validate_transaction_data(data, feature_names):
    """Check whether uploaded data has the required features."""

    if not isinstance(data, pd.DataFrame):
        return False, "Uploaded data must be a CSV table."

    missing = [
        column for column in feature_names
        if column not in data.columns
    ]

    if missing:
        return False, (
            "Missing required columns: "
            + ", ".join(missing)
        )

    if data.empty:
        return False, "The uploaded CSV contains no transactions."

    if data[feature_names].isnull().any().any():
        return False, "Some required features contain missing values."

    if not all(
        pd.api.types.is_numeric_dtype(data[col])
        for col in feature_names
    ):
        return False, "All required features must be numeric."

    if not data[feature_names].apply(
        lambda col: col.map(
            lambda value: pd.notna(value)
            and abs(float(value)) != float("inf")
        ).all()
    ).all():
        return False, "Features must contain finite numeric values."

    return True, "Transaction data is valid."
