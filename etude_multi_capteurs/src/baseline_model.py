import numpy as np

from src.evaluation import regression_metrics


def previous_value_baseline(X_test, target_index):
    """Predicts the next target value with the last observed target value in each sequence."""
    if X_test.ndim != 3:
        raise ValueError("X_test must have shape (samples, sequence_length, features).")
    return X_test[:, -1, target_index].reshape(-1)


def evaluate_previous_value_baseline(X_test, y_test, target_index, model_name="NaivePreviousValue"):
    predictions = previous_value_baseline(X_test, target_index)
    metrics = regression_metrics(np.asarray(y_test), predictions, model_name)
    return predictions, metrics
