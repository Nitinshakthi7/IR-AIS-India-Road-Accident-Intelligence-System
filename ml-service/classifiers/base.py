#!/usr/bin/env python3
"""
IR-AIS Classifiers — Shared Evaluation Utilities
Common metrics computation and printing for all classification models.
"""

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_auc_score,
)


def evaluate(model, X_test, y_test, approach="base"):
    """
    Evaluate a fitted classifier and return metrics dict + F1 score.

    Parameters
    ----------
    model : fitted sklearn-compatible classifier
    X_test : array-like — test features
    y_test : array-like — test labels (encoded)
    approach : str — label for the training approach used

    Returns
    -------
    metrics : dict with accuracy, precision, recall, F1, confusion matrix
    f1 : float — the weighted F1 score
    """
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_test, y_pred).tolist()

    metrics = {
        "accuracy": round(float(acc), 4),
        "precision_weighted": round(float(prec), 4),
        "recall_weighted": round(float(rec), 4),
        "f1_weighted": round(float(f1), 4),
        "confusion_matrix": cm,
        "approach": approach,
    }

    # ROC-AUC (OvR) — only if model supports predict_proba
    try:
        y_proba = model.predict_proba(X_test)
        auc = roc_auc_score(y_test, y_proba, multi_class="ovr", average="weighted")
        metrics["roc_auc_ovr"] = round(float(auc), 4)
    except Exception:
        pass

    return metrics, f1


def print_metrics(metrics):
    """Pretty-print classifier metrics to console."""
    line = (
        f"    Accuracy: {metrics['accuracy']:.4f}, "
        f"Precision: {metrics['precision_weighted']:.4f}, "
        f"Recall: {metrics['recall_weighted']:.4f}, "
        f"F1: {metrics['f1_weighted']:.4f}"
    )
    if "roc_auc_ovr" in metrics:
        line += f", ROC-AUC: {metrics['roc_auc_ovr']:.4f}"
    print(line)
