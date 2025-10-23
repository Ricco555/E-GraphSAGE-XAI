# eval_metrics.py
import numpy as np, json, os
from typing import Dict
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support,
                             classification_report, confusion_matrix, f1_score)

def per_class_far(cm: np.ndarray) -> np.ndarray:
    """
    FAR per class (one-vs-rest) = FP / (FP + TN)
    cm: confusion matrix shape (C, C)
    """
    C = cm.shape[0]
    FAR = np.zeros(C, dtype=np.float64)
    for c in range(C):
        TP = cm[c, c]
        FN = cm[c, :].sum() - TP
        FP = cm[:, c].sum() - TP
        TN = cm.sum() - (TP + FN + FP)
        FAR[c] = FP / (FP + TN) if (FP + TN) > 0 else 0.0
    return FAR

def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, label2id: Dict[str,int]):
    id2label = {v:k for k,v in label2id.items()}
    labels    = list(range(len(label2id)))
    target_names = [id2label[i] for i in labels]

    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )
    macro = {
        "precision": float(np.mean(prec)),
        "recall":    float(np.mean(rec)),
        "f1":        float(np.mean(f1)),
    }
    weighted = {
        "precision": float((prec*support).sum()/max(1,support.sum())),
        "recall":    float((rec*support).sum()/max(1,support.sum())),
        "f1":        float((f1*support).sum()/max(1,support.sum())),
    }
    cm  = confusion_matrix(y_true, y_pred, labels=labels)
    far = per_class_far(cm)
    metrics = {
        "accuracy": acc,
        "macro": macro,
        "weighted": weighted,
        "per_class": {
            target_names[i]: {
                "precision": float(prec[i]), "recall": float(rec[i]),
                "f1": float(f1[i]), "support": int(support[i]), "FAR": float(far[i])
            } for i in labels
        },
        "macro_FAR": float(far.mean()),
        "confusion_matrix": cm.tolist(),
    }
    return metrics
