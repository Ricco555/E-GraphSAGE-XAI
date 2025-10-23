# eval_roc.py
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

def plot_roc_ovr(y_true, y_prob, classes, out_path=None):
    """
    y_prob: (N,C) probabilities
    classes: list[str] class names in order [0..C-1]
    """
    C = len(classes)
    y_bin = label_binarize(y_true, classes=list(range(C)))  # (N,C)
    fpr, tpr, roc_auc = {}, {}, {}
    for i in range(C):
        if y_bin[:, i].sum() == 0:
            continue
        fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])

    plt.figure(figsize=(8,6))
    for i in range(C):
        if i not in fpr: continue
        plt.plot(fpr[i], tpr[i], lw=1.5, label=f"{classes[i]} (AUC={roc_auc[i]:.3f})")
    plt.plot([0,1], [0,1], 'k--', lw=1)
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("ROC curves (one-vs-rest)")
    plt.legend(loc="best")
    if out_path: plt.savefig(out_path, bbox_inches="tight", dpi=150)
    plt.show()

def false_negatives_by_class(cm: np.ndarray, classes: list[str]) -> dict:
    FN = {}
    for i, name in enumerate(classes):
        tp = cm[i, i]
        row_sum = cm[i, :].sum()
        fn = row_sum - tp
        FN[name] = int(fn)
    return FN
