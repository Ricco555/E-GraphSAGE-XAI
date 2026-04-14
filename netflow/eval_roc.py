# eval_roc.py
import numpy as np
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import confusion_matrix
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

def plot_confusion_matrix(
    y_true,
    y_pred,
    labels,
    normalize=True,
    figsize=(8, 7),
    cmap="Blues",
    title=None,
    annotate=True,
    savepath=None,
):
    """
    Plot a confusion matrix heatmap.
    
    Parameters
    ----------
    y_true, y_pred : array-like, shape (n_samples,)
        Ground truth and predicted labels (as integer class IDs).
    labels : list[str]
        Class names in the same order as the integer IDs (0..K-1).
    normalize : bool
        If True, each row is normalized to sum to 1 (per-class recall proportions).
    figsize : tuple
        Figure size.
    title : str or None
        Title for the plot.
    annotate : bool
        If True, annotate each cell with the value.
    savepath : str or None
        If set, save figure to this path.
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    cm = confusion_matrix(y_true, y_pred, labels=np.arange(len(labels)))
    accuracy = np.trace(cm) / float(np.sum(cm))
    misclass = 1 - accuracy

    if normalize:
        with np.errstate(all="ignore"):
            cm_display = cm.astype("float") / cm.sum(axis=1, keepdims=True)
            cm_display = np.nan_to_num(cm_display)
        fmt = ".2f"
        cbar_label = "Proportion"
        plot_title = title or "Confusion Matrix (Normalized by True Class)"
    else:
        cm_display = cm
        fmt = "d"
        cbar_label = "Count"
        plot_title = title or "Confusion Matrix"

    fig, ax = plt.subplots(figsize=figsize)
    im = ax.imshow(cm_display, interpolation="nearest", cmap=cmap)
    cbar = fig.colorbar(im, ax=ax)
    cbar.ax.set_ylabel(cbar_label, rotation=-90, va="bottom")

    ax.set(
        xticks=np.arange(len(labels)),
        yticks=np.arange(len(labels)),
        xticklabels=labels,
        yticklabels=labels,
        ylabel="True label",
        xlabel=f'Predicted label accuracy={accuracy:0.4f}; misclass={misclass:0.4f}',
        title=plot_title,
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")

    if annotate:
        thresh = cm_display.max() / 2.0
        for i in range(cm_display.shape[0]):
            for j in range(cm_display.shape[1]):
                text_val = f"{cm_display[i, j]:{fmt}}"
                ax.text(
                    j, i, text_val,
                    ha="center", va="center",
                    color="white" if cm_display[i, j] > thresh else "black",
                    fontsize=10,
                )

    fig.tight_layout()
    if savepath:
        plt.savefig(savepath, dpi=150, bbox_inches="tight")
    plt.show()

    return cm  # return raw counts for further processing if needed