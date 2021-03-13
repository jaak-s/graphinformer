import sklearn.metrics
import numpy as np

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def roc_auc(y_true, y_score, ignore_index):
    keep = (y_true != ignore_index)
    if (y_true[keep][0] == y_true[keep]).all():
        return np.nan
    return sklearn.metrics.roc_auc_score(
        y_true  = y_true[keep],
        y_score = y_score[keep]
    )

def pr_auc(y_true, y_score, ignore_index):
    """PR-AUC following DeepChem implementation."""
    keep = (y_true != ignore_index)
    if (y_true[keep][0] == y_true[keep]).all():
        return np.nan
    pr, re, _ = sklearn.metrics.precision_recall_curve(y_true[keep], y_score[keep])
    return sklearn.metrics.auc(re, pr)

def avg_prec(y_true, y_score, ignore_index):
    """PR-AUC following DeepChem implementation."""
    keep = (y_true != ignore_index)
    if (y_true[keep][0] == y_true[keep]).all():
        return np.nan
    return sklearn.metrics.average_precision_score(
        y_true  = y_true[keep],
        y_score = y_score[keep]
    )

def roc_auc_mt(y_true, y_score, ignore_index):
    """Multi-task setup. Calculating ROC-AUC for each task."""
    return np.array([roc_auc(
            y_true  = y_true[:,i],
            y_score = y_score[:,i],
            ignore_index = ignore_index)
        for i in range(y_true.shape[-1])
    ])

def pr_auc_mt(y_true, y_score, ignore_index):
    """Multi-task setup. Calculating PR-AUC for each task."""
    return np.array([pr_auc(
            y_true  = y_true[:,i],
            y_score = y_score[:,i],
            ignore_index = ignore_index)
        for i in range(y_true.shape[-1])
    ])

def avg_prec_mt(y_true, y_score, ignore_index):
    """Multi-task setup. Calculating PR-AUC for each task."""
    return np.array([avg_prec(
            y_true  = y_true[:,i],
            y_score = y_score[:,i],
            ignore_index = ignore_index)
        for i in range(y_true.shape[-1])
    ])

