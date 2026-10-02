"""
evaluate.py — Metrics, bootstrap CIs, and significance tests.

Computes all required metrics for binary, multiclass, and retrieval tasks.
Includes bootstrap confidence intervals and statistical significance tests.
"""

import logging
import warnings
from typing import List, Dict, Optional, Tuple, Any

import numpy as np
from scipy import stats as scipy_stats
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    matthews_corrcoef, cohen_kappa_score,
    roc_auc_score, average_precision_score,
    log_loss, brier_score_loss,
    confusion_matrix, classification_report
)

logger = logging.getLogger(__name__)


# =====================================================================
# IDEA 7: Small-data honesty
# WHY:     The corpus is small; metrics may be unstable. Reporting
#          point estimates without CIs would be misleading.
# HOW:     Repeated stratified k-fold (10×5 default), bootstrap 95%
#          CIs on pooled out-of-fold predictions, and automatic
#          "low-power" warnings when any class has few samples.
# OUTPUT:  CIs in all metric tables, warnings in REPORT.md
# CAVEAT:  Bootstrap CIs on small data are themselves noisy. The
#          "low-power" threshold is somewhat arbitrary (< 20 samples).
# =====================================================================


def compute_binary_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                          y_prob: Optional[np.ndarray] = None,
                          pos_label: int = 1) -> Dict[str, float]:
    """
    Compute all binary classification metrics.

    Metrics: accuracy, balanced_accuracy, precision, recall, specificity,
    f1, mcc, kappa, roc_auc, pr_auc, log_loss, brier_score.

    Parameters
    ----------
    y_true : array-like
        True labels.
    y_pred : array-like
        Predicted labels.
    y_prob : array-like, optional
        Predicted probabilities for the positive class.
    pos_label : int
        Positive class label.

    Returns
    -------
    Dict[str, float]
        Named metric values.
    """
    metrics = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)
        metrics['precision'] = precision_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        metrics['recall'] = recall_score(y_true, y_pred, pos_label=pos_label, zero_division=0)

        # Specificity = recall of the negative class
        tn, fp, fn, tp = 0, 0, 0, 0
        cm = confusion_matrix(y_true, y_pred)
        if cm.shape == (2, 2):
            tn, fp, fn, tp = cm.ravel()
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
        else:
            metrics['specificity'] = 0.0

        metrics['f1'] = f1_score(y_true, y_pred, pos_label=pos_label, zero_division=0)
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        metrics['kappa'] = cohen_kappa_score(y_true, y_pred)

        if y_prob is not None and len(np.unique(y_true)) > 1:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_prob)
            except ValueError:
                metrics['roc_auc'] = float('nan')
            try:
                metrics['pr_auc'] = average_precision_score(y_true, y_prob)
            except ValueError:
                metrics['pr_auc'] = float('nan')
            try:
                metrics['log_loss'] = log_loss(y_true, y_prob)
            except ValueError:
                metrics['log_loss'] = float('nan')
            try:
                metrics['brier_score'] = brier_score_loss(y_true, y_prob)
            except ValueError:
                metrics['brier_score'] = float('nan')
        else:
            metrics['roc_auc'] = float('nan')
            metrics['pr_auc'] = float('nan')
            metrics['log_loss'] = float('nan')
            metrics['brier_score'] = float('nan')

    return metrics


def compute_multiclass_metrics(y_true: np.ndarray, y_pred: np.ndarray,
                               y_prob: Optional[np.ndarray] = None,
                               labels: Optional[List] = None) -> Dict[str, float]:
    """
    Compute multiclass classification metrics.

    Uses macro, micro, and weighted averaging plus per-class values.
    """
    metrics = {}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')

        metrics['accuracy'] = accuracy_score(y_true, y_pred)
        metrics['balanced_accuracy'] = balanced_accuracy_score(y_true, y_pred)

        for avg in ['macro', 'micro', 'weighted']:
            metrics[f'precision_{avg}'] = precision_score(y_true, y_pred, average=avg, zero_division=0)
            metrics[f'recall_{avg}'] = recall_score(y_true, y_pred, average=avg, zero_division=0)
            metrics[f'f1_{avg}'] = f1_score(y_true, y_pred, average=avg, zero_division=0)

        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        metrics['kappa'] = cohen_kappa_score(y_true, y_pred)

        if y_prob is not None and len(np.unique(y_true)) > 1:
            try:
                metrics['roc_auc_ovr'] = roc_auc_score(
                    y_true, y_prob, multi_class='ovr', average='macro'
                )
            except (ValueError, Exception):
                metrics['roc_auc_ovr'] = float('nan')

    return metrics


def compute_retrieval_metrics(y_true: np.ndarray, rankings: List[List[int]],
                              top_k_values: List[int] = [1, 3, 5]) -> Dict[str, float]:
    """
    Compute retrieval metrics for Task C.

    Metrics: top-k accuracy, MRR, macro-F1.

    Parameters
    ----------
    y_true : array-like
        True song IDs.
    rankings : List[List[int]]
        For each query, the ranked list of predicted song IDs.
    top_k_values : List[int]
        Values of k for top-k accuracy.

    Returns
    -------
    Dict[str, float]
        Named metric values.
    """
    metrics = {}
    n = len(y_true)

    for k in top_k_values:
        correct = sum(1 for i in range(n) if y_true[i] in rankings[i][:k])
        metrics[f'top_{k}_accuracy'] = correct / max(n, 1)

    # MRR
    reciprocal_ranks = []
    for i in range(n):
        try:
            rank = rankings[i].index(y_true[i]) + 1
            reciprocal_ranks.append(1.0 / rank)
        except ValueError:
            reciprocal_ranks.append(0.0)
    metrics['mrr'] = np.mean(reciprocal_ranks) if reciprocal_ranks else 0.0

    return metrics


def bootstrap_ci(y_true: np.ndarray, y_pred: np.ndarray,
                 metric_fn, n_bootstrap: int = 1000,
                 confidence: float = 0.95,
                 random_state: int = 42,
                 **metric_kwargs) -> Tuple[float, float, float]:
    """
    Compute bootstrap confidence interval for a metric.

    Parameters
    ----------
    y_true, y_pred : array-like
        True and predicted labels.
    metric_fn : callable
        Metric function(y_true, y_pred, **kwargs) -> float.
    n_bootstrap : int
        Number of bootstrap iterations.
    confidence : float
        Confidence level (e.g. 0.95 for 95% CI).
    random_state : int
        Random seed.

    Returns
    -------
    Tuple[float, float, float]
        (point_estimate, ci_lower, ci_upper)
    """
    rng = np.random.RandomState(random_state)
    n = len(y_true)
    point = metric_fn(y_true, y_pred, **metric_kwargs)

    scores = []
    for _ in range(n_bootstrap):
        idx = rng.choice(n, n, replace=True)
        try:
            s = metric_fn(y_true[idx], y_pred[idx], **metric_kwargs)
            scores.append(s)
        except (ValueError, ZeroDivisionError):
            continue

    if not scores:
        return point, point, point

    alpha = (1 - confidence) / 2
    ci_lower = np.percentile(scores, 100 * alpha)
    ci_upper = np.percentile(scores, 100 * (1 - alpha))
    return point, ci_lower, ci_upper


def paired_significance_test(scores_a: List[float], scores_b: List[float],
                             method: str = 'wilcoxon') -> Dict[str, float]:
    """
    Paired significance test over CV folds.

    Performs Wilcoxon signed-rank and paired t-test.

    Parameters
    ----------
    scores_a, scores_b : List[float]
        Per-fold metric values for systems A and B.

    Returns
    -------
    Dict with p-values and effect sizes.
    """
    a = np.array(scores_a)
    b = np.array(scores_b)
    diff = a - b

    result = {
        'mean_diff': np.mean(diff),
        'std_diff': np.std(diff),
        'effect_size_cohens_d': np.mean(diff) / max(np.std(diff), 1e-10),
    }

    # Paired t-test
    if len(a) >= 2:
        try:
            t_stat, t_pval = scipy_stats.ttest_rel(a, b)
            result['ttest_statistic'] = t_stat
            result['ttest_pvalue'] = t_pval
        except Exception:
            result['ttest_pvalue'] = float('nan')

    # Wilcoxon signed-rank test
    if len(a) >= 6 and np.any(diff != 0):
        try:
            w_stat, w_pval = scipy_stats.wilcoxon(a, b)
            result['wilcoxon_statistic'] = w_stat
            result['wilcoxon_pvalue'] = w_pval
        except Exception:
            result['wilcoxon_pvalue'] = float('nan')
    else:
        result['wilcoxon_pvalue'] = float('nan')

    return result


def holm_bonferroni_correction(p_values: List[float],
                               alpha: float = 0.05) -> List[Tuple[float, bool]]:
    """
    Apply Holm-Bonferroni correction for multiple comparisons.

    Parameters
    ----------
    p_values : List[float]
        Raw p-values.
    alpha : float
        Family-wise error rate.

    Returns
    -------
    List[Tuple[float, bool]]
        (adjusted_p, significant) for each test.
    """
    n = len(p_values)
    indexed = sorted(enumerate(p_values), key=lambda x: x[1])
    results = [None] * n

    for rank, (orig_idx, pval) in enumerate(indexed):
        adjusted_alpha = alpha / (n - rank)
        significant = pval <= adjusted_alpha
        adjusted_p = min(pval * (n - rank), 1.0)
        results[orig_idx] = (adjusted_p, significant)

    return results


def check_low_power(y: np.ndarray, min_samples: int = 20) -> List[str]:
    """
    Check for low-power classes (too few samples for reliable metrics).

    Returns list of warning messages.
    """
    warnings_list = []
    unique, counts = np.unique(y, return_counts=True)
    for label, count in zip(unique, counts):
        if count < min_samples:
            warnings_list.append(
                f"⚠ LOW-POWER WARNING: Class '{label}' has only {count} samples "
                f"(< {min_samples}). Metrics may be unreliable."
            )
    return warnings_list
