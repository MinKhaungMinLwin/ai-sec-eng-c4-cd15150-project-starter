"""
fairness_metrics.py — Implement fairness metrics from confusion-matrix primitives.

You will use these in the technical audit (Task 4). Do NOT import these from
fairlearn or sklearn.metrics — implement them yourself. The point of this lab
is to make sure you understand what each metric *is* before you start
interpreting its values.

Run the unit tests to check your work:
    cd starter/
    pytest lab/tests/test_fairness_metrics.py -v
"""
from __future__ import annotations
import numpy as np
import pandas as pd


def false_negative_rate(y_true, y_pred) -> float:
    """
    Compute the false-negative rate.

    Definition. The false-negative rate is the fraction of actual positives
    that the model failed to flag — among patients who truly have diabetes,
    the share the model predicted as not-at-risk.

    Why this matters in HealthGuard. A false negative is a missed diagnosis,
    the dominant clinical harm we are trying to avoid. Your D8 launch
    decision will hinge on the false-negative pattern across subgroups.

    Edge case. If the input has no actual positives, the metric is
    mathematically undefined. Do not return zero — that would falsely
    suggest perfect performance on a group where you simply have no signal.
    Return a not-a-number value to signal "undefined" to downstream code.

    Args:
        y_true: ground-truth labels (0 or 1).
        y_pred: predicted labels after thresholding (0 or 1).

    Returns:
        A single number between zero and one, or a not-a-number value when
        the metric is undefined.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    positives = y_true == 1
    n_pos = int(positives.sum())
    if n_pos == 0:
        return float("nan")
    fn = int((y_pred[positives] == 0).sum())
    return fn / n_pos


def false_positive_rate(y_true, y_pred) -> float:
    """
    Compute the false-positive rate.

    Definition. The fraction of actual negatives the model incorrectly
    flagged — among patients who do NOT have diabetes, the share the model
    predicted as at-risk.

    Why this matters. False-positive rate measures unnecessary risk flags.
    A high false-positive rate means clinicians are alerted on patients who
    do not have diabetes — alert fatigue, wasted referrals, lost trust in
    the tool. It is the symmetric counterpart to false-negative rate.

    Edge case. Return a not-a-number value when there are no actual
    negatives in the input.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    negatives = y_true == 0
    n_neg = int(negatives.sum())
    if n_neg == 0:
        return float("nan")
    fp = int((y_pred[negatives] == 1).sum())
    return fp / n_neg


def selection_rate(y_pred) -> float:
    """
    Compute the selection rate.

    Definition. The fraction of inputs the model flagged as positive,
    regardless of whether the prediction was correct.

    Why this matters. Selection rate is the simplest fairness baseline —
    is the model recommending action at similar rates across groups?
    It is the basis for *demographic parity*, the parity-style fairness
    definition that ignores ground truth.
    """
    y_pred = np.asarray(y_pred).astype(int)
    if y_pred.size == 0:
        return float("nan")
    return float(y_pred.mean())


def demographic_parity_difference(y_pred, sensitive) -> dict:
    """
    Compute the demographic-parity difference across sensitive groups.

    Definition. The largest gap in selection rate between any two groups
    in the input — the most-favored group's selection rate minus the
    least-favored group's selection rate.

    Why this matters. This is one of the canonical group-fairness measures.
    A demographic-parity difference of zero means every subgroup gets
    flagged at the same rate. Demographic parity is *outcome-blind* — it
    does not look at whether the model is correct.

    Args:
        y_pred: predicted labels (0 or 1).
        sensitive: group label for each row (same length as y_pred).

    Returns:
        A dictionary that includes the difference, the label of the
        most-favored group, the label of the least-favored group, and a
        per-group breakdown of selection rates so the caller can inspect
        all groups, not just the extremes.
    """
    y_pred = np.asarray(y_pred).astype(int)
    sensitive = np.asarray(sensitive)
    by_group = {
        g: selection_rate(y_pred[sensitive == g]) for g in pd.unique(sensitive)
    }
    rates = pd.Series(by_group, dtype=float)
    max_group, min_group = rates.idxmax(), rates.idxmin()
    return {
        "difference": float(rates[max_group] - rates[min_group]),
        "max_group": max_group,
        "min_group": min_group,
        "max_rate": float(rates[max_group]),
        "min_rate": float(rates[min_group]),
        "by_group": by_group,
    }


def equalized_odds_difference(y_true, y_pred, sensitive) -> dict:
    """
    Compute the equalized-odds difference across sensitive groups.

    Definition. The larger of two cross-group gaps:
        - the gap in false-positive rate between groups, and
        - the gap in false-negative rate between groups.

    Why this matters. Equalized odds is a stricter fairness definition than
    demographic parity because it requires equal error rates on BOTH the
    positive and negative class. It accounts for ground truth. A model can
    pass demographic parity and still fail equalized odds if it makes
    different *kinds* of errors for different groups — exactly what we
    expect to find for HealthGuard.

    Returns:
        A dictionary that includes the headline equalized-odds difference,
        the false-negative-rate gap and false-positive-rate gap separately,
        an indicator of which of the two is the larger driver, and the
        per-group breakdowns of both rates.
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    sensitive = np.asarray(sensitive)
    groups = pd.unique(sensitive)
    fnr_by_group = {
        g: false_negative_rate(y_true[sensitive == g], y_pred[sensitive == g])
        for g in groups
    }
    fpr_by_group = {
        g: false_positive_rate(y_true[sensitive == g], y_pred[sensitive == g])
        for g in groups
    }
    # Groups where a rate is undefined (nan) carry no signal for that gap.
    fnr = pd.Series(fnr_by_group, dtype=float).dropna()
    fpr = pd.Series(fpr_by_group, dtype=float).dropna()
    fnr_diff = float(fnr.max() - fnr.min()) if len(fnr) else float("nan")
    fpr_diff = float(fpr.max() - fpr.min()) if len(fpr) else float("nan")
    if np.isnan(fnr_diff) and np.isnan(fpr_diff):
        difference, driver = float("nan"), None
    elif np.isnan(fpr_diff) or (not np.isnan(fnr_diff) and fnr_diff >= fpr_diff):
        difference, driver = fnr_diff, "FNR"
    else:
        difference, driver = fpr_diff, "FPR"
    return {
        "difference": difference,
        "fnr_difference": fnr_diff,
        "fpr_difference": fpr_diff,
        "driver": driver,
        "fnr_by_group": fnr_by_group,
        "fpr_by_group": fpr_by_group,
    }
