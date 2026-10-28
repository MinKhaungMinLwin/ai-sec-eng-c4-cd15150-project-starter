"""
audit.py — Subgroup, intersectional, and proxy-pathway analysis.

These functions assemble the metrics from fairness_metrics.py into the kind
of analysis a real AI Risk Officer would run.

Run the unit tests to check your work:
    cd starter/
    pytest lab/tests/test_audit.py -v
"""
from __future__ import annotations
from typing import Optional
import numpy as np
import pandas as pd

# You will import your implementations from the same package
from .fairness_metrics import (
    false_negative_rate, false_positive_rate, selection_rate,
)


def compute_subgroup_metrics(
    y_true,
    y_pred,
    sensitive_features,
) -> pd.DataFrame:
    """
    Compute per-subgroup metrics and return them as a table.

    What the input looks like. The sensitive_features argument can be either
    a single column of group labels (single-axis analysis — e.g., region by
    itself) OR a multi-column structure of group labels (intersectional
    analysis — e.g., region together with insurance type). Your function
    must handle both shapes.

    For each unique value or combination of values in the sensitive
    features, compute against the rows belonging to that subgroup:
        - the total number of rows in the subgroup,
        - the count of actual positives in the subgroup,
        - the selection rate,
        - the false-negative rate,
        - the false-positive rate,
        - the within-subgroup accuracy.

    Why n_positives matters. The false-negative rate of a cell with five
    positives is far less stable than the false-negative rate of a cell
    with fifty. Always carry the positive count alongside the metric so
    downstream code can apply a statistical-power filter.

    Returns:
        A table with one row per subgroup, indexed by the subgroup label
        (single-axis) or the combination of labels (intersectional).
    """
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    # Normalize both shapes to a DataFrame of group columns aligned
    # positionally with y_true / y_pred.
    if isinstance(sensitive_features, pd.DataFrame):
        groups = sensitive_features.reset_index(drop=True)
    else:
        s = pd.Series(np.asarray(sensitive_features))
        name = getattr(sensitive_features, "name", None) or "subgroup"
        groups = s.to_frame(name=name)
    if len(groups) != len(y_true) or len(y_true) != len(y_pred):
        raise ValueError("y_true, y_pred and sensitive_features must be the same length")

    keys = list(groups.columns)
    rows = []
    for label, idx in groups.groupby(keys, sort=True).indices.items():
        yt, yp = y_true[idx], y_pred[idx]
        rows.append({
            "_label": label,
            "n_total": int(len(idx)),
            "n_positives": int(yt.sum()),
            "selection_rate": selection_rate(yp),
            "FNR": false_negative_rate(yt, yp),
            "FPR": false_positive_rate(yt, yp),
            "accuracy": float((yt == yp).mean()),
        })

    labels = [r.pop("_label") for r in rows]
    if len(keys) == 1:
        # groupby on a one-element key list yields 1-tuples in recent pandas
        labels = [l[0] if isinstance(l, tuple) else l for l in labels]
        index = pd.Index(labels, name=keys[0])
    else:
        index = pd.MultiIndex.from_tuples(labels, names=keys)
    return pd.DataFrame(rows, index=index)[
        ["n_total", "n_positives", "selection_rate", "FNR", "FPR", "accuracy"]
    ]


def find_most_harmed_subgroup(
    subgroup_df: pd.DataFrame,
    metric_name: str = "FNR",
    min_n_positives: int = 20,
    higher_is_worse: bool = True,
) -> Optional[dict]:
    """
    Return the subgroup row with the worst value of `metric_name`,
    AMONG subgroups that meet a minimum-positive-count threshold.

    Why this matters — the small-n trap. A cell with five positives can
    show a very high false-negative rate just by random sampling — three
    more or fewer false negatives swing it twenty percentage points. Calling
    that cell "most harmed" without statistical power is a finding you
    cannot defend in a launch-decision meeting.

    The minimum-positive-count filter is a deliberate choice. Document the
    choice in your D8 limitations, and surface it as a residual risk in D5:
    the smallest subgroup cells are not statistically powered, so disparities
    in those cells cannot be quantified.

    Args:
        subgroup_df: the table produced by compute_subgroup_metrics.
        metric_name: which column to maximize (or minimize, depending on
            higher_is_worse).
        min_n_positives: minimum positive count required for a row to be
            considered.
        higher_is_worse: True for false-negative rate and false-positive
            rate; False for accuracy or selection rate, when applicable.

    Returns:
        A dictionary describing the most-harmed subgroup — the subgroup
        label, which metric was used, the metric value, the positive count,
        and the total count. Return nothing (None) if no subgroup meets
        the minimum-positive-count threshold.
    """
    if metric_name not in subgroup_df.columns:
        raise KeyError(f"metric {metric_name!r} not in subgroup table")
    eligible = subgroup_df[subgroup_df["n_positives"] >= min_n_positives]
    eligible = eligible[eligible[metric_name].notna()]
    if eligible.empty:
        return None

    metric = eligible[metric_name].astype(float)
    label = metric.idxmax() if higher_is_worse else metric.idxmin()
    row = eligible.loc[label]
    return {
        "subgroup": label,
        "metric": metric_name,
        "metric_value": float(row[metric_name]),
        "n_positives": int(row["n_positives"]),
        "n_total": int(row["n_total"]),
        "min_n_positives": int(min_n_positives),
        "n_eligible": int(len(eligible)),
        "n_suppressed": int(len(subgroup_df) - len(eligible)),
    }


def assess_proxy_pathway(
    intersectional_df: pd.DataFrame,
    primary_axis_value,
    secondary_axis_low,
    secondary_axis_high,
    metric_name: str = "FNR",
) -> dict:
    """
    Compare the within-primary-axis disparity to the across-primary-
    axis baseline, and return a verdict on whether a proxy pathway is
    consistent with the data.

    Why this matters — this IS the proxy-bias test, in code.

    The reasoning. Suppose the model has false-negative-rate disparities
    across the primary axis (say, region). Two explanations are possible:
        (a) the primary axis is the operative variable, OR
        (b) a second variable that is correlated with the primary axis
            is the operative variable — i.e., a proxy for the primary axis.

    If you slice within a single value of the primary axis and find that
    the metric varies just as much across the secondary variable, then the
    secondary variable is a candidate proxy. If the metric does not vary
    within the primary axis, the primary axis is likely doing the work
    directly.

    What to compute. Look up the metric value at two specific cells of the
    intersectional table:
        - the cell where the primary axis equals primary_axis_value AND the
          secondary axis equals secondary_axis_low,
        - the cell where the primary axis equals primary_axis_value AND the
          secondary axis equals secondary_axis_high.
    The within-axis gap is the difference between these two values.

    Then compute the same gap for a reference primary value (a value other
    than primary_axis_value — pick the one with the largest sample size so
    your reference is stable). The reference gap tells you what the
    secondary-axis effect looks like in a population where you do NOT
    expect the proxy mechanism to be operating.

    Decision rule.
        - If the within-axis gap is meaningfully larger than the reference
          gap (and both are positive), call the proxy pathway supported.
        - If the within-axis gap is comparable to the reference gap, call
          it inconsistent with the proxy hypothesis.
        - If a required cell is missing or the gap is not directional,
          call it inconclusive and explain what is missing.

    Returns:
        A dictionary with the within-axis gap (in percentage points), the
        reference gap (in percentage points), the verdict, the reference
        primary value used, and a short rationale string.
    """
    # A within-axis gap must exceed the reference gap by at least this many
    # percentage points before we call the proxy pathway supported.
    margin_pp = 5.0

    def _cell(primary, secondary):
        try:
            row = intersectional_df.loc[(primary, secondary)]
        except KeyError:
            return None
        if isinstance(row, pd.DataFrame):
            row = row.iloc[0]
        value = row.get(metric_name)
        return None if value is None or pd.isna(value) else row

    def _gap_pp(primary):
        low, high = _cell(primary, secondary_axis_low), _cell(primary, secondary_axis_high)
        if low is None or high is None:
            return None, None, None
        return 100.0 * (float(low[metric_name]) - float(high[metric_name])), low, high

    result = {
        "metric": metric_name,
        "primary_axis_value": primary_axis_value,
        "secondary_axis_low": secondary_axis_low,
        "secondary_axis_high": secondary_axis_high,
        "within_gap_pp": float("nan"),
        "reference_gap_pp": float("nan"),
        "reference_primary_value": None,
        "verdict": "inconclusive",
        "rationale": "",
    }

    within, w_low, w_high = _gap_pp(primary_axis_value)
    if within is None:
        result["rationale"] = (
            f"Missing or undefined {metric_name} for ({primary_axis_value}, "
            f"{secondary_axis_low}) or ({primary_axis_value}, {secondary_axis_high}).")
        return result
    result["within_gap_pp"] = within
    result["within_n_positives"] = (int(w_low["n_positives"]), int(w_high["n_positives"]))

    # Reference = the other primary value with the largest sample that has
    # both secondary cells populated.
    primary_level = intersectional_df.index.get_level_values(0)
    sizes = (intersectional_df["n_total"].groupby(primary_level).sum()
             .drop(primary_axis_value, errors="ignore")
             .sort_values(ascending=False))
    reference_value, reference = None, None
    for candidate in sizes.index:
        gap, _, _ = _gap_pp(candidate)
        if gap is not None:
            reference_value, reference = candidate, gap
            break
    if reference is None:
        result["rationale"] = (
            "No reference primary value has both secondary cells populated; "
            "cannot separate the proxy effect from the baseline.")
        return result
    result["reference_gap_pp"] = reference
    result["reference_primary_value"] = reference_value

    if within <= 0:
        result["rationale"] = (
            f"Within-{primary_axis_value} gap is {within:+.1f} pp — not in the "
            f"hypothesised direction ({secondary_axis_low} worse than {secondary_axis_high}).")
    # A reference gap at or below zero means the secondary axis carries no
    # harm in the reference population — the strongest form of support.
    elif within - reference >= margin_pp:
        result["verdict"] = "proxy_supported"
        result["rationale"] = (
            f"Within {primary_axis_value}, {secondary_axis_low} {metric_name} exceeds "
            f"{secondary_axis_high} by {within:+.1f} pp vs {reference:+.1f} pp in "
            f"{reference_value}; the secondary axis amplifies harm beyond baseline.")
    else:
        result["verdict"] = "inconsistent_with_proxy"
        result["rationale"] = (
            f"Within-{primary_axis_value} gap ({within:+.1f} pp) is comparable to the "
            f"{reference_value} reference gap ({reference:+.1f} pp); no amplification.")
    return result
