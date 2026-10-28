"""
shap_analysis.py — SHAP setup and proxy-feature detection.

The conceptual jump in this lab is the proxy-feature detector. Anyone can
call a SHAP plotting helper. Encoding what a proxy feature *looks like* in
code is the hard part — and is exactly the skill an AI Risk Officer is
hired for.

Run the unit tests to check your work:
    cd starter/
    pytest lab/tests/test_shap_analysis.py -v
"""
from __future__ import annotations
from typing import List, Sequence
import numpy as np
import pandas as pd
import shap


def _dense(X):
    """Return a dense ndarray (one-hot encoders may emit sparse matrices)."""
    return X.toarray() if hasattr(X, "toarray") else np.asarray(X)


def _positive_class(shap_values) -> np.ndarray:
    """Normalize SHAP output to a 2-D (n_samples, n_features) array for class 1."""
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    values = np.asarray(shap_values)
    if values.ndim == 3:
        values = values[:, :, -1]
    return values


def setup_explainer(pipeline, X_background: pd.DataFrame):
    """
    Initialize a SHAP explainer for the trained scikit-learn pipeline.

    What the pipeline contains. The trained model that ships with the
    project is a scikit-learn pipeline with two named steps: a
    preprocessing step (which standard-scales the numeric columns and
    one-hot-encodes the categorical columns), and a tree-based classifier
    step.

    Why a tree-aware explainer. Tree-aware SHAP explainers are the right
    choice for tree-based models — they are exact, fast, and tree-aware.
    Model-agnostic SHAP explainers exist but are slow; you would only use
    them when the model is not tree-based.

    What you need to do.
        1. Pull the preprocessing step and the classifier step out of the
           pipeline by their named-steps lookup.
        2. Push the background sample through the preprocessing step. SHAP
           needs the post-preprocessing feature matrix that the classifier
           actually sees, not the raw input.
        3. Get the post-preprocessing feature names from the preprocessing
           step. These will look different from the raw column names —
           one-hot encoding turns one categorical column into many
           one-hot columns.
        4. Initialize a tree-aware SHAP explainer on the classifier step
           using the tree-path-dependent perturbation method (which does
           not require a background sample to be passed at construction
           time and is more robust for high-dimensional categorical inputs
           after one-hot encoding).

    Returns:
        Three values: the explainer, the background sample after
        preprocessing, and the list of post-preprocessing feature names.

    Note. The background sample returned here is NOT the test set. To
    compute SHAP values for the test set, push the test set through the
    same preprocessing step yourself in the notebook.
    """
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]

    X_bg_t = _dense(preprocessor.transform(X_background))
    feature_names = [str(f) for f in preprocessor.get_feature_names_out()]

    explainer = shap.TreeExplainer(
        classifier, feature_perturbation="tree_path_dependent")
    return explainer, X_bg_t, feature_names


def detect_proxy_features(
    shap_values: np.ndarray,
    feature_names: Sequence[str],
    protected_feature_substrings: Sequence[str],
    top_k: int = 8,
) -> pd.DataFrame:
    """
    Identify candidate proxy features and return a ranked table.

    What is a proxy feature? A non-protected feature that:
        (1) the model relies on heavily — it has a large average absolute
            SHAP magnitude across the dataset;
        (2) pushes predictions in a particular direction — its average
            signed SHAP is non-trivially positive or negative, not just
            noise that averages to zero;
        (3) is NOT itself a protected attribute.

    For HealthGuard, the planted proxy is insurance type. A real audit will
    not tell you what the proxy is — your function must SURFACE candidates
    so the human auditor can investigate.

    Algorithm to implement.
        - For each feature, compute three numbers:
            * its average absolute SHAP magnitude (raw importance);
            * its average signed SHAP (which direction it pushes
              predictions, on average);
            * a signed-strength score, which is the ratio of the magnitude
              of the average signed SHAP to the average absolute SHAP.
              When this score is high, the feature has a consistent
              directional effect; when low, the feature's effect is
              bidirectional and washes out on average.
        - Drop any feature whose name contains any of the substrings in
          protected_feature_substrings (case-insensitive). After one-hot
          encoding, a single protected attribute appears as many columns,
          one per category. Substring matching is the cleanest way to
          exclude all of them at once.
        - From the remaining features, return the top_k by raw importance,
          along with their signed-strength scores so the auditor can see
          which ones have a directional effect.

    Why filter by name. After one-hot encoding you get column names that
    embed the original column name as a substring. Substring matching
    against the original column name removes every category of the
    protected attribute in one step.

    Returns:
        A table with one row per surfaced candidate, sorted by raw
        importance descending, with columns for the feature name, its raw
        importance, its average signed effect, its signed-strength score,
        and its rank.
    """
    values = _positive_class(shap_values)
    if values.shape[1] != len(feature_names):
        raise ValueError("shap_values columns do not match feature_names")

    mean_abs = np.abs(values).mean(axis=0)
    mean_signed = values.mean(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        signed_strength = np.where(mean_abs > 0, np.abs(mean_signed) / mean_abs, 0.0)

    table = pd.DataFrame({
        "feature": list(feature_names),
        "mean_abs_shap": mean_abs,
        "mean_shap": mean_signed,
        "signed_strength": signed_strength,
    })
    protected = [p.lower() for p in protected_feature_substrings]
    is_protected = table["feature"].str.lower().apply(
        lambda name: any(p in name for p in protected))
    table = (table[~is_protected]
             .sort_values("mean_abs_shap", ascending=False)
             .head(top_k)
             .reset_index(drop=True))
    table["rank"] = np.arange(1, len(table) + 1)
    return table


def explain_false_negatives(
    explainer,
    X_transformed: np.ndarray,
    feature_names: Sequence[str],
    y_true,
    y_pred,
    mask,
    n_examples: int = 3,
) -> List[dict]:
    """
    For up to n_examples false negatives matching the mask, return
    their top contributing features (positive AND negative) with SHAP values.

    Why this matters. A global SHAP summary tells you the population-level
    pattern. A local explanation tells you the per-patient story. For the
    most-harmed subgroup, the local story is what proves the proxy
    pathway: the clinical features push the prediction up, but the proxy
    features push it down hard enough to flip the prediction below the
    threshold. That is not something the global plot can show you on its
    own.

    What you need to do.
        - Identify the rows that are false negatives (true label is one,
          predicted label is zero) AND that lie within the supplied mask
          (the most-harmed-subgroup mask, in the typical use case).
        - Take up to n_examples of them.
        - For each of those rows, compute the per-feature SHAP values and
          rank the features both by most-positive contribution and by
          most-negative contribution. Take the top few of each.

    Returns:
        A list with one entry per selected false negative. Each entry is a
        dict with these keys (the audit notebook reads them by name):
            - "instance_index": int row index of the false negative
            - "shap_values": the per-feature SHAP values for that row
            - "top_positive_features": list of (feature_name, shap_value)
              tuples for the strongest positive contributors
            - "top_negative_features": list of (feature_name, shap_value)
              tuples for the strongest negative contributors
            - "sum_shap_positive" / "sum_shap_negative": the summed positive
              and negative contributions, so the auditor can see how they
              netted out.
    """
    top_n = 5
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    mask = np.asarray(mask, dtype=bool)

    fn_idx = np.flatnonzero((y_true == 1) & (y_pred == 0) & mask)[:n_examples]
    if fn_idx.size == 0:
        return []

    X_sel = _dense(X_transformed[fn_idx])
    values = _positive_class(explainer.shap_values(X_sel))
    names = np.asarray(feature_names)

    examples = []
    for row_idx, sv in zip(fn_idx, values):
        order = np.argsort(sv)
        pos = [(str(names[j]), float(sv[j])) for j in order[::-1][:top_n] if sv[j] > 0]
        neg = [(str(names[j]), float(sv[j])) for j in order[:top_n] if sv[j] < 0]
        examples.append({
            "instance_index": int(row_idx),
            "shap_values": sv,
            "top_positive_features": pos,
            "top_negative_features": neg,
            "sum_shap_positive": float(sv[sv > 0].sum()),
            "sum_shap_negative": float(sv[sv < 0].sum()),
        })
    return examples
