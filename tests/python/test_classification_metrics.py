import pytest
import numpy as np

from pybrush import BrushClassifier

from sklearn.datasets import make_classification
from sklearn.metrics import (average_precision_score, precision_score,
                             recall_score, roc_auc_score)


def _ovr_macro(binary_metric, y, proba, sample_weight):
    scores = []
    for label in range(proba.shape[1]):
        y_bin = (y == label).astype(float)
        if 0 < y_bin.sum() < len(y_bin):
            scores.append(binary_metric(y_bin, proba[:, label], sample_weight=sample_weight))
    return np.mean(scores)


def _sklearn_score(scorer, ind, data, n_classes, sample_weight):
    y = np.array(data.y)
    multiclass = n_classes > 2

    if scorer in ["precision", "recall"]:
        metric = precision_score if scorer == "precision" else recall_score
        return metric(y, np.array(ind.predict(data)).astype(float),
                      average="macro" if multiclass else "binary",
                      zero_division=0, sample_weight=sample_weight)

    metric = roc_auc_score if scorer == "roc_auc" else average_precision_score
    proba = np.array(ind.predict_proba(data)).astype(float)
    if multiclass:
        return _ovr_macro(metric, y, proba, sample_weight)
    return metric(y, proba, sample_weight=sample_weight)


@pytest.mark.parametrize("n_classes", [2, 3])
@pytest.mark.parametrize("scorer", ["precision", "recall", "roc_auc", "average_precision_score"])
@pytest.mark.parametrize("class_weights", ["unbalanced", "support"])
def test_metrics_match_sklearn(n_classes, scorer, class_weights):
    X, y = make_classification(n_samples=150, n_features=6, n_informative=4,
                               n_classes=n_classes, weights=None,
                               random_state=42)

    est = BrushClassifier(
        max_gens=3,
        pop_size=30,
        scorer=scorer,
        class_weights=class_weights,
        functions=['Add', 'Sub', 'Mul', 'SplitBest'],
        validation_size=0.3,
        final_model_selection="",
        random_state=42,
        verbosity=0,
    ).fit(X, y)

    data = est.validation_
    y_val = np.array(data.y)

    sample_weight = None
    if class_weights == "support":
        classes, counts = np.unique(y_val, return_counts=True)
        support = {int(c): len(y_val) / (n_classes * n) for c, n in zip(classes, counts)}
        sample_weight = np.array([support[int(label)] for label in y_val])

    brush_scores = [ind.fitness.loss_v for ind in est.archive_]
    sklearn_scores = [_sklearn_score(scorer, ind, data, n_classes, sample_weight)
                      for ind in est.archive_]

    assert np.allclose(brush_scores, sklearn_scores, atol=1e-4), \
        f"brush={brush_scores}, sklearn={sklearn_scores}"


@pytest.mark.parametrize("n_classes", [2, 3])
def test_partial_fit_changes_scorer(n_classes):
    X, y = make_classification(n_samples=150, n_features=6, n_informative=4,
                               n_classes=n_classes, random_state=0)

    # no validation split and no class weights, so the stored fitness is the
    # plain metric on the full data
    est = BrushClassifier(max_gens=3, pop_size=20, scorer="roc_auc",
                          validation_size=0.0, class_weights="unbalanced",
                          random_state=0, verbosity=0).fit(X, y)
    assert est.engine_.params.scorer == "roc_auc"

    est.scorer = "average_precision_score"
    est.partial_fit(X, y, lock_nodes_depth=est.max_depth + 1,
                    keep_leaves_unlocked=False, keep_current_weights=False)

    assert est.engine_.params.scorer == "average_precision_score"
    assert est.parameters_.scorer == "average_precision_score"
    assert est.best_estimator_.fitness.weights[0] == +1.0

    proba = est.predict_proba(X)
    if n_classes == 2:
        expected = average_precision_score(y, proba[:, 1])
    else:
        expected = _ovr_macro(average_precision_score, y, proba, None)

    assert np.isclose(est.best_estimator_.fitness.loss, expected, atol=1e-4)
