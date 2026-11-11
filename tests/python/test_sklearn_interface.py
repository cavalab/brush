import pytest
import os
import tempfile

import numpy as np
import pandas as pd

from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import GridSearchCV
from pybrush import BrushClassifier, BrushRegressor
from pybrush import Dataset
from pybrush._brush import set_random_state as set_brush_random_state
from sklearn.metrics import accuracy_score


def test_classification_batches_preserve_prevalence_and_cycle_training_rows():
    """Public Dataset batches retain each class while covering an epoch."""

    # equivalent to cpp test -- just in python now
    set_brush_random_state(42)

    identifiers = np.arange(16, dtype=np.float32)

    X = np.column_stack((identifiers, identifiers % 2, identifiers % 4)).astype(np.float32)
    y = np.where(identifiers < 12, 0.0, 1.0).astype(np.float32)

    data = Dataset(
        X=X,
        y=y,
        feature_names=["id", "flag", "category"],
        feature_types=["ArrayF", "ArrayB", "ArrayI"],
        c=True,
        validation_size=0.25,
        shuffle_split=True,
    )

    training = data.get_training_data()
    training.set_batch_size(0.25)

    # each batch will be a new instance of a brush dataset
    batches = [training.get_batch() for _ in range(5)]

    assert all(batch.get_feature_names() == ["id", "flag", "category"] for batch in batches)
    assert all(batch.get_feature_types() == ["ArrayF", "ArrayB", "ArrayI"] for batch in batches)

    # Every batch reserves one minority-class example. The final two may be
    # repeats because the training partition contains only three such rows.
    # (obs: see explanation for those numbers in test_data.cpp)
    assert [np.count_nonzero(batch.y == 0.0) for batch in batches] == [2, 2, 2, 2, 1]
    assert [batch.get_n_samples() for batch in batches] == [3, 3, 3, 3, 2]

    assert all(np.any(batch.y == 1.0) for batch in batches)

    # The binding exposes labels on sliced datasets. One full epoch must yield
    # All nine majority rows are visited in the five batches. The minority
    # count is larger because its examples are deliberately repeated.

    epoch_labels = np.concatenate([batch.y for batch in batches]).astype(int)
    training_labels = training.y.astype(int)

    assert np.bincount(epoch_labels)[0] == np.bincount(training_labels)[0]


def test_classification_batch_apportions_all_remaining_slots():
    """Large batches distribute floor and remainder slots without overflow."""
    set_brush_random_state(42)
    y = np.array([0.0] * 20 + [1.0] * 20, dtype=np.float32)
    data = Dataset(X=y.reshape(-1, 1), y=y, c=True, batch_size=0.25)

    batch = data.get_batch()

    assert batch.get_n_samples() == 10

    # we have 40 samples and 0.25 batch size --> 10 samples each batch.
    # the batch will have 50/50 (because we set the data as a balanced problem)
    assert np.bincount(batch.y.astype(int), minlength=2).tolist() == [5, 5]


def test_classification_batch_expands_when_smaller_than_class_count():
    set_brush_random_state(42)

    # For classification, if the requested batch has fewer rows than there are
    # classes, Brush enlarges it to include one row per class. 
    # For example, a requested 2-row batch with 4 classes becomes 4 rows.

    y = np.repeat(np.arange(4, dtype=np.float32), 2) # 4 classes
    data = Dataset(X=y.reshape(-1, 1), y=y, c=True, batch_size=0.05)

    batch = data.get_batch()

    assert batch.get_n_samples() == 4
    assert np.bincount(batch.y.astype(int), minlength=4).tolist() == [1, 1, 1, 1]


def test_batched_classifier_completes_multiple_generations():
    """Batch-sized lexicase weights remain aligned with batch case indices."""
    X = np.arange(80, dtype=np.float32).reshape(40, 2)
    y = np.array([0] * 20 + [1] * 20)
    estimator = BrushClassifier(
        pop_size=10,
        max_gens=100,
        max_stall=0,
        max_time=-1,
        max_depth=3,
        batch_size=0.25,
        validation_size=0.2,
        random_state=7,
        n_jobs=1,
    )

    estimator.fit(X, y)

    assert estimator.engine_.is_fitted


def test_regression_batches_stratify_target_quartiles():
    set_brush_random_state(42)
    y = np.arange(16, dtype=np.float32)
    X = y.reshape(-1, 1)
    data = Dataset(X=X, y=y, feature_names=["id"], batch_size=0.25)

    batches = [data.get_batch() for _ in range(4)]
    for batch in batches:
        assert batch.get_n_samples() == 4
        assert np.bincount((batch.y // 4).astype(int), minlength=4).tolist() == [1, 1, 1, 1]

    assert np.array_equal(np.sort(np.concatenate([batch.y for batch in batches])), y)


def test_brush_regressor_grid_search():
    # Generate synthetic regression data
    X, y = make_regression(n_samples=50, n_features=3, noise=0.1, random_state=42)
    
    # Define the BrushRegressor
    model = BrushRegressor()
    
    # Define the parameter grid
    param_grid = {
        'max_gens': [10, 20],
        'pop_size': [10, 20],
        'max_depth': [3, 5]
    }
    
    # Perform grid search
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=2)
    grid_search.fit(X, y)
    
    # Check if the best estimator is found
    assert grid_search.best_estimator_ is not None
    assert grid_search.best_score_ is not None

    # Print the best parameters and score
    print("Best parameters found: ", grid_search.best_params_)
    print("Best score: ", grid_search.best_score_)


def test_brush_classifier():

    # Generate synthetic classification data
    X, y = make_classification(n_samples=80, n_features=10, n_classes=2, random_state=42)

    # Define the BrushClassifier
    clf = BrushClassifier(max_gens=10, pop_size=10)
    clf.fit(X, y)

    # Predict on training data
    y_pred = clf.predict(X)

    # Check accuracy is reasonable
    acc = accuracy_score(y, y_pred)
    assert acc > 0.5


def test_fixed_seed_produces_identical_brush_runs():
    X, y = make_classification(
        n_samples=60, n_features=6, n_informative=4, n_redundant=0,
        random_state=42,
    )
    config = dict(
        max_gens=4,
        pop_size=12,
        num_islands=2,
        n_jobs=2,
        random_state=42,
        verbosity=0,
        constants_simplification=False,
        inexact_simplification=False,
        bandit='dummy',
        shuffle_split=False,
    )

    first = BrushClassifier(**config).fit(X, y)
    second = BrushClassifier(**config).fit(X, y)

    assert first.best_estimator_.get_model() == second.best_estimator_.get_model()
    assert all(np.isclose(first.best_estimator_.fitness.values, second.best_estimator_.fitness.values, atol=1e-3))
    assert [ind.get_model() for ind in first.population_] == [
        ind.get_model() for ind in second.population_
    ]


@pytest.mark.parametrize("verbosity", [0, 1, 2])
def test_brush_classifier_verbosity_levels(verbosity):
    X, y = make_classification(n_samples=60, n_features=8, n_classes=2, random_state=42)

    clf = BrushClassifier(
        max_gens=3,
        pop_size=10,
        verbosity=verbosity,
        random_state=42,
    )
    clf.fit(X, y)

    y_pred = clf.predict(X)

    assert y_pred.shape[0] == X.shape[0]
    assert clf.best_estimator_ is not None


def test_brush_logfile_output(tmp_path):
    # Generate synthetic regression data
    X, y = make_regression(n_samples=80, n_features=3, noise=0.1, random_state=42)

    logfile = tmp_path / "brush_run.log"

    BrushRegressor(
        inexact_simplification=True,
        max_gens=5,
        pop_size=10,
        logfile=str(logfile),
        verbosity=0,
    ).fit(X, y)

    assert logfile.exists()
    assert logfile.stat().st_size > 0

    simplification_log = tmp_path / "brush_run.log_simplification_table"
    assert simplification_log.exists()


def test_brush_classifier_population_reuse(tmp_path):
    # Synthetic dataset for speed
    X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)

    pop_file = tmp_path / "population.json"

    # First run with one metric
    est1 = BrushClassifier(
        functions=['SplitBest','Add','Mul','Sin','Cos','Exp','Logabs'],
        max_gens=5,
        max_size=20,
        objectives=["scorer", "complexity"],
        scorer="log",
        save_population=str(pop_file),
        pop_size=30,
        verbosity=0,
    )
    est1.fit(X, y)
    score1 = est1.score(X, y)
    assert est1.best_estimator_ is not None
    assert score1 is not None

    # Second run with reloaded population and different objective
    est2 = BrushClassifier(
        functions=['SplitBest','Add','Mul','Sin','Cos','Exp','Logabs'],
        load_population=str(pop_file),
        objectives=["scorer", "linear_complexity"],
        scorer="accuracy",
        max_gens=5,
        pop_size=30,  # must match
        verbosity=0,
    )
    est2.fit(X, y)
    score2 = est2.score(X, y)

    assert est2.best_estimator_ is not None
    assert score2 is not None

    # Ensure second run reuses and trains successfully
    assert score2 >= 0.5


def test_brush_classifier_checkpoint_training(tmp_path):
    # Small synthetic dataset
    X, y = make_classification(n_samples=80, n_features=8, n_classes=2, random_state=42)

    checkpoint = tmp_path / "brush_checkpoint.json"

    est = BrushClassifier(
        objectives=["scorer", "linear_complexity"],
        scorer="balanced_accuracy",
        max_gens=10,
        pop_size=20,
        max_depth=4,
        max_size=10,
        verbosity=0,
    )

    step = 10
    max_gens = est.max_gens
    est.max_gens = step
    est.save_population = str(checkpoint)
    est.load_population = ""

    for g in range(max_gens // step):
        est.fit(X, y)
        est.load_population = str(checkpoint)
        score = est.score(X, y)

        assert est.best_estimator_ is not None
        assert score is not None
        assert score >= 0.5

    # Restore state
    est.max_gens = max_gens


def test_brush_lock_nodes_and_leaves():
    # Small synthetic dataset
    X, y = make_classification(n_samples=60, n_features=6, n_classes=2, random_state=42)

    est = BrushClassifier(
        functions=['Add','Mul','Sin','Cos'],
        max_gens=10,
        pop_size=15,
        scorer='accuracy',
        verbosity=0,
    )
    est.fit(X, y)

    # Get model string + fitness before locking
    model_before = est.best_estimator_.get_model()
    fitness_before = est.best_estimator_.fitness

    # Lock nodes and fit again
    est.partial_fit(X, y, lock_nodes_depth=999, keep_leaves_unlocked=False)

    # Get model string + fitness after locking
    model_after = est.best_estimator_.get_model()
    fitness_after = est.best_estimator_.fitness

    # Assert model size is unchanged
    assert fitness_before.size == fitness_after.size

    # Fitness should still be valid (not None)
    assert fitness_after is not None

    assert fitness_after.loss_v >= fitness_before.loss_v


def test_brush_multiclass_probabilities_and_labels():
    X, y = make_classification(
        n_samples=45, n_features=5, n_informative=4, n_redundant=0,
        n_classes=3, n_clusters_per_class=1, random_state=42)
    labels = np.array(["class-a", "class-b", "class-c"])[y]

    est = BrushClassifier(
        max_gens=2, pop_size=8, max_size=30, max_depth=4,
        num_islands=1, validation_size=0.0, random_state=42)
    est.fit(X, labels)

    probabilities = est.predict_proba(X)
    prediction = est.predict(X)
    assert probabilities.shape == (X.shape[0], 3)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
    assert set(prediction).issubset(set(labels))
    assert np.array_equal(est.classes_, np.array(["class-a", "class-b", "class-c"]))


if __name__ == "__main__":
    pytest.main()
