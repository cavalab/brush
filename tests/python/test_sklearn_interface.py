import pytest
import os
import tempfile

import numpy as np
import pandas as pd

from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import GridSearchCV
from pybrush import BrushClassifier, BrushRegressor
from sklearn.metrics import accuracy_score


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

    def fit():
        BrushRegressor(
            functions=["Add", "Sub", "Mul", "Div", "Pow", "Sin", "Cos"],
            inexact_simplification=True,
            max_gens=5,
            pop_size=10,
            num_islands=2,
            logfile=str(logfile),
            verbosity=0,
        ).fit(X, y)

    # We will train two different instances with same logfile. This is 
    # designed to make partial_fits to also write on the same file (but with a 
    # different run)
    fit()
    fit()  # appending a second run must keep every file parseable

    gens = pd.read_csv(logfile)
    assert list(gens.columns) == [
        "run_id", "random_state", "generation", "time",
        "best_score", "best_score_val", "med_score", "med_score_val",
        "med_size", "med_complexity", "max_size", "max_complexity",
        "best_size", "best_complexity",
        "stall_count", "archive_size", "n_evaluations",
        "best_model",
    ]
    assert gens["run_id"].nunique() == 2
    assert (gens.groupby("run_id").size() == 5).all()
    assert gens["best_model"].notna().all()
    for _, run in gens.groupby("run_id"):
        assert run["n_evaluations"].is_monotonic_increasing
        assert list(run["generation"]) == list(range(5))

    islands = pd.read_csv(str(logfile) + "_islands.csv")
    assert len(islands) == 2 * 5 * 2  # runs * generations * islands
    assert set(islands["island"]) == {0, 1}

    simplifications = pd.read_csv(str(logfile) + "_simplifications.csv")
    assert list(simplifications.columns) == [
        "run_id", "generation", "individual_id", "simplifier", "ret_type",
        "original", "replacement", "distance",
    ]
    assert set(simplifications["simplifier"]) <= {"constants", "inexact"}

    table = pd.read_csv(str(logfile) + "_simplification_table")
    assert list(table.columns) == ["run_id", "DataType", "Plane", "Key", "Tree"]
    assert table["run_id"].nunique() == 2

    runs = pd.read_json(str(logfile) + "_runs.jsonl", lines=True)
    assert len(runs) == 2
    assert set(runs["run_id"]) == set(gens["run_id"])
    assert runs["params"].iloc[0]["logfile"] == str(logfile)



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
