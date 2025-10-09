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


def test_brush_simplification_log():
    # Generate synthetic regression data
    X, y = make_regression(n_samples=80, n_features=3, noise=0.1, random_state=42)
    
    # Define the BrushRegressor
    model = BrushRegressor(inexact_simplification=True, max_gens=20, logfile='./temp.log').fit(X, y)


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
    X, y = make_classification(n_samples=80, n_features=8, n_classes=2, random_state=123)

    checkpoint = tmp_path / "brush_checkpoint.json"

    est = BrushClassifier(
        objectives=["scorer", "linear_complexity"],
        scorer="balanced_accuracy",
        max_gens=12,
        pop_size=20,
        max_depth=10,
        max_size=30,
        verbosity=0,
    )

    step = 4
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
    X, y = make_classification(n_samples=60, n_features=6, n_classes=2, random_state=99)

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


if __name__ == "__main__":
    pytest.main()