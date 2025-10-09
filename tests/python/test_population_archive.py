import pytest
import os
import tempfile
import pickle

import numpy as np
import pandas as pd

from sklearn.datasets import make_classification, make_regression
from sklearn.model_selection import GridSearchCV
from pybrush import BrushClassifier, BrushRegressor, Dataset
from sklearn.metrics import accuracy_score


def test_archive_not_empty():
    X, y = make_classification(n_samples=40, n_features=5, n_classes=2, random_state=7)
    est = BrushClassifier(max_gens=5, pop_size=10, verbosity=0)
    est.fit(X, y)

    # Archive should not be empty
    assert len(est.archive_) > 0


def test_archive_individual_get_model_and_predict():
    X, y = make_classification(n_samples=40, n_features=5, n_classes=2, random_state=7)
    est = BrushClassifier(max_gens=5, pop_size=10, verbosity=0)
    est.fit(X, y)

    # Get last archived individual
    ind_from_arch = est.archive_[-1]
    model_str = ind_from_arch.get_model()
    assert isinstance(model_str, str)
    assert len(model_str) > 0

    # Dataset wrapper required for predict
    data = Dataset(X=X, ref_dataset=est.data_, feature_names=est.feature_names_)
    y_pred = ind_from_arch.predict(data)

    assert len(y_pred) == len(y)


def test_population_individual_predict():
    X, y = make_classification(n_samples=50, n_features=6, n_classes=2, random_state=13)
    est = BrushClassifier(max_gens=5, pop_size=12, verbosity=0)
    est.fit(X, y)

    # Pick an individual from the population
    ind = est.population_[0]
    data = Dataset(X=X, ref_dataset=est.data_, feature_names=est.feature_names_)
    y_pred = ind.predict(data)

    assert len(y_pred) == len(y)

    
def test_brush_pickle_and_predict():
    # Small dataset
    X, y = make_classification(n_samples=50, n_features=5, n_classes=2, random_state=42)

    est = BrushClassifier(
        functions=['Add','Mul','Sin','Cos'],
        max_gens=5,
        pop_size=15,
        verbosity=0,
    )
    est.fit(X, y)

    # Serialize estimator
    est_file = os.path.join(tempfile.mkdtemp(), 'est.pkl')
    with open(est_file, 'wb') as f:
        pickle.dump(est, f)

    # Load estimator
    with open(est_file, 'rb') as f:
        est_loaded = pickle.load(f)

    # Predictions with the estimator
    y_pred = est_loaded.predict(X)
    assert len(y_pred) == len(y)

    # Build Dataset wrapper for individuals
    data = Dataset(X=X, ref_dataset=est.data_, 
                   feature_names=est.feature_names_)

    # Predict with an individual from archive
    if est_loaded.archive_:
        y_pred_archive = est_loaded.archive_[-1].predict(data)
        assert len(y_pred_archive) == len(y)

    # Predict with an individual from population
    if est_loaded.population_:
        y_pred_pop = est_loaded.population_[0].predict(data)
        assert len(y_pred_pop) == len(y)