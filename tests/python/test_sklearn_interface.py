import pytest
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_regression
from pybrush import BrushRegressor
import numpy as np
from sklearn.datasets import make_classification
from pybrush import BrushClassifier
from sklearn.metrics import accuracy_score

def test_brush_regressor_grid_search():
    # Generate synthetic regression data
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
    
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

def test_regression_model_selection():
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
    
    model = BrushRegressor(max_gens=10, pop_size=10, final_model_selection="").fit(X, y)
    idx = np.argmin([p['fitness']['linear_complexity'] for p in model.archive_])
    
    # TODO: write better asserts
    # assert model.best_estimator_.fitness.complexity != model.archive_[idx]['fitness']['linear_complexity']

    model = BrushRegressor(max_gens=10, pop_size=10, final_model_selection="smallest_complexity").fit(X, y)
    idx = np.argmin([p['fitness']['linear_complexity'] for p in model.archive_])

    # TODO: write better asserts
    # Archive is sorted by complexity
    # assert model.best_estimator_.fitness.complexity <= model.archive_[-1]['fitness']['linear_complexity']

    model = BrushRegressor(max_gens=10, pop_size=10, final_model_selection="best_validation_ci").fit(X, y)
    idx = np.argmin([p['fitness']['linear_complexity'] for p in model.archive_])
    # assert False
    # TODO: write better asserts
    # assert model.best_estimator_.fitness.complexity != model.archive_[idx]['fitness']['linear_complexity']

def test_classification_selection():
    X, y = make_classification(
        n_samples=100, n_features=10, n_informative=5, 
        n_redundant=0, random_state=42
    )
    
    # Default selection
    model = BrushClassifier(max_gens=10, pop_size=10, final_model_selection="").fit(X, y)
    idx = np.argmin([p['fitness']['linear_complexity'] for p in model.archive_])

    # Smallest complexity selection
    model = BrushClassifier(max_gens=10, pop_size=10, final_model_selection="smallest_complexity").fit(X, y)
    idx = np.argmin([p['fitness']['linear_complexity'] for p in model.archive_])

    # Best validation CI selection
    model = BrushClassifier(max_gens=10, pop_size=10, final_model_selection="best_validation_ci").fit(X, y)
    idx = np.argmin([p['fitness']['linear_complexity'] for p in model.archive_])


    model = BrushClassifier(max_gens=10, pop_size=10, scorer='average_precision_score', final_model_selection="best_validation_ci").fit(X, y)
    idx = np.argmin([p['fitness']['linear_complexity'] for p in model.archive_])

    # assert False

def test_brush_classifier():

    # Generate synthetic classification data
    X, y = make_classification(n_samples=100, n_features=10, n_classes=2, random_state=42)

    # Define the BrushClassifier
    clf = BrushClassifier(max_gens=10, pop_size=10)
    clf.fit(X, y)

    # Predict on training data
    y_pred = clf.predict(X)

    # Check accuracy is reasonable
    acc = accuracy_score(y, y_pred)
    assert acc > 0.7

def test_brush_simplification_log():
    # Generate synthetic regression data
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
    
    # Define the BrushRegressor
    model = BrushRegressor(inexact_simplification=True, max_gens=20, logfile='./temp.log').fit(X, y)


if __name__ == "__main__":
    pytest.main()