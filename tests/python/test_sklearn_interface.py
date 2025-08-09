import pytest
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_regression
from pybrush import BrushRegressor
import numpy as np

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

def test_final_model_selection():
    X, y = make_regression(n_samples=100, n_features=10, noise=0.1, random_state=42)
    
    model = BrushRegressor(max_gens=10, pop_size=10, final_model_selection="").fit(X, y)
    idx = np.argmin([p['fitness']['complexity'] for p in model.archive_])
    
    # TODO: write better asserts
    # assert model.best_estimator_.fitness.complexity != model.archive_[idx]['fitness']['complexity']

    model = BrushRegressor(max_gens=10, pop_size=10, final_model_selection="smallest_complexity").fit(X, y)
    idx = np.argmin([p['fitness']['complexity'] for p in model.archive_])

    # TODO: write better asserts
    # Archive is sorted by complexity
    # assert model.best_estimator_.fitness.complexity <= model.archive_[-1]['fitness']['complexity']

if __name__ == "__main__":
    pytest.main()