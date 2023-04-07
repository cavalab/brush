#!/usr/bin/env python3
import brush
import pytest
import numpy as np
import pandas as pd
from pmlb import fetch_data
from sklearn.utils import resample

import traceback
import logging

@pytest.fixture
def brush_args():
    return dict(
        max_gen=10, 
        pop_size=20, 
        max_size=50, 
        max_depth=6,
        mutation_options = {"point":0.25, "insert": 0.5, "delete":  0.25},
    )
    
@pytest.fixture
def classification_setup():
    df = pd.read_csv('docs/examples/datasets/d_analcatdata_aids.csv')
    X  = df.drop(columns='target')
    y  = df['target']

    return brush.BrushClassifier, X, y

@pytest.fixture
def multiclass_classification_setup():
    df = pd.read_csv('docs/examples/datasets/d_analcatdata_aids.csv')
    X  = df.drop(columns='target')
    y  = df['target']

    return brush.BrushClassifier, X, y

@pytest.fixture
def regression_setup():
    df = pd.read_csv('docs/examples/datasets/d_enc.csv')
    X  = df.drop(columns='label')
    y  = df['label']

    return brush.BrushRegressor, X, y

@pytest.mark.parametrize('setup', ['classification_setup', 'regression_setup'])
def test_fit(setup, brush_args, request):
    """Testing common utilities related to fitting and generic brush estimator.
    """
    
    Estimator, X, y = request.getfixturevalue(setup)

    try:
        est = Estimator(**brush_args)
        est.fit(X, y)
        
        print('score:',est.score(X,y))
        
    except Exception as e:
        pytest.fail(f"Unexpected Exception caught: {e}")
        logging.error(traceback.format_exc())
        
