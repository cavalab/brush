#!/usr/bin/env python3

from pybrush import Dataset, SearchSpace
from pybrush import _brush

import numpy as np
import pandas as pd
from pmlb import fetch_data
import json
import pytest
import pickle

@pytest.fixture
def test_data():
    test_y = np.array([1.,0.,1.4,1.,0.,1.,1.,0.,0.,0.])
    test_X = np.array([[1.1,2.0,3.0,4.0,5.0,6.5,7.0,8.0,9.0,10.0],
                       [2.0,1.2,6.0,4.0,5.0,8.0,7.0,5.0,9.0,10.0]]).T
    
    return (test_X, test_y)

def test_create_search_space(test_data):
    test_X, test_y = test_data
    data = Dataset(test_X, test_y)
    SS   = SearchSpace(data)
    
def test_make_program(test_data):
    test_X, test_y = test_data
    data = Dataset(test_X, test_y)
    SS   = SearchSpace(data)
    # pytest.set_trace()
    for d in range(1,4):
        for s in range(1,20):
            prg = SS.make_regressor(d, s)
            print(f"Tree model for depth {d}, size {s}:", prg.get_model())

def test_fit_regressor(test_data):
    test_X, test_y = test_data
    data = Dataset(test_X, test_y)
    SS   = SearchSpace(data)
    # pytest.set_trace()
    for d in range(1,4):
        for s in range(1,20):
            prg = SS.make_regressor(d, s)
            print(f"Tree model for depth {d}, size {s}:", prg.get_model())
            # prg.fit(data)
            y = prg.fit(data).predict(data)
            print(y)

def test_fit_classifier():
    df   = pd.read_csv('docs/examples/datasets/d_analcatdata_aids.csv')
    data = Dataset(df.drop(columns='target'), df['target'])
    SS   = SearchSpace(data)
    # pytest.set_trace()
    for d in range(1,4):
        for s in range(1,20):
            prg = SS.make_classifier(d, s)
            print(f"Tree model for depth {d}, size {s}:", prg.get_model())
            print(f"fitting {prg.get_model()}")
            # prg.fit(data)
            y = prg.fit(data).predict(data)
            print(y)

def test_json_regressor():
    data = _brush.read_csv("docs/examples/datasets/d_2x1_plus_3x2.csv","target")
    json_program = {
        "Tree": [
                { "node_type":"Add", "is_weighted": False },
                { "node_type":"Terminal", "feature":"x1", "is_weighted": True},
                { "node_type":"Terminal", "feature":"x2", "is_weighted": True}
            ],
            "is_fitted_":False
    }
    print( "initial json: {}\n", json_program)
    prg = _brush.program.Regressor(json_program)
    print( "program:", prg.get_model())
    # fit model
    print( "fit")
    prg.fit(data)
    print( "predict")
    y_pred = prg.predict(data)

    learned_weights = prg.get_weights()
    print('learned weights:', learned_weights)
    
    true_weights = [2.0, 3.0]

    assert np.sum(np.abs(data.y-y_pred)) <= 1e-4
    #assert all(round(i,4) == round(j, 4) for i,j in zip(learned_weights, true_weights)) 
    np.allclose(learned_weights, true_weights, atol=1e-4)

def test_serialization():
    data = _brush.read_csv("docs/examples/datasets/d_2x1_plus_3x2.csv","target")
    SS   = _brush.SearchSpace(data)

    for d in range(1,4):
        for s in range(1, 20):
            prg = SS.make_regressor(d, s)
            prg.fit(data)
            print(f"Initial Model:", prg.get_model())
            y_pred = prg.predict(data)
            pgr_pickle = pickle.dumps(prg)

            new_pgr = pickle.loads(pgr_pickle)
            #new_pgr.fit(data)
            print(f"Loaded  Model:", new_pgr.get_model())
            new_y_pred = new_pgr.predict(data)

            assert prg.get_model() == new_pgr.get_model()
            assert np.allclose(new_y_pred, y_pred, atol=1e-3, equal_nan=True)