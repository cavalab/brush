#!/usr/bin/env python3

import brush
import pytest
import numpy as np
import pandas as pd
from pmlb import fetch_data
from sklearn.utils import resample

import _brush
import json

import traceback
import logging

@pytest.fixture
def optimize_addition_positive_weights():
    data = _brush.read_csv("docs/examples/datasets/d_2x1_plus_3x2.csv","target")

    json_program = {
        "Tree": [
            { "node_type":"Add", "is_weighted": True },
                { "node_type":"Terminal", "feature":"x1" },
                { "node_type":"Terminal", "feature":"x2" }
        ],
        "is_fitted_":False
    }

    weight_check = lambda learned_weights: np.allclose(learned_weights, [2.0, 3.0], atol=1e-3)

    return (data, json_program, weight_check)

@pytest.fixture
def optimize_addition_negative_weights():
    data = _brush.read_csv("docs/examples/datasets/d_2x1_subtract_3x2.csv","target")

    json_program = {
        "Tree": [
            { "node_type":"Add", "is_weighted": True },
                { "node_type":"Terminal", "feature":"x1" },
                { "node_type":"Terminal", "feature":"x2" }
        ],
        "is_fitted_":False
    }

    weight_check = lambda learned_weights: np.allclose(learned_weights, [2.0, -3.0], atol=1e-3)

    return (data, json_program, weight_check)

@pytest.fixture
def optimize_subtraction_positive_weights():
    data = _brush.read_csv("docs/examples/datasets/d_2x1_subtract_3x2.csv","target")

    json_program = {
        "Tree": [
            { "node_type":"Sub", "is_weighted": True },
                { "node_type":"Terminal", "feature":"x1" },
                { "node_type":"Terminal", "feature":"x2" }
        ],
        "is_fitted_":False
    }

    weight_check = lambda learned_weights: np.allclose(learned_weights, [2.0, 3.0], atol=1e-3)

    return (data, json_program, weight_check)

@pytest.fixture
def optimize_subtraction_negative_weights():
    data = _brush.read_csv("docs/examples/datasets/d_2x1_plus_3x2.csv","target")

    json_program = {
        "Tree": [
            { "node_type":"Sub", "is_weighted": True },
                { "node_type":"Terminal", "feature":"x1" },
                { "node_type":"Terminal", "feature":"x2" }
        ],
        "is_fitted_":False
    }

    weight_check = lambda learned_weights: np.allclose(learned_weights, [2.0, -3.0], atol=1e-3)

    return (data, json_program, weight_check)

@pytest.fixture
def optimize_multiply():
    data = _brush.read_csv("docs/examples/datasets/d_2x1_multiply_3x2.csv","target")

    json_program = {
        "Tree": [
            { "node_type":"Mul", "is_weighted": True },
                { "node_type":"Terminal", "feature":"x1" },
                { "node_type":"Terminal", "feature":"x2" }
        ],
        "is_fitted_":False
    }

    weight_check = lambda learned_weights: np.allclose(np.prod(learned_weights), 2.0*3.0, atol=1e-3)

    return (data, json_program, weight_check)

@pytest.fixture
def optimize_divide():
    data = _brush.read_csv("docs/examples/datasets/d_2x1_divide_3x2.csv","target")

    json_program = {
        "Tree": [
            { "node_type":"Div", "is_weighted": True },
                { "node_type":"Terminal", "feature":"x1" },
                { "node_type":"Terminal", "feature":"x2" }
        ],
        "is_fitted_":False
    }

    weight_check = lambda learned_weights: np.allclose(learned_weights[0]/learned_weights[1], 2.0/3.0, atol=1e-3)

    return (data, json_program, weight_check)

@pytest.fixture
def optimize_sqrt():
    data = _brush.read_csv("docs/examples/datasets/d_2_sqrt_x1.csv","target")

    json_program = {
        "Tree": [
            { "node_type":"Sqrt", "is_weighted": True },
                { "node_type":"Terminal", "feature":"x1" }
        ],
        "is_fitted_":False
    }

    weight_check = lambda learned_weights: np.allclose(learned_weights, [4.0], atol=1e-3)

    return (data, json_program, weight_check)


@pytest.mark.parametrize(
    'optimization_problem', ['optimize_addition_positive_weights',
                             'optimize_addition_negative_weights',
                             'optimize_subtraction_positive_weights',
                             'optimize_subtraction_negative_weights',
                             'optimize_multiply',
                             'optimize_divide',
                             'optimize_sqrt'
                             ])
def test_optimizer(optimization_problem, request):

    data, json_program, weight_check = request.getfixturevalue(optimization_problem)

    print( "initial json: {}\n", json_program)
    prg = _brush.program.Regressor(json_program)
    print( "program:", prg.get_model())

    # fit model
    print( "fit")
    prg.fit(data)
    print( "predict")
    y_pred = prg.predict(data)

    learned_weights = prg.get_weights();
    print('learned weights:', learned_weights)

    assert np.sum(np.abs(data.y-y_pred)) <= 1e-3
    assert np.allclose(data.y, y_pred, atol=1e-3)
    assert weight_check(learned_weights)