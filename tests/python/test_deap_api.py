#!/usr/bin/env python3
import pybrush
import pytest
import numpy as np
import pandas as pd
from pmlb import fetch_data
from sklearn.utils import resample

import traceback
import logging

# TODO: get deap api back and implement it as deap_nsga2 (or something like that. the idea is that it can be used as a reference. I could even do a documentation prototyping_with_brush.ipynb)
@pytest.fixture
def brush_args():
    return dict(
        gens=10, 
        pop_size=20, 
        max_size=50, 
        max_depth=6,
        cx_prob= 1/7,
        mutation_probs = {"point":1/6, "insert":1/6, "delete":1/6, "subtree":1/6,
                            "toggle_weight_on":1/6, "toggle_weight_off":1/6},
    )
    
@pytest.fixture
def classification_setup():
    df = pd.read_csv('docs/examples/datasets/d_analcatdata_aids.csv')
    X  = df.drop(columns='target')
    y  = df['target']

    return pybrush.DeapClassifier, X, y

@pytest.fixture
def multiclass_classification_setup():
    df = pd.read_csv('docs/examples/datasets/d_analcatdata_aids.csv')
    X  = df.drop(columns='target')
    y  = df['target']

    return pybrush.DeapClassifier, X, y

@pytest.fixture
def regression_setup():
    df = pd.read_csv('docs/examples/datasets/d_enc.csv')
    X  = df.drop(columns='label')
    y  = df['label']

    return pybrush.DeapRegressor, X, y

@pytest.mark.parametrize('setup,algorithm',
                         [('classification_setup', 'nsga2island'),
                          ('classification_setup', 'nsga2'      ),
                          ('classification_setup', 'gaisland'   ),
                          ('classification_setup', 'ga'         ),
                          ('regression_setup',     'nsga2island'),
                          ('regression_setup',     'nsga2'      ),
                          ('regression_setup',     'gaisland'   ),
                          ('regression_setup',     'ga'         )])
def test_fit(setup, algorithm, brush_args, request):
    """Testing common utilities related to fitting and generic brush estimator.
    """
    
    Estimator, X, y = request.getfixturevalue(setup)

    brush_args["algorithm"] = algorithm
    try:
        est = Estimator(**brush_args)
        est.fit(X, y)
        
        print('score:',est.score(X,y))
        
    except Exception as e:
        pytest.fail(f"Unexpected Exception caught: {e}")
        logging.error(traceback.format_exc())

@pytest.mark.parametrize('setup',
                         [('classification_setup'),
                          ('multiclass_classification_setup')])
def test_predict_proba(setup, brush_args, request):

    Estimator, X, y = request.getfixturevalue(setup)

    est = Estimator(**brush_args)
    est.fit(X, y)

    y_prob = est.predict_proba(X)
    assert len(y_prob.shape) == 2, "predict_proba should be 2-dimensional"
    assert y_prob.shape[1] >= 2, \
        "every class should have its own column (even for binary clf)"
            

# @pytest.mark.parametrize('setup',
#                          [('regression_setup')])
# def test_brush_engine(setup, brush_args, request):

#     Estimator, X, y = request.getfixturevalue(setup)

#     dataset = pybrush.Dataset(X=X, y=y)
    
#     # TODO: pybrush parameters could have named arguments
#     params = pybrush.Parameters()
#     params.pop_size    = 10
#     params.gens        = 10
#     params.num_islands = 1

#     eng = pybrush.RegressorEngine(params)
#     # eng.run(dataset)

            

@pytest.mark.parametrize('setup,fixed_node', [
                                                ('classification_setup', 'Logistic'),
                                                # ('multiclass_classification_setup', 'Softmax')
                                             ])
def test_fixed_nodes(setup, fixed_node, brush_args, request):
    # Classification has a fixed root that should not change after mutation or crossover

    Estimator, X, y = request.getfixturevalue(setup)

    est = Estimator(**brush_args)
    est.fit(X, y) # Calling fit to make it create the setup toolbox and variation functions

    for i in range(10):
        # Initial population
        pop = est.toolbox_.population(n=100)
        pop_models = []
        for p in pop:
            pop_models.append(p.program.get_model())
            assert p.program.get_model().startswith(fixed_node), \
                (f"An individual for {setup} was criated without {fixed_node} " +
                 f"node on root. Model was {p.ind.get_model()}")

        # Clones
        clones = [est.toolbox_.Clone(p) for p in pop]
        for c in clones:
            assert c.program.get_model().startswith(fixed_node), \
                (f"An individual for {setup} was cloned without {fixed_node} " +
                 f"node on root. Model was {c.ind.get_model()}")

        # Mutation
        xmen = [est.toolbox_.mutate(c) for c in clones]
        xmen = [x for x in xmen if x is not None]
        assert len(xmen) > 0, "Mutation didn't worked for any individual"
        for x in xmen:
            assert x.program.get_model().startswith(fixed_node), \
                (f"An individual for {setup} was mutated without {fixed_node} " +
                 f"node on root. Model was {x.ind.get_model()}")
        
        # Crossover
        cxmen = []
        [cxmen.append(est.toolbox_.mate(c1, c2))
         for (c1, c2) in zip(clones[::2], clones[1::2])]
        cxmen = [x for x in cxmen if x is not None]
        assert len(cxmen) > 0, "Crossover didn't worked for any individual"
        for cx in cxmen:
            assert cx.program.get_model().startswith(fixed_node), \
                (f"An individual for {setup} was crossovered without {fixed_node} " +
                 f"node on root. Model was {cx.ind.get_model()}")
            
        # Originals still the same
        for p, p_original_model in zip(pop, pop_models):
            assert p.program.get_model() == p_original_model, \
                "Variation operator changed the original model."
        


# def test_random_state(): # TODO: make it work
#     test_y = np.array( [1. , 0. , 1.4, 1. , 0. , 1. , 1. , 0. , 0. , 0.  ])
#     test_X = np.array([[1.1, 2.0, 3.0, 4.0, 5.0, 6.5, 7.0, 8.0, 9.0, 10.0],
#                        [2.0, 1.2, 6.0, 4.0, 5.0, 8.0, 7.0, 5.0, 9.0, 10.0]]).T
    
#     est1 = brush.BrushRegressor(random_state=42).fit(test_X, test_y)
#     est2 = brush.BrushRegressor(random_state=42).fit(test_X, test_y)

#     assert est1.best_estimator_.get_model() == est2.best_estimator_.get_model(), \
#            "random state failed to generate same results"