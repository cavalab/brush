import pytest

import _brush
from pybrush import BrushRegressor
import time
from multiprocessing import Pool
import numpy as np


# TODO; get this to work again
# def test_param_random_state():
#     # Check if make_regressor, mutation and crossover will create the same expressions
#     test_y = np.array( [1. , 0. , 1.4, 1. , 0. , 1. , 1. , 0. , 0. , 0.  ])
#     test_X = np.array([[1.1, 2.0, 3.0, 4.0, 5.0, 6.5, 7.0, 8.0, 9.0, 10.0],
#                        [2.0, 1.2, 6.0, 4.0, 5.0, 8.0, 7.0, 5.0, 9.0, 10.0]]).T
    
#     data = _brush.Dataset(test_X, test_y)
#     SS   = _brush.SearchSpace(data)
    
#     _brush.set_random_state(123)

#     first_run = []
#     for d in range(1,4):
#         for s in range(1,20):
#             prg = SS.make_regressor(d, s)
#             prg, _ = prg.mutate()
            
#             if prg != None: prg, _ = prg.cross(prg)    
#             if prg != None: first_run.append(prg.get_model())
    
#     assert len(first_run) > 0, "either mutation or crossover is always failing"

#     _brush.set_random_state(123)

#     second_run = []
#     for d in range(1,4):
#         for s in range(1,20):
#             prg = SS.make_regressor(d, s)
#             prg, _ = prg.mutate()

#             if prg != None: prg, _ = prg.cross(prg)
#             if prg != None: second_run.append(prg.get_model())
        
#     assert len(second_run) > 0, "either mutation or crossover is always failing"

#     for fr, sr in zip(first_run, second_run):
#         assert fr==sr,  "random state failed to generate same expressions"


# def _change_and_wait(config):
#     "Will change the mutation weights to set only the `index` to 1, then wait "
#     "`seconts` to retrieve the _brush PARAMS and print weight values"
#     index, seconds = config

#     # Sample configuration
#     params = {
#         'verbosity': False, 
#         'pop_size' : 100,
#         'gens'  : 100,
#         'max_depth': 5,
#         'max_size' : 50,
#         'mutation_probs': {'point'            : 0.0,
#                              'insert'           : 0.0,
#                              'delete'           : 0.0,
#                              'subtree'          : 0.0,
#                              'toggle_weight_on' : 0.0,
#                              'toggle_weight_off': 0.0}
#     }

#     # We need to guarantee order to use the index correctly
#     mutations = ['point', 'insert', 'delete', 'subtree', 'toggle_weight_on', 'toggle_weight_off']

#     for i, m in enumerate(mutations):
#         params['mutation_probs'][m] = 0 if i != index else 1.0

#     print(f"(Thread id {index}{seconds}) Setting mutation {mutations[index]} to 1 and wait {seconds} seconds")

#     _brush.set_params(params)
#     time.sleep(seconds)
    
#     print(f"(Thread id {index}{seconds}) Retrieving PARAMS: {_brush.get_params()['mutation_probs']}")

#     assert params['mutation_probs']==_brush.get_params()['mutation_probs'], \
#         f"(Thread id {index}{seconds}) BRUSH FAILED TO KEEP SEPARATE INSTANCES OF `PARAMS` BETWEEN MULTIPLE THREADS"
    
# def test_global_PARAMS_sharing():
#     print("By default, all threads starts with all mutations having weight zero.")
    
#     scale = 0.25 # Scale the time of each thread (for human manual checking) 

#     # Checking if brush's PARAMS can be modified inside a pool without colateral effects.
#     # Each configuration will start in the same order as they are listed, but they
#     # will finish in different times. They are all modifying the brush's PARAMS.
#     Pool(processes=3).map(_change_and_wait, [(0, 3*scale),
#                                              (1, 1*scale),
#                                              (2, 2*scale)])
    

def test_max_gens():
    y = np.array( [1. , 0. , 1.4, 1. , 0. , 1. , 1. , 0. , 0. , 0.  ])
    X = np.array([[1.1, 2.0, 3.0, 4.0, 5.0, 6.5, 7.0, 8.0, 9.0, 10.0],
                       [2.0, 1.2, 6.0, 4.0, 5.0, 8.0, 7.0, 5.0, 9.0, 10.0]]).T
    
    for max_gen in [0, 1, 10]:
        print(f"Testing with max_gen={max_gen}")
        reg = BrushRegressor(max_gens=max_gen).fit(X, y)

        predictions = reg.predict(X)
        assert predictions is not None, "Prediction failed"

        print(f"Best individual program: {reg.best_estimator_.program.get_model()}")
        print(f"Best individual fitness: {reg.best_estimator_.fitness}")
    
    # assert False