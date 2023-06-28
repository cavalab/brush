import pytest

import _brush
import time
from multiprocessing import Pool

def _change_and_wait(config):
    "Will change the mutation weights to set only the `index` to 1, then wait "
    "`seconts` to retrieve the _brush PARAMS and print weight values"
    index, seconds = config

    # Sample configuration
    params = {
        'verbosity': False, 
        'pop_size' : 100,
        'max_gen'  : 100,
        'max_depth': 5,
        'max_size' : 50,
        'mutation_options': {'point'        : 0.0,
                             'insert'       : 0.0,
                             'delete'       : 0.0,
                             'toggle_weight': 0.0}
    }

    # We need to guarantee order to use the index correctly
    mutations = ['point', 'insert', 'delete', 'toggle_weight']

    for i, m in enumerate(mutations):
        params['mutation_options'][m] = 0 if i != index else 1.0

    print(f"(Thread id {index}{seconds}) Setting mutation {mutations[index]} to 1 and wait {seconds} seconds")

    _brush.set_params(params)
    time.sleep(seconds)
    
    print(f"(Thread id {index}{seconds}) Retrieving PARAMS: {_brush.get_params()['mutation_options']}")

    assert params['mutation_options']==_brush.get_params()['mutation_options'], \
        f"(Thread id {index}{seconds}) BRUSH FAILED TO KEEP SEPARATE INSTANCES OF `PARAMS` BETWEEN MULTIPLE THREADS"
    

def test_global_PARAMS_sharing():
    print("By default, all threads starts with all mutations having weight zero.")
    
    scale = 1 # Scale the time of each thread (for human manual checking) 

    # Checking if brush's PARAMS can be modified inside a pool without colateral effects.
    # Each configuration will start in the same order as they are listed, but they
    # will finish in different times. They are all modifying the brush's PARAMS.
    Pool(processes=3).map(_change_and_wait, [(0, 3*scale),
                                             (1, 1*scale),
                                             (2, 2*scale)])