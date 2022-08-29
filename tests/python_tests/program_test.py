#!/usr/bin/env python3

import pytest
import numpy as np
# from pmlb import fetch_data

from brushgp import Program, Data, SearchSpace

test_y = np.array([1.,0.,1.4,1.,0.,1.,1.,0.,0.,0.])
test_X = np.array([[1.1,2.0,3.0,4.0,5.0,6.5,7.0,8.0,9.0,10.0],
                   [2.0,1.2,6.0,4.0,5.0,8.0,7.0,5.0,9.0,10.0]])

data = Data(test_X, test_y)
SS = SearchSpace()
SS.init(data, {"ADD": 1, "SUB": 1, "DIV": 0.5, "TIMES": 0.5})


class TestProgram():
    def test_make_program(self):
        data = Data(test_X, test_y)
        SS = SearchSpace(data)
        pytest.set_trace()
        prg = Program(SS, 1, 0, 1)