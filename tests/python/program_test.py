#!/usr/bin/env python3

import pytest
import numpy as np
import pandas as pd
from pmlb import fetch_data

from brushgp import Dataset, SearchSpace, Regressor, Classifier

test_y = np.array([1.,0.,1.4,1.,0.,1.,1.,0.,0.,0.])
test_X = np.array([[1.1,2.0,3.0,4.0,5.0,6.5,7.0,8.0,9.0,10.0],
                   [2.0,1.2,6.0,4.0,5.0,8.0,7.0,5.0,9.0,10.0]])

class TestProgram():
    def test_make_program(self):
        data = Dataset(test_X, test_y)
        SS = SearchSpace(data)
        # pytest.set_trace()
        for d in range(1,4):
            for s in range(1,20):
                prg = SS.make_regressor(d, s)
                print(f"Tree model for depth {d}, size {s}:", prg.get_model())

    def test_fit_regressor(self):
        data = Dataset(test_X, test_y)
        SS = SearchSpace(data)
        # pytest.set_trace()
        for d in range(1,4):
            for s in range(1,20):
                prg = SS.make_regressor(d, s)
                print(f"Tree model for depth {d}, size {s}:", prg.get_model())
                # prg.fit(data)
                y = prg.fit(data).predict(data)
                print(y)

    def test_fit_classifier(self):
        df = pd.read_csv('examples/datasets/d_analcatdata_aids.csv')
        data = Dataset(df.drop(columns='target'), df['target'])
        SS = SearchSpace(data)
        # pytest.set_trace()
        for d in range(1,4):
            for s in range(1,20):
                prg = SS.make_classifier(d, s)
                print(f"Tree model for depth {d}, size {s}:", prg.get_model())
                print(f"fitting {prg.get_model()}")
                # prg.fit(data)
                y = prg.fit(data).predict(data)
                print(y)

if __name__ == '__main__':
    TestProgram().test_fit_program()