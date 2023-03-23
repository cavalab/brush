#!/usr/bin/env python3

import pytest
import numpy as np
import pandas as pd
from pmlb import fetch_data
from sklearn.utils import resample

# from _brush import Dataset, SearchSpace 
# from _brush.program import Regressor, Classifier
import brush

# test_y = np.array([1.,0.,1,1.,0.,1.,1.,0.,0.,0.])
# test_X = np.array([[1.1,2.0,3.0,4.0,5.0,6.5,7.0,8.0,9.0,10.0],
#                    [2.0,1.2,6.0,4.0,5.0,8.0,7.0,5.0,9.0,10.0]])
dfc = pd.read_csv('docs/examples/datasets/d_analcatdata_aids.csv')
Xc = dfc.drop(columns='target')
yc = dfc['target']

dfr = pd.read_csv('docs/examples/datasets/d_enc.csv')
Xr = dfr.drop(columns='label')
yr = dfr['label']

brush_args = dict(
    max_gen=100, 
    pop_size=100, 
    max_size=50, 
    max_depth=6,
    mutation_options = {"point":0.25, "insert": 0.5, "delete":  0.25},
)

class TestBrush():
    def test_fit_brush_classifier(self):
        est = brush.BrushClassifier(**brush_args)
        est.fit(Xc, yc)
        print('score:',est.score(Xc,yc))

    def test_fit_brush_regressor(self):
        est = brush.BrushRegressor(**brush_args)
        
        est.fit(Xr, yr)
        print('score:',est.score(Xr,yr))

    # def test_fit_classifier(self):
    #     df = pd.read_csv('docs/examples/datasets/d_analcatdata_aids.csv')
    #     data = Dataset(df.drop(columns='target'), df['target'])
    #     SS = SearchSpace(data)
    #     # pytest.set_trace()
    #     for d in range(1,4):
    #         for s in range(1,20):
    #             prg = SS.make_classifier(d, s)
    #             print(f"Tree model for depth {d}, size {s}:", prg.get_model())
    #             print(f"fitting {prg.get_model()}")
    #             # prg.fit(data)
    #             y = prg.fit(data).predict(data)
    #             print(y)

if __name__ == '__main__':
    # TestBrush().test_fit_brush_classifier()
    TestBrush().test_fit_brush_regressor()