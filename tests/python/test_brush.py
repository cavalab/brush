#!/usr/bin/env python3

import brush
import pytest
import numpy as np
import pandas as pd
from pmlb import fetch_data
from sklearn.utils import resample


dfc = pd.read_csv('docs/examples/datasets/d_analcatdata_aids.csv')
Xc = dfc.drop(columns='target')
yc = dfc['target']

dfr = pd.read_csv('docs/examples/datasets/d_enc.csv')
Xr = dfr.drop(columns='label')
yr = dfr['label']
# Xr, yr = fetch_data('192_vineyard', return_X_y = True)

brush_args = dict(
    functions=['SplitBest','Add','Sub','Mul','Div'],
    pop_size=20,
    max_gen=2
)
#     max_gen=100, 
#     pop_size=100, 
#     max_size=50, 
#     max_depth=6,
#     mutation_options = {"point":0.25, "insert": 0.5, "delete":  0.25},
# )

class TestBrush():
    def test_fit_brush_classifier(self):
        est = brush.BrushClassifier(**brush_args)
        est.fit(Xc, yc)
        y_pred = est.predict(Xc)
        y_pred_proba = est.predict_proba(Xc)
        print('score:',est.score(Xc,yc))

    def test_fit_brush_regressor(self):
        est = brush.BrushRegressor(**brush_args)
        
        est.fit(Xr, yr)
        print('score:',est.score(Xr,yr))
        est.best_estimator_.get_model('dot')

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