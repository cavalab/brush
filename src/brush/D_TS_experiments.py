
from brush import BrushRegressor
import pandas as pd

if __name__ == '__main__':
    
    data = pd.read_csv('docs/examples/datasets/d_example_patients.csv')
    X = data.drop(columns='target')
    y = data['target']

    est = BrushRegressor().fit(X,y)
            