
from _brush import Dataset, SearchSpace # TODO: stop calling cbrush, rename it
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin

# TODO: LOGGER AND ARCHIVE

# TODO: GET DOCUMENTATION BACK
class BrushEstimator(BaseEstimator):
    def __init__(self):
        # self.cbrush_ = CBrush()
        pass

    def fit(self, X, y, Z=None):
        pass

    def predict(self,X,Z=None):
        pass

    def transform(self,X,Z=None):
        pass

    def fit_predict(self,X,y,Z=None):
        pass

    def fit_transform(self,X,y,Z=None):
        pass

    def score(self,X,y,Z=None):
        pass


class BrushRegressor(BrushEstimator):
    def __init__(self,**kwargs):
        pass


class BrushClassifier(BrushEstimator):
    def __init__(self,**kwargs):
        pass

    def predict(self,X,Z=None):
        pass

    def predict_proba(self,X,Z=None):
        pass