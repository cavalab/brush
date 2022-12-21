"""
sklearn-compatible wrapper for GP analyses.

See brushgp.cpp for Python (via pybind11) modules that give more fine-grained
control of the underlying GP objects.
"""

import brushgp

class BrushClassifier(object):
    def __init__(self, max_depth=0, max_breadth=0, max_size=0):
        """
        Binary classifier using a GP tree.

        Parameters
        ----------
        max_depth : int, default 0
            Maximum depth of GP trees in the GP program. Use 0 for no limit.
        max_breadth : int, default 0
            Maximum width of the tree at its widest point. Use 0 for no limit.
        max_size : int, default 0
            Maximum number of nodes in a tree. Use 0 for no limit.
        """
        self.max_depth = max_depth
        self.max_breadth = max_breadth
        self.max_size = max_size

    def fit(self, X, y):
        """
        Fit a GP tree to a dataset.

        Parameters
        ----------
        X : np.ndarray
            2-d array of input data.
        y : np.ndarray
            1-d array of (boolean) target values.
        """
        self.data_ = brushgp.Data(X, y)
        self.search_space_ = brushgp.SearchSpace(self.data_)

        self.prg_ = brushgp.Program(
            self.search_space_,
            self.max_depth,
            self.max_breadth,
            self.max_size
        )

    def transform(self, X_test, y_test):
        """
        Transform a new set of data using a trained GP tree.
        """
        pass