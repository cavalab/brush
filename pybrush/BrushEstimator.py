"""
sklearn-compatible wrapper for GP analyses.

See engine.cpp for Python (via pybind11) modules that give more fine-grained
control of the underlying GP objects.
"""

import numpy as np
import pandas as pd

from sklearn.base import BaseEstimator, ClassifierMixin, \
                         RegressorMixin, TransformerMixin

from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_X_y

from pybrush import Parameters, Dataset, SearchSpace, brush_rng, individual
from pybrush.EstimatorInterface import EstimatorInterface
from pybrush import RegressorEngine, ClassifierEngine, MultiClassifierEngine

from pandas.api.types import is_float_dtype, is_bool_dtype, is_integer_dtype

class BrushEstimator(EstimatorInterface, BaseEstimator):
    """
    This is the base class for Brush estimators using the c++ engine. 
    
    Parameters are defined and documented in 
    :py:class:`EstimatorInterface <pybrush.EstimatorInterface.EstimatorInterface>`

    Attributes
    ----------
    best_estimator_ : pybrush.Program
        The final model picked from training. Used in subsequent calls to :func:`predict`. 
    archive_ : list[deap_api.DeapIndividual]
        The final population from training. 
    data_ : pybrush.Dataset
        The complete data in Brush format. 
    train_ : pybrush.Dataset
        Partition of `data_` containing `(1-validation_size)`% of the data, in Brush format.
    validation_ : pybrush.Dataset
        Partition of `data_` containing `(validation_size)`% of the data, in Brush format.
    search_space_ : a Brush `SearchSpace` object. 
        Holds the operators and terminals and sampling utilities to update programs.

    > NOTE: as for now, when serializing the model with pickle, the objects of type `Dataset` and `SearchSpace` are not serialized.
    """
    
    def __init__(self, **kwargs):
        EstimatorInterface.__init__(self, **kwargs)

    def fit(self, X, y):
        """
        Fit an estimator to X,y.

        Parameters
        ----------
        X : np.ndarray
            2-d array of input data.
        y : np.ndarray
            1-d array of (boolean) target values.
        """
        
        self.feature_names_ = []
        self.feature_types_ = []
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.to_list()
            for values, dtype in zip(X.values.T, X.dtypes):
                if is_bool_dtype(dtype):
                    self.feature_types_.append('ArrayB')
                elif is_integer_dtype(dtype):
                    if np.all(np.logical_or(values == 0, values == 1)):
                        self.feature_types_.append('ArrayB')
                    else:
                        self.feature_types_.append('ArrayI')
                elif is_float_dtype(dtype):
                    self.feature_types_.append('ArrayF')
                else:
                    raise ValueError(
                        "Unsupported data type. Please try using an "
                        "encoding method to convert the data to a supported "
                        "format.")

        X, y = check_X_y(X, y)

        self.data_ = self._make_data(X, y, 
                                     feature_names=self.feature_names_,
                                     feature_types=self.feature_types_,
                                     validation_size=self.validation_size,
                                     shuffle_split=self.shuffle_split)

        # These have a default behavior to return something meaningfull if 
        # no values are set
        self.train_ = self.data_.get_training_data()
        self.train_.set_batch_size(self.batch_size) # TODO: update batch indexes at the beggining of every generation
        self.validation_ = self.data_.get_validation_data()

        self.parameters_ = self._wrap_parameters(self.train_.y)

        self.search_space_ = SearchSpace(self.data_,
                                         self.parameters_.functions,
                                         self.parameters_.weights_init)
                
        self.engine_ = None
        if self.mode == 'classification':
            self.engine_ = ( ClassifierEngine
                             if self.parameters_.n_classes == 2 else
                             MultiClassifierEngine)(self.parameters_, 
                                                    self.search_space_)
        else:
            self.engine_ = RegressorEngine(self.parameters_, self.search_space_)

        self.engine_.fit(self.data_)
        
        self.archive_ = self.engine_.get_archive()
        self.population_ = self.engine_.get_population()
        self.best_estimator_ = self.engine_.best_ind
        
        return self
    
    def partial_fit(self, X, y, lock_nodes_depth=0, keep_leaves_unlocked=True):
        """
        Fit an estimator to X,y, without reseting the estimator.

        Parameters
        ----------
        X : np.ndarray
            2-d array of input data.
        y : np.ndarray
            1-d array of (boolean) target values.
        lock_nodes_depth : int, optional
            The depth of the tree to lock. Default is 0.
        keep_leaves_unlocked : bool, optional
            Whether to skip leaves when locking nodes. Default is True.
        """

        if isinstance(X, pd.DataFrame):
            assert self.feature_names_ == X.columns.to_list(), \
                "Feature names must be the same as in data from previous fit"

        new_data = self._make_data(X, y, 
                                     feature_names=self.feature_names_,
                                     feature_types=self.feature_types_,
                                     validation_size=self.validation_size,
                                     shuffle_split=self.shuffle_split)
        
        # We need to update class weights, this is what is happening here.
        new_parameters = self._wrap_parameters(new_data.get_training_data().y)

        assert self.engine_ is not None, \
            "You must call `fit` before calling `partial_fit`"
        
        # The logistic root is not affected by locking or unlocking.
        # It is fixed due to prob_change==0.0.

        # This updates the parameters (such as class weights)
        self.engine_.params = new_parameters
        
        self.engine_.lock_nodes(lock_nodes_depth, keep_leaves_unlocked)
        self.engine_.fit(new_data)
        self.engine_.lock_nodes(0, False) # unlocking everything

        self.archive_ = self.engine_.get_archive()
        self.population_ = self.engine_.get_population()
        self.best_estimator_ = self.engine_.best_ind

        if self.final_model_selection != "":
            self._update_final_model()

        return self

    def predict(self, X):
        """Predict using the best estimator in the archive. """

        check_is_fitted(self)

        if self.data_ is None:
            self.data_ = self._make_data(X, 
                                    feature_names=self.feature_names_,
                                    feature_types=self.feature_types_,
                                    validation_size=self.validation_size,
                                    shuffle_split=self.shuffle_split)
            
        if isinstance(X, pd.DataFrame):
            X = X.values

        assert isinstance(X, np.ndarray)

        # Need to provide feature names because reference does not store order
        data = Dataset(X=X, ref_dataset=self.data_, 
                              feature_names=self.feature_names_)
        
        return self.best_estimator_.program.predict(data)

    def get_params(self, deep=True):
        out = dict()
        for (key, value) in self.__dict__.items():
            if not key.endswith('_'):
                if deep and hasattr(value, "get_params") and not isinstance(value, type):
                    deep_items = value.get_params().items()
                    out.update((key + "__" + k, val) for k, val in deep_items)
                out[key] = value
        return out
    
    def predict_archive(self, X):
        """Returns a list of dictionary predictions for all models."""

        check_is_fitted(self)

        if self.data_ is None:
            self.data_ = self._make_data(X, 
                                    feature_names=self.feature_names_,
                                    feature_types=self.feature_types_,
                                    validation_size=self.validation_size,
                                    shuffle_split=self.shuffle_split)
            
        if isinstance(X, pd.DataFrame):
            X = X.values

        assert isinstance(X, np.ndarray)

        # Need to provide feature names because reference does not store order
        data = Dataset(X=X, ref_dataset=self.data_, 
                            feature_names=self.feature_names_)

        archive = self.engine_.get_archive()

        preds = []
        for ind in archive:
            tmp = {
                'id' : ind['id'],
                'y_pred' : self.engine_.predict_archive(ind['id'], data)
            }
            preds.append(tmp)

        return preds
    
    def _update_final_model(self):
        # selecting the final individual based on the final_model_selection function
        # if the user specified something other than the default cpp pick method

        candidate = None
        if self.final_model_selection == "smallest_complexity":
            candidates = [p for p in self.archive_ if p['fitness']['size'] > 1 + (4 if self.mode == 'classification' else 0)]
            if len(candidates)==0: # fallback to all elements
                candidates = self.archive_

            idx = np.argmin([p['fitness']['complexity'] for p in candidates])

            candidate = candidates[idx]
        elif callable(self.final_model_selection):
            try:
                candidate = self.final_model_selection(self.population_, self.archive_)
            except Exception as e:
                raise RuntimeError("Failed to use the provided model "
                                   "selection function. Raised the following "
                                   f"error: {e}")
        else:
            raise ValueError("Unknown model selection method. Please try using "
                             "a valid function or one of the default methods.")
        
        # Casting the json from the archive_ or population_ into something callable
        if isinstance(candidate, dict):
            if self.mode == 'classification':
                self.best_estimator_ = (
                    individual.ClassifierIndividual
                    if self.parameters_.n_classes == 2 else
                    individual.MultiClassifierIndividual
                ).from_json(candidate)
            else:
                self.best_estimator_ = individual.RegressorIndividual.from_json(candidate)


class BrushClassifier(BrushEstimator, ClassifierMixin):
    """Brush with c++ engine for classification.

    Parameters are defined and documented in 
    :py:class:`EstimatorInterface <pybrush.EstimatorInterface.EstimatorInterface>`

    This class inherits from :py:class:`BrushEstimator <pybrush.BrushEstimator.BrushEstimator>`.
    A full documentation of the methods and attributes can be found there.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.read_csv('docs/examples/datasets/d_analcatdata_aids.csv')
    >>> X = df.drop(columns='target')
    >>> y = df['target']
    >>> from pybrush import BrushClassifier
    >>> est = BrushClassifier()
    >>> est.fit(X,y)
    >>> # print('score:', est.score(X,y))
    """
    def __init__( self, **kwargs):
        kwargs.pop('mode', None)
        super().__init__(mode='classification',**kwargs)

    def predict_proba(self, X):
        """Predict class probabilities for X.

        Parameters
        ----------
        X : {array-like} of shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples.

        """
        
        check_is_fitted(self)

        if self.data_ is None:
            self.data_ = self._make_data(X, 
                                    feature_names=self.feature_names_,
                                    feature_types=self.feature_types_,
                                    validation_size=self.validation_size,
                                    shuffle_split=self.shuffle_split)
            
        if isinstance(X, pd.DataFrame):
            X = X.values

        assert isinstance(X, np.ndarray)

        # Need to provide feature names because reference does not store order
        data = Dataset(X=X, ref_dataset=self.data_, 
                              feature_names=self.feature_names_)

        prob = self.best_estimator_.program.predict_proba(data)

        if self.parameters_.n_classes == 2:
            prob = np.hstack( (np.ones(X.shape[0]).reshape(-1,1), prob.reshape(-1,1)) )  
            prob[:, 0] -= prob[:, 1]

        return prob
    
        
    def predict_proba_archive(self, X):
        """Returns a list of dictionary predictions for all models."""

        check_is_fitted(self)

        if self.data_ is None:
            self.data_ = self._make_data(X, 
                                    feature_names=self.feature_names_,
                                    feature_types=self.feature_types_,
                                    validation_size=self.validation_size,
                                    shuffle_split=self.shuffle_split)
            
        if isinstance(X, pd.DataFrame):
            X = X.values

        assert isinstance(X, np.ndarray)

        # Need to provide feature names because reference does not store order
        data = Dataset(X=X, ref_dataset=self.data_, 
                            feature_names=self.feature_names_)
        
        archive = self.engine_.get_archive()

        preds = []
        for ind in archive:
            tmp = {
                'id' : ind['id'],
                'y_pred' : self.engine_.predict_proba_archive(ind['id'], data)
            }
            preds.append(tmp)

        return preds


class BrushRegressor(BrushEstimator, RegressorMixin):
    """Brush with c++ engine for regression.

    Parameters are defined and documented in 
    :py:class:`EstimatorInterface <pybrush.EstimatorInterface.EstimatorInterface>`

    This class inherits from :py:class:`BrushEstimator <pybrush.BrushEstimator.BrushEstimator>`.
    A full documentation of the methods and attributes can be found there.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.read_csv('docs/examples/datasets/d_enc.csv')
    >>> X = df.drop(columns='label')
    >>> y = df['label']
    >>> from pybrush import BrushRegressor
    >>> est = BrushRegressor()
    >>> est.fit(X,y)
    >>> # print('score:', est.score(X,y))
    """
    
    def __init__(self, **kwargs):
        kwargs.pop('mode', None)
        super().__init__(mode='regression', **kwargs)