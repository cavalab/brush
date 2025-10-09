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

from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from sklearn.metrics import average_precision_score, mean_squared_error

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
                    # For Brush, it does matter if it is realy an integer or a boolean in disguise
                    if np.all(np.logical_or(values == 0, values == 1)):
                        self.feature_types_.append('ArrayB')
                    else:
                        if len(np.unique(values))<=10: # Categories, encoded
                            self.feature_types_.append('ArrayI')
                        else: # Integers, we'll treat as floats so we can do all the math normally
                            self.feature_types_.append('ArrayF')
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
        
        # retrieving archive and population as a list of individuals.
        self.archive_ = self.engine_.get_archive()
        self.population_ = self.engine_.get_population()

        # Serialized version of the above
        # self.archive_ = self.engine_.get_archive_as_json()
        # self.population_ = self.engine_.get_population_as_json()

        self.best_estimator_ = self.engine_.best_ind

        if self.final_model_selection != "":
            self._update_final_model(self.data_.get_validation_data())

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

        # Using the same engine as before --- it will keep the population
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
            self._update_final_model(new_data.get_validation_data())

        return self

    def predict(self, X):
        """Predict using the best estimator in the archive. """

        check_is_fitted(self)

        if isinstance(X, pd.DataFrame):
            X = X.values

        assert isinstance(X, np.ndarray)

        # Need to provide feature names because reference does not store order.
        # Some of the self.<attr> are created just after fitting the estimator,
        # and they are properly serialized in the python-side.
        data = Dataset(X=X, # ref_dataset=self.data_, 
                            feature_types=self.feature_types_,
                            feature_names=self.feature_names_,
                            validation_size=0.0,
                            )
        
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
    
    def _update_final_model(self, data=None):
        # selecting the final individual based on the final_model_selection function
        # if the user specified something other than the default cpp pick method

        # TODO : figure out the individual class here and save it to avoid repetition

        if data is None:
            data = self.validation_ #.get_validation_data()

        candidate = None
        if self.final_model_selection == "smallest_complexity":
            candidates = [p for p in self.archive_ if p.fitness.size > 1 + (4 if self.mode == 'classification' else 0)]
            if len(candidates)==0: # fallback to all elements
                candidates = self.archive_

            idx = np.argmin([p.fitness.complexity for p in candidates])

            candidate = candidates[idx]
        elif self.final_model_selection == "best_validation_ci":
            loss_f_dict = { # using sklearn metric, equivalent to what is used internally in brush
                "mse": mean_squared_error, 
                "log": log_loss, 
                "accuracy": accuracy_score, 
                "balanced_accuracy": balanced_accuracy_score,
                "average_precision_score": average_precision_score
            }
            loss_f = loss_f_dict[self.parameters_.scorer]

            def eval(ind, data, sample=None):
                if sample is None:
                    sample = np.arange(len(data.y))

                if self.parameters_.scorer in ["log", "average_precision_score"]:
                    y_pred = np.array(ind.predict_proba(data))
                else: # accuracy, balanced accuracy, or regression metrics
                    y_pred = np.array(ind.predict(data))

                y_pred = np.nan_to_num(y_pred) # Protecting the evaluation

                # if user_defined, sample_weight is given by his custom weights. if
                # support, I calculate it here. otherwise, no weight is used
                if self.class_weights not in ['unbalanced', 'balanced_accuracy']:
                    sample_weight = []
                    if isinstance(self.class_weights, list): # using user-defined values
                        sample_weight = [self.class_weights[int(label)] for label in data.y]
                    else: # support
                        # Calculate class weights by support
                        classes, counts = np.unique(data.y, return_counts=True)
        
                        support_weights = {
                            int(cls): len(data.y) / (len(classes)*count) 
                            if count > 0 else 0.0 for cls, count in zip(classes, counts)}
                        
                        sample_weight = [support_weights[int(label)] for label in data.y]
                    sample_weight = np.array(sample_weight)
                    return loss_f(y[sample], y_pred[sample], sample_weight=sample_weight[sample])
                else: # unbalanced metrics, ignoring weights
                    return loss_f(y[sample], y_pred[sample])

            y = np.array(data.y)
            np.random.seed(0)
            val_samples = []
            for i in range(100):
                sample = np.random.randint(0, len(y), size=len(y))
                val_samples.append( eval(self.best_estimator_, data, sample) )

            lower_ci, upper_ci = np.quantile(val_samples,0.05), np.quantile(val_samples,0.95)

            # Recalculate metric with new data
            new_losses = [eval(ind, data) for ind in self.archive_]

            # Filter for overlapping points. Adding the best estimator to assert there is at least one sample
            candidates = [(l, p) for l, p in zip(new_losses, self.archive_) if lower_ci <= l <= upper_ci]
            
            # There is a chance no candidate exists, since the best individual
            # may not be from the last generation, and brush internally uses the training
            # partition in the evolutionary loop. The individual is picked with 
            # validation loss.

            if len(candidates) > 0:
                # Select the row with the smallest complexity among overlapping rows
                idx = np.argmin([p.fitness.complexity for _, p in candidates])
                
                candidate = candidates[idx][1]
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
        
        if candidate is not None:
            self.best_estimator_ = candidate


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
    
        if isinstance(X, pd.DataFrame):
            X = X.values

        assert isinstance(X, np.ndarray)

        # Need to provide feature names because reference does not store order
        data = Dataset(X=X, # ref_dataset=self.data_, 
                            feature_types=self.feature_types_,
                            feature_names=self.feature_names_,
                            validation_size=0.0)


        prob = self.best_estimator_.program.predict_proba(data)

        if self.parameters_.n_classes == 2:
            prob = np.hstack( (np.ones(X.shape[0]).reshape(-1,1), prob.reshape(-1,1)) )  
            prob[:, 0] -= prob[:, 1]

        return prob
    
    
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