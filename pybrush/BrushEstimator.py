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
from sklearn.metrics import precision_score, recall_score, roc_auc_score

from pybrush import Parameters, Dataset, SearchSpace, brush_rng, individual
from pybrush._brush import set_random_state as set_brush_random_state
from pybrush.EstimatorInterface import EstimatorInterface
from pybrush import RegressorEngine, ClassifierEngine, MultiClassifierEngine

def _ovr_macro(binary_metric):
    """Macro one-vs-rest average of a binary ranking metric, skipping classes
    that are absent (or the only class present) in `y_true`. Matches brush's
    multiclass `roc_auc` and `average_precision_score`."""

    # Auxiliary function to use in final model selection.
    # The goal is to have in python an equivalent implementation of the metrics
    # in c++, so we can do any kind of final model selection after running the
    # evolution.

    def metric(y_true, y_score, sample_weight=None):
        y_true, y_score = np.asarray(y_true), np.asarray(y_score)
        scores = []
        for label in range(y_score.shape[1]):
            y_bin = (y_true == label).astype(float)
            if 0 < y_bin.sum() < len(y_bin):
                scores.append(binary_metric(y_bin, y_score[:, label],
                                            sample_weight=sample_weight))
        return np.mean(scores) if scores else 0.5
    return metric


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
        
        # Dataset construction can optionally shuffle a split, so seed the
        # native generator before constructing any Brush object, not only when
        # Engine::init runs later in this method.
        if isinstance(self.random_state, int):
            set_brush_random_state(self.random_state)

        self._prepare_feature_types(X)

        # Beyong this point, X is not a dataframe anymore
        X, y = check_X_y(self._encode_categories(X), y)
        if self.mode == 'classification':
            # The C++ core indexes probability columns with class labels.  Keep
            # that representation internal while preserving sklearn's original
            # labels at the public API boundary.
            self.classes_, y = np.unique(y, return_inverse=True)
            y = y.astype(np.float32)

        # X has already been encoded above, before sklearn validation.
        self.data_ = self._make_data(X, y, _encoded=True,
                                     feature_names=self.feature_names_,
                                     feature_types=self.feature_types_,
                                     validation_size=self.validation_size,
                                     shuffle_split=self.shuffle_split)
        
        # Get feature_types from the dataset if we didn't have them (numpy array case)
        # but DON'T overwrite feature_names_ because the dataset may return them
        # in alphabetical order due to std::map sorting!
        if not self.feature_types_ or any(not t for t in self.feature_types_):
            self.feature_types_ = self.data_.get_feature_types()

        # These have a default behavior to return something meaningfull if 
        # no values are set
        self.train_ = self.data_.get_training_data()
        self.train_.set_batch_size(self.batch_size) # TODO: update batch indexes at the beggining of every generation
        self.validation_ = self.data_.get_validation_data()

        self.parameters_ = self._wrap_parameters(self.train_.y)

        self.search_space_ = SearchSpace(self.data_,
                                         self.parameters_.functions,
                                         self.parameters_.weights_init)
                
        # Creating a new brush engine
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
    
    def partial_fit(self, X, y, *, 
                    lock_nodes_depth=0, keep_leaves_unlocked=True,
                    keep_current_weights=False, refit=True):
        """
        Continue fitting the current population on X and y.

        With ``max_gens=0``, the population is refitted and reevaluated without
        evolutionary or final model selection; enabled simplifiers run once.
        Use it to re-evaluate the expressions and simplify them based on 
        their predictions.

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
        keep_current_weights : bool, optional
            Whether to keep current weights at the spot they appear, and preventing 
            them to be changed during optimization. Default is False.
        refit : bool, default True
            Fit every existing expression once on the first training batch.
            Set ``batch_size=1.0`` to fit on the whole training partition. The
            validation partition remains separate. Set to False to reevaluate
            existing weights and split thresholds without fitting them again.
        """

        check_is_fitted(self, ["engine_", "classes_" if self.mode == "classification" else "best_estimator_"])
        if isinstance(self.random_state, int):
            set_brush_random_state(self.random_state)

        if isinstance(X, pd.DataFrame):
            assert self.feature_names_ == X.columns.to_list(), \
                "Feature names must be the same as in data from previous fit"

        if self.mode == 'classification':
            labels = np.asarray(y)
            indices = np.searchsorted(self.classes_, labels)
            if np.any(indices >= len(self.classes_)) \
            or np.any(self.classes_[indices] != labels):
                raise ValueError("partial_fit received a class not seen in fit")
            y = indices.astype(np.float32)

        new_data = self._make_data(X, y, 
                                     feature_names=self.feature_names_,
                                     feature_types=self.feature_types_,
                                     validation_size=self.validation_size,
                                     shuffle_split=self.shuffle_split)
        
        # We need to update class weights, this is what is happening here.
        new_parameters = self._wrap_parameters(new_data.get_training_data().y)

        if self.pop_size != len(self.engine_.get_population()):
            raise ValueError("pop_size cannot be changed between fit and partial_fit")
        if self.num_islands != self.engine_.params.num_islands:
            raise ValueError("num_islands cannot be changed between fit and partial_fit")
        
        # The logistic root is not affected by locking or unlocking.
        # It is fixed due to prob_change==0.0.

        # This updates the parameters (such as class weights and the scorer)
        self.parameters_ = new_parameters
        self.engine_.params = new_parameters
        
        # The selected individual may have left the current population. Keep
        # it as a candidate without replacing every member with copies of it.
        self.engine_.set_best_ind(self.best_estimator_)

        self.engine_.lock_nodes(lock_nodes_depth, keep_leaves_unlocked, keep_current_weights)
        if refit:
            self.engine_.invalidate_population_fit()
        self.engine_.fit(new_data)
        # self.engine_.lock_nodes(0, False, False) # unlocking everything

        # getting a new reference to the search space (it is not serialized, so 
        # this ensures that loading a model with pickle and calling either fit()
        # or partial_fit() will restore the search space reference)
        self.search_space_ = self.engine_.search_space

        self.archive_ = self.engine_.get_archive()
        self.population_ = self.engine_.get_population()
        self.best_estimator_ = self.engine_.best_ind
        self.data_ = new_data
        self.train_ = new_data.get_training_data()
        self.train_.set_batch_size(self.batch_size)
        self.validation_ = new_data.get_validation_data()

        if self.max_gens != 0 and self.final_model_selection != "":
            self._update_final_model(self.validation_)

        return self

    def predict(self, X):
        """Predict using the best estimator in the archive. """

        check_is_fitted(self)

        data = self._make_data(X, feature_types=self.feature_types_,
                               feature_names=self.feature_names_)
        
        prediction = np.asarray(self.best_estimator_.program.predict(data))
        if self.mode == 'classification':
            return self.classes_[prediction.astype(int)]
        return prediction

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

        y = np.array(data.y)

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
                "multi_log": log_loss,
                "accuracy": accuracy_score, 
                "balanced_accuracy": balanced_accuracy_score,
                "average_precision_score": average_precision_score,
                "precision": precision_score,
                "recall": recall_score,
                "roc_auc": roc_auc_score,
            }
            loss_f = loss_f_dict[self.parameters_.scorer]

            multiclass = self.mode == 'classification' and self.parameters_.n_classes > 2
            if multiclass and self.parameters_.scorer in ["roc_auc", "average_precision_score"]:
                loss_f = _ovr_macro(loss_f)

            def eval(ind, sample=None):
                if sample is None:
                    sample = np.arange(len(y))

                if self.parameters_.scorer in ["log", "multi_log", "average_precision_score", "roc_auc"]:
                    y_pred = np.array(ind.predict_proba(data))
                else: # accuracy, balanced accuracy, or regression metrics
                    y_pred = np.array(ind.predict(data))

                sample_y = y[sample]
                sample_pred = y_pred[sample]
                metric_kwargs = {}
                if self.parameters_.scorer in ["log", "multi_log"]:
                    metric_kwargs["labels"] = np.arange(self.parameters_.n_classes)
                elif self.parameters_.scorer in ["precision", "recall"]:
                    metric_kwargs["zero_division"] = 0
                    metric_kwargs["average"] = "macro" if multiclass else "binary"

                # Match the native scorer when a binary AUROC sample has only
                # one class: AUROC is undefined, and Brush returns 0.5.
                if self.parameters_.scorer == "roc_auc" and not multiclass \
                        and len(np.unique(sample_y)) < 2:
                    return 0.5

                # y_pred = np.nan_to_num(y_pred) # Protecting the evaluation

                # if user_defined, sample_weight is given by his custom weights. if
                # support, I calculate it here. otherwise, no weight is used
                if self.mode == 'classification' \
                and (self.class_weights not in ['unbalanced'] and self.parameters_.scorer not in ['balanced_accuracy']):
                    if isinstance(self.class_weights, list): # using user-defined values
                        sample_weight = np.array([self.class_weights[int(label)] for label in sample_y])
                    else: # support
                        # Weight only sampled labels: a bootstrap sample may
                        # omit classes that are present in the full dataset.
                        classes, inverse, counts = np.unique(
                            sample_y, return_inverse=True, return_counts=True)
                        sample_weight = (len(sample_y) / (len(classes) * counts))[inverse]
                    return loss_f(sample_y, sample_pred,
                                  sample_weight=sample_weight, **metric_kwargs)
                else: # unbalanced metrics, ignoring weights
                    return loss_f(sample_y, sample_pred, **metric_kwargs)

            seed = self.random_state if isinstance(self.random_state, (int, np.integer)) else 0
            bootstrap_rng = np.random.RandomState(seed)
            val_samples = []
            for i in range(100):
                sample = bootstrap_rng.randint(0, len(y), size=len(y))
                val_samples.append( eval(self.best_estimator_, sample) )

            lower_ci, upper_ci = np.quantile(val_samples,0.05), np.quantile(val_samples,0.95)

            # Recalculate metric with new data
            new_losses = [eval(ind) for ind in self.archive_]

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

    Supports two to five classes. Original class labels are preserved.

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
    
        data = self._make_data(X, feature_types=self.feature_types_,
                               feature_names=self.feature_names_)

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
