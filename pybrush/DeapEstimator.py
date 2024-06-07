"""
sklearn-compatible wrapper for GP analyses.

See brushgp.cpp for Python (via pybind11) modules that give more fine-grained
control of the underlying GP objects.
"""

import functools

import numpy as np
import pandas as pd

from deap import algorithms, base, creator, tools

from sklearn.metrics import average_precision_score
from sklearn.preprocessing import MinMaxScaler

from sklearn.utils.validation  import check_is_fitted
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, \
                         TransformerMixin

from pybrush.EstimatorInterface import EstimatorInterface
from pybrush.deap_api import nsga2
from pybrush import individual
from pybrush import RegressorEvaluator, ClassifierEvaluator, MultiClassifierEvaluator
from pybrush import RegressorSelector, ClassifierSelector, MultiClassifierSelector
from pybrush import RegressorVariator, ClassifierVariator, MultiClassifierVariator
from pybrush import brush_rng, Parameters, Dataset, SearchSpace

class DeapEstimator(EstimatorInterface, BaseEstimator):
    """
    This is the base class for Brush estimators in python. 
    
    Parameters are defined and documented in pybrush.EstimatorInterface.EstimatorInterface

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
    toolbox_ : deap.Toolbox
        The toolbox used by DEAP for EA algorithm. 
    """
    
    def __init__(self, **kwargs):
        EstimatorInterface.__init__(self, **kwargs)

    def _setup_toolbox(self):
        """Setup the deap toolbox"""
        toolbox: base.Toolbox = base.Toolbox()

        # create Individual class, inheriting from self.Individual with a fitness attribute
        if self.mode == 'classification':
            self.Individual = ( individual.ClassifierIndividual
                                 if self.n_classes_ == 2 else
                                 individual.MultiClassifierIndividual)  
            self.eval_ = ( ClassifierEvaluator()
                     if self.n_classes_ == 2 else
                     MultiClassifierEvaluator() )  
            self.sel_  = ( ClassifierSelector("nsga2", False)
                     if self.n_classes_ == 2 else
                     MultiClassifierSelector("nsga2", False) )  
            self.surv_ = ( ClassifierSelector("nsga2", True)
                     if self.n_classes_ == 2 else
                     MultiClassifierSelector("nsga2", True) )  
        else:
            self.Individual = individual.RegressorIndividual  
            self.sel_  = RegressorSelector("lexicase", False)
            self.surv_ = RegressorSelector("nsga2", True)
            self.eval_ = RegressorEvaluator()

        toolbox.register("select",  lambda pop: self.sel_.select(pop, self.parameters_)) 
        toolbox.register("survive", lambda pop: self.surv_.survive(pop, self.parameters_))

        # it could be both sel or surv. 
        toolbox.register("migrate", lambda pop: self.surv_.migrate(pop, self.parameters_)) 

        def update_current_gen(gen): self.parameters_.current_gen = gen
        toolbox.register("update_current_gen", update_current_gen) 

        def assign_fit(ind, validation=False):
            ind.program.fit(self.data_.get_training_data())
            self.eval_.assign_fit(ind, self.data_, self.parameters_, validation)
            return ind
        
        toolbox.register("assign_fit", assign_fit)
        
        toolbox.register("Clone", lambda ind: self.Individual(ind.program.copy()))
        
        toolbox.register("mate", self.variator_.cross)
        toolbox.register("mutate", self.variator_.mutate)
        toolbox.register("vary_pop", lambda pop: self.variator_.vary_pop(pop, self.parameters_))

        # When solving multi-objective problems, selection and survival must
        # support this feature. This means that these selection operators must
        # accept a tuple of fitnesses as argument)
        # if self.algorithm=="nsga2" or self.algorithm=="nsga2island":
        #     toolbox.register("select", tools.selTournamentDCD) 
        #     toolbox.register("survive", tools.selNSGA2)
        # elif self.algorithm=="ga" or self.algorithm=="gaisland":
        #     toolbox.register("select", tools.selTournament, tournsize=3) 
        #     def offspring(pop, MU): return pop[-MU:]
        #     toolbox.register("survive", offspring)


        # toolbox.population will return a list of elements by calling toolbox.individual
        toolbox.register("createRandom", self._make_individual)
        toolbox.register("population", tools.initRepeat, list, toolbox.createRandom)

        toolbox.register("get_objectives", lambda: self.objectives)

        return toolbox

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
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.to_list()

        self.data_ = self._make_data(X, y, 
                                     feature_names=self.feature_names_,
                                     validation_size=self.validation_size)

        # set n classes if relevant
        self.n_classes_ = 0
        if self.mode=="classification":
            self.n_classes_ = len(np.unique(y))

        # These have a default behavior to return something meaningfull if 
        # no values are set
        self.train_ = self.data_.get_training_data()
        self.train_.set_batch_size(self.batch_size)
        
        self.validation_ = self.data_.get_validation_data()

        self.parameters_ = self._wrap_parameters()
        self.search_space_ = SearchSpace(self.data_, self.parameters_.functions, self.weights_init)

        if self.mode == "classification":
            self.variator_ = (ClassifierVariator
                              if self.n_classes_ == 2 else
                              MultiClassifierVariator
                              )(self.parameters_, self.search_space_)
        elif self.mode == "regressor":
            self.variator_ = RegressorVariator(self.parameters_, self.search_space_)
            
            # from pybrush import RegressorEngine
            # brush_estimator = RegressorEngine(self.parameters_)
            # brush_estimator.run(self.data_)
            # print(brush_estimator.is_fitted)
            # print(brush_estimator.best_ind)
        else:
            raise("Unsupported mode")
        
        self.toolbox_ = self._setup_toolbox()

        # nsga2 and ga differ in the toolbox
        self.archive_, self.logbook_ = nsga2(
            self.toolbox_, self.max_gens, self.pop_size, self.cx_prob, 
            (0.0<self.batch_size<1.0), self.verbosity, brush_rng)

        final_ind_idx = 0

        # Each individual is a point in the Multi-Objective space. We multiply
        # the fitness by the weights so greater numbers are always better
        points = np.array([self.toolbox_.assign_fit(ind, True).fitness.wvalues
                           for ind in self.archive_])

        if self.validation_size==0.0:  # Using the multi-criteria decision making on training data
            # Selecting the best estimator using training data
            # (train data==val data if validation_size is set to 0.0)
            # and multi-criteria decision making

            # Normalizing
            points = MinMaxScaler().fit_transform(points)
            
            # Reference should be best value each obj. can have (after normalization)
            reference = np.array([1.0, 1.0])

            # closest to the reference (smallest distance)
            final_ind_idx = np.argmin( np.linalg.norm(points - reference, axis=1) )
        else: # Best in obj.1 (loss) in validation data
            final_ind_idx = max(
                range(len(points)),
                key=lambda index: (points[index][0], points[index][1]) )

        self.best_estimator_ = self.archive_[final_ind_idx]

        if self.verbosity > 0:
            print(f'best model {self.best_estimator_.program.get_model()}'      +
                  f' with size {self.best_estimator_.program.size()}, '         +
                  f' depth {self.best_estimator_.program.depth()}, '            +
                  f' and fitness {self.archive_[final_ind_idx].fitness}')

        return self
    
    def _make_data(self, X, y=None, feature_names=[], validation_size=0.0):
        # This function should not partition data (since it may be used in `predict`).
        # partitioning is done by `fit`. Feature names should be inferred
        # before calling _make_data (so predict can be made with np arrays or
        # pd dataframes).

        if isinstance(y, pd.Series):
            y = y.values
        if isinstance(X, pd.DataFrame):
            X = X.values
        
        assert isinstance(X, np.ndarray)

        if y is None:
            return Dataset(X=X,
                    feature_names=feature_names, validation_size=validation_size)

        return Dataset(X=X, y=y,
            feature_names=feature_names, validation_size=validation_size)


    def _make_individual(self):
        # C++'s PTC2-based `make_individual` will create a tree of at least
        # the given size. By uniformly sampling the size, we can instantiate a
        # population with more diversity
        
        if self.initialization not in ["uniform", "max_size"]:
            raise ValueError(f"Invalid argument value for `initialization`. "
                             f"expected 'max_size' or 'uniform'. got {self.initialization}")

        ind = self.Individual()
        ind.init(self.search_space_, self.parameters_)
        ind.objectives = self.objectives
        
        return ind
    
    def predict(self, X):
        """Predict using the best estimator in the archive. """

        check_is_fitted(self)

        if isinstance(X, pd.DataFrame):
            X = X.values

        assert isinstance(X, np.ndarray)

        data = Dataset(X=X, ref_dataset=self.data_, 
                              feature_names=self.feature_names_)
        
        # data = self._make_data(X, feature_names=self.feature_names_)

        return self.best_estimator_.program.predict(data)

    # def _setup_population(self):
    #     """initialize programs"""
    #     if self.mode == 'classification':
    #         generate = self.search_space_.make_classifier
    #     else:
    #         generate = self.search_space_.make_regressor

    #     programs = [
    #         DeapIndividual(generate(self.max_depth, self.max_size))
    #         for i in range(self.pop_size)
    #     ]
    #     # return [self._create_deap_individual_(p) for p in programs]
    #     return programs

    def get_params(self, deep=True):
        out = dict()
        for (key, value) in self.__dict__.items():
            if not key.endswith('_'):
                if deep and hasattr(value, "get_params") and not isinstance(value, type):
                    deep_items = value.get_params().items()
                    out.update((key + "__" + k, val) for k, val in deep_items)
                out[key] = value
        return out
    

class DeapClassifier(DeapEstimator,ClassifierMixin):
    """Deap-based Brush for classification.

    For options, see :py:class:`DeapEstimator <brush.estimator.DeapEstimator>`. 

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.read_csv('docs/examples/datasets/d_analcatdata_aids.csv')
    >>> X = df.drop(columns='target')
    >>> y = df['target']
    >>> from pybrush import DeapClassifier
    >>> est = DeapClassifier()
    >>> est.fit(X,y)
    >>> # print('score:', est.score(X,y))
    """
    def __init__( self, **kwargs):
        super().__init__(mode='classification',**kwargs)

    def predict_proba(self, X):
        """Predict class probabilities for X.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            The input samples. Internally, it will be converted to
            ``dtype=np.float32``.

        Returns
        -------
        p : ndarray of shape (n_samples, n_classes)
            The class probabilities of the input samples. The order of the
            classes corresponds to that in the attribute :term:`classes_`.

        """
        
        check_is_fitted(self)

        if isinstance(X, pd.DataFrame):
            X = X.values

        assert isinstance(X, np.ndarray)

        data = Dataset(X=X, ref_dataset=self.data_, 
                              feature_names=self.feature_names_)

        # data = self._make_data(X, feature_names=self.feature_names_)

        prob = self.best_estimator_.program.predict_proba(data)

        if self.n_classes_ <= 2:
            prob = np.hstack( (np.ones(X.shape[0]).reshape(-1,1), prob.reshape(-1,1)) )  
            prob[:, 0] -= prob[:, 1]

        return prob


class DeapRegressor(DeapEstimator, RegressorMixin):
    """Deap-based Brush for regression.

    For options, see :py:class:`DeapEstimator <brush.estimator.DeapEstimator>`. 

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.read_csv('docs/examples/datasets/d_enc.csv')
    >>> X = df.drop(columns='label')
    >>> y = df['label']
    >>> from pybrush import DeapRegressor
    >>> est = DeapRegressor()
    >>> est.fit(X,y)
    >>> # print('score:', est.score(X,y))
    """
    def __init__(self, **kwargs):
        super().__init__(mode='regressor',**kwargs)

# Under development
# class DeapRepresenter(DeapEstimator, TransformerMixin):
#     """Deap-based  Brush for representation learning.

#     For options, see :py:class:`DeapEstimator <brush.estimator.DeapEstimator>`. 

#     Examples
#     --------
#     >>> import pandas as pd
#     >>> df = pd.read_csv('docs/examples/datasets/d_enc.csv')
#     >>> X = df.drop(columns='label')
#     >>> y = df['label']
#     >>> from pybrush import DeapRegressor
#     >>> est = DeapRegressor()
#     >>> est.fit(X,y)
#     >>> # print('score:', est.score(X,y))
#     """
#     def __init__(self, **kwargs):
#         super().__init__(mode='regressor',**kwargs)

#     def _fitness_function(self, ind, data: Dataset):
#         ind.program.fit(data)
#         return (
#             # todo: need to return a matrix from X for this
#             np.sum((data.get_X()- ind.program.predict(data))**2),
#             ind.program.size()
#         )

#     def _make_individual(self):
#         return creator.Individual(
#             self.search_space_.make_representer(self.max_depth, self.max_size)
#         )

#     def transform(self, X):
#         """Transform X using the best estimator in the archive. """
#         return self.predict(X)