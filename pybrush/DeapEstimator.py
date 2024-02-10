"""
sklearn-compatible wrapper for GP analyses.

See brushgp.cpp for Python (via pybind11) modules that give more fine-grained
control of the underlying GP objects.
"""
from sklearn.base import BaseEstimator, ClassifierMixin, RegressorMixin, TransformerMixin
from sklearn.utils.validation  import check_is_fitted
# from sklearn.metrics import mean_squared_error
import numpy as np
import pandas as pd
# import deap as dp
from deap import algorithms, base, creator, tools
# from tqdm import tqdm
from types import NoneType
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import MinMaxScaler
import _brush # TODO: stop using _brush and use whats in pybrush
import functools
from pybrush.deap_api import nsga2
# from _brush import Dataset, SearchSpace
from pybrush import RegressorIndividual, ClassifierIndividual, MultiClassifierIndividual
from pybrush import RegressorEvaluator, ClassifierEvaluator, MultiClassifierEvaluator


# TODO: LOGGER AND ARCHIVE
class DeapEstimator(BaseEstimator):
    """
    This is the base class for Deap-based Brush estimators. 
    This class shouldn't be called directly; instead, call a child class like 
    :py:class:`DeapRegressor <brush.estimator.DeapRegressor>` or :py:class:`DeapClassifier <brush.estimator.DeapClassifier>`. 
    All of the shared parameters are documented here. 

    Parameters
    ----------
    mode : str, default 'classification'
        The mode of the estimator. Used by subclasses
    pop_size : int, default 100
        Population size.
    gens : int, default 100
        Maximum iterations of the algorithm.
    verbosity : int, default 0
        Controls level of printouts.
    max_depth : int, default 0
        Maximum depth of GP trees in the GP program. Use 0 for no limit.
    max_size : int, default 0
        Maximum number of nodes in a tree. Use 0 for no limit.
    num_islands : int, default 5
        Number of independent islands to use in evolutionary framework. 
        Ignored if `algorithm!="nsga2island"`.
    mig_prob : float, default 0.05
        Probability of occuring a migration between two random islands at the
        end of a generation, must be between 0 and 1.
    cx_prob : float, default 1/7
        Probability of applying the crossover variation when generating the offspring,
        must be between 0 and 1.
        Given that there are `n` mutations, and either crossover or mutation is 
        used to generate each individual in the offspring (but not both at the
        same time), we want to have by default an uniform probability between
        crossover and every possible mutation. By setting `cx_prob=1/(n+1)`, and
        `1/n` for each mutation, we can achieve an uniform distribution.
    mutation_probs : dict, default {"point":1/6, "insert":1/6, "delete":1/6, "subtree":1/6, "toggle_weight_on":1/6, "toggle_weight_off":1/6}
        A dictionary with keys naming the types of mutation and floating point 
        values specifying the fraction of total mutations to do with that method.
        The probability of having a mutation is `(1-cx_prob)` and, in case the mutation
        is applied, then each mutation option is sampled based on the probabilities
        defined in `mutation_probs`. The set of probabilities should add up to 1.0.
    functions: dict[str,float] or list[str], default {}
        A dictionary with keys naming the function set and values giving the probability
        of sampling them, or a list of functions which will be weighted uniformly.
        If empty, all available functions are included in the search space.
    initialization : {"uniform", "max_size"}, default "uniform" 
        Distribution of sizes on the initial population. If `max_size`, then every
        expression is created with `max_size` nodes. If `uniform`, size will be
        uniformly distributed between 1 and `max_size`.
    objectives : list[str], default ["error", "size"]
        list with one or more objectives to use. Options are `"error", "size", "complexity"`.
        If `"error"` is used, then it will be the mean squared error for regression,
        and accuracy for classification.
    algorithm : {"nsga2island", "nsga2", "gaisland", "ga"}, default "nsga2"
        Which Evolutionary Algorithm framework to use to evolve the population.
    weights_init : bool, default True
        Whether the search space should initialize the sampling weights of terminal nodes
        based on the correlation with the output y. If `False`, then all terminal nodes
        will have the same probability of 1.0.
    validation_size : float, default 0.0
        Percentage of samples to use as a hold-out partition. These samples are used
        to calculate statistics during evolution, but not used to train the models.
        The `best_estimator_` will be selected using this partition. If zero, then
        the same data used for training is used for validation.
    batch_size : float, default 1.0
        Percentage of training data to sample every generation. If `1.0`, then
        all data is used. Very small values can improve execution time, but 
        also lead to underfit.
    random_state: int or None, default None
        If int, then the value is used to seed the c++ random generator; if None,
        then a seed will be generated using a non-deterministic generator. It is
        important to notice that, even if the random state is fixed, it is
        unlikely that running brush using multiple threads will have the same
        results. This happens because the Operating System's scheduler is
        responsible to choose which thread will run at any given time, thus 
        reproductibility is not guaranteed.

    Attributes
    ----------
    best_estimator_ : _brush.Program
        The final model picked from training. Used in subsequent calls to :func:`predict`. 
    archive_ : list[deap_api.DeapIndividual]
        The final population from training. 
    data_ : _brush.Dataset
        The complete data in Brush format. 
    train_ : _brush.Dataset
        Partition of `data_` containing `(1-validation_size)`% of the data, in Brush format.
    validation_ : _brush.Dataset
        Partition of `data_` containing `(validation_size)`% of the data, in Brush format.
    search_space_ : a Brush `SearchSpace` object. 
        Holds the operators and terminals and sampling utilities to update programs.
    toolbox_ : deap.Toolbox
        The toolbox used by DEAP for EA algorithm. 
    """
    
    def __init__(
        self, 
        mode='classification',
        pop_size=100,
        gens=100,
        verbosity=0,
        max_depth=3,
        max_size=20,
        num_islands=5,
        mig_prob=0.05,
        cx_prob= 1/7,
        mutation_probs = {"point":1/6, "insert":1/6, "delete":1/6, "subtree":1/6,
                            "toggle_weight_on":1/6, "toggle_weight_off":1/6},
        functions: list[str]|dict[str,float] = {},
        initialization="uniform",
        algorithm="nsga2",
        objectives=["error", "size"],
        random_state=None,
        weights_init=True,
        validation_size: float = 0.0,
        batch_size: float = 1.0
        ):

        self.pop_size=pop_size
        self.gens=gens
        self.verbosity=verbosity
        self.algorithm=algorithm
        self.mode=mode
        self.max_depth=max_depth
        self.max_size=max_size
        self.num_islands=num_islands
        self.mig_prob=mig_prob
        self.cx_prob=cx_prob
        self.mutation_probs=mutation_probs
        self.functions=functions
        self.objectives=objectives
        self.initialization=initialization
        self.random_state=random_state
        self.batch_size=batch_size
        self.weights_init=weights_init
        self.validation_size=validation_size


    def _setup_toolbox(self):
        """Setup the deap toolbox"""
        toolbox: base.Toolbox = base.Toolbox()

        # creator.create is used to "create new functions", and takes at least
        # 2 arguments: the name of the newly created class and a base class

        if hasattr(creator, "Individual"):
            del creator.Individual

        # create Individual class, inheriting from self.Individual with a fitness attribute
        if self.mode == 'classification':
            creator.create("Individual", ClassifierIndividual
                                         if self.n_classes_ == 2 else
                                         MultiClassifierIndividual)  
            self.eval_ = ( ClassifierEvaluator()
                     if self.n_classes_ == 2 else
                     MultiClassifierEvaluator() )  
        else:
            creator.create("Individual", RegressorIndividual)  
            self.eval_ = RegressorEvaluator()

        def assign_fit(ind, validation=False):
            ind.program.fit(self.data_.get_training_data())
            self.eval_.assign_fit(ind, self.data_, self.parameters_, validation)
            return ind
        
        toolbox.register("assign_fit", assign_fit)
        
        toolbox.register("Clone", lambda ind: creator.Individual(ind.program.copy()))
        
        toolbox.register("mate", self.variator_.cross)
        toolbox.register("mutate", self.variator_.mutate)

        # When solving multi-objective problems, selection and survival must
        # support this feature. This means that these selection operators must
        # accept a tuple of fitnesses as argument)
        if self.algorithm=="nsga2" or self.algorithm=="nsga2island":
            toolbox.register("select", tools.selTournamentDCD) 
            toolbox.register("survive", tools.selNSGA2)
        elif self.algorithm=="ga" or self.algorithm=="gaisland":
            toolbox.register("select", tools.selTournament, tournsize=3) 
            def offspring(pop, MU): return pop[-MU:]
            toolbox.register("survive", offspring)

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
        
        if self.random_state is not None:
            _brush.set_random_state(self.random_state)

        self.feature_names_ = []
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.to_list()

        self.data_ = self._make_data(X, y, 
                                     feature_names=self.feature_names_,
                                     validation_size=self.validation_size)

        if isinstance(self.functions, list):
            self.functions_ = {k:1.0 for k in self.functions}
        else:
            self.functions_ = self.functions

        # set n classes if relevant
        if self.mode=="classification":
            self.n_classes_ = len(np.unique(y))

            # Including necessary functions for classification programs. This
            # is needed so the search space can create the hash and mapping of
            # the functions.
            if self.n_classes_ == 2 and "Logistic" not in self.functions_:
                self.functions_["Logistic"] = 1.0 
            # elif "Softmax" not in self.functions_: # TODO: implement multiclassific.
            #     self.functions_["Softmax"] = 1.0 

        # These have a default behavior to return something meaningfull if 
        # no values are set
        self.train_ = self.data_.get_training_data()
        self.train_.set_batch_size(self.batch_size) # TODO: update batch indexes at the beggining of every generation
        self.validation_ = self.data_.get_validation_data()

        self.search_space_ = _brush.SearchSpace(self.train_, self.functions_, self.weights_init)
                
        self.parameters_ = _brush.Parameters()
        self.parameters_.pop_size = self.pop_size
        self.parameters_.gens = self.gens
        self.parameters_.num_islands = self.num_islands
        self.parameters_.max_depth = self.max_depth
        self.parameters_.max_size = self.max_size
        self.parameters_.objectives = self.objectives
        self.parameters_.cx_prob = self.cx_prob
        self.parameters_.mig_prob = self.mig_prob
        self.parameters_.functions = self.functions
        self.parameters_.mutation_probs = self.mutation_probs

        if self.mode == "classification":
            self.variator_ = (_brush.ClassifierVariator
                              if self.n_classes_ == 2 else
                              _brush.MultiClassifierVariator
                              )(self.parameters_, self.search_space_)
        elif self.mode == "regressor":
            self.variator_ = _brush.RegressorVariator(self.parameters_, self.search_space_)
        else:
            raise("Unsupported mode")
        
        self.toolbox_ = self._setup_toolbox()

        # nsga2 and ga differ in the toolbox
        self.archive_, self.logbook_ = nsga2(
            self.toolbox_, self.gens, self.pop_size, self.cx_prob, 
            (0.0<self.batch_size<1.0), self.verbosity, _brush.rnd_flt)

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

        if isinstance(y, NoneType):
            return _brush.Dataset(X=X,
                    feature_names=feature_names, validation_size=validation_size)

        return _brush.Dataset(X=X, y=y,
            feature_names=feature_names, validation_size=validation_size)


    def _make_individual(self):
        # C++'s PTC2-based `make_individual` will create a tree of at least
        # the given size. By uniformly sampling the size, we can instantiate a
        # population with more diversity
        
        if self.initialization not in ["uniform", "max_size"]:
            raise ValueError(f"Invalid argument value for `initialization`. "
                             f"expected 'max_size' or 'uniform'. got {self.initialization}")

        # TODO: implement initialization with uniform or max_size
        # No arguments (or zero): brush will use PARAMS passed in set_params.
        # max_size is sampled between 1 and params['max_size'] if zero is provided
        
        ind = creator.Individual()
        ind.init(self.search_space_, self.parameters_)
        ind.objectives = self.objectives
        
        return ind
    
    def predict(self, X):
        """Predict using the best estimator in the archive. """

        check_is_fitted(self)

        if isinstance(X, pd.DataFrame):
            X = X.values

        assert isinstance(X, np.ndarray)

        data = _brush.Dataset(X=X, ref_dataset=self.data_, 
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

    # TODO: test with number of islands =1
       
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

        data = _brush.Dataset(X=X, ref_dataset=self.data_, 
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

#     def _fitness_function(self, ind, data: _brush.Dataset):
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