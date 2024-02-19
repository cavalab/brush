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
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import MinMaxScaler
import _brush
from .deap_api import nsga2, nsga2island, DeapIndividual, e_lexicase
# from _brush import Dataset, SearchSpace


class BrushEstimator(BaseEstimator):
    """
    This is the base class for Brush estimators. 
    This class shouldn't be called directly; instead, call a child class like 
    :py:class:`BrushRegressor <brush.estimator.BrushRegressor>` or :py:class:`BrushClassifier <brush.estimator.BrushClassifier>`. 
    All of the shared parameters are documented here. 

    Parameters
    ----------
    mode : str, default 'classification'
        The mode of the estimator. Used by subclasses
    pop_size : int, default 100
        Population size.
    max_gen : int, default 100
        Maximum iterations of the algorithm.
    verbosity : int, default 0
        Controls level of printouts.
    max_depth : int, default 0
        Maximum depth of GP trees in the GP program. Use 0 for no limit.
    max_size : int, default 0
        Maximum number of nodes in a tree. Use 0 for no limit.
    n_islands : int, default 5
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
    mutation_options : dict, default {"point":1/6, "insert":1/6, "delete":1/6, "subtree":1/6, "toggle_weight_on":1/6, "toggle_weight_off":1/6}
        A dictionary with keys naming the types of mutation and floating point 
        values specifying the fraction of total mutations to do with that method.
        The probability of having a mutation is `(1-cx_prob)` and, in case the mutation
        is applied, then each mutation option is sampled based on the probabilities
        defined in `mutation_options`. The set of probabilities should add up to 1.0.
    functions: dict[str,float] or list[str], default {}
        A dictionary with keys naming the function set and values giving the probability
        of sampling them, or a list of functions which will be weighted uniformly.
        If empty, all available functions are included in the search space.
    initialization : {"uniform", "max_size"}, default "uniform" 
        Distribution of sizes on the initial population. If `max_size`, then every
        expression is created with `max_size` nodes. If `uniform`, size will be
        uniformly distributed between 1 and `max_size`.
    selection : {"e-lexicase", "tournament"}, default "e-lexicase"
        A string with the selection method to use. Default is automatic epsilon
        lexicase. Ignored if algorithm is not "nsga2" or "nsga2island".
    objectives : list[str], default ["error", "size"]
        list with one or more objectives to use. Options are `"error", "size", "complexity"`.
        If `"error"` is used, then it will be the mean squared error for regression,
        and accuracy for classification.
    algorithm : {"nsga2island", "nsga2", "gaisland", "ga"}, default "nsga2"
        Which Evolutionary Algorithm framework to use to evolve the population.
        nsga2 will solve a multi-objective problem, while ga will use only the first
        objective as fitness.
    pick_criteria : {"error", "MCDM"}, default "error"
        How to chose the best individual from final population. If `"error"`,
        then the individual with best value for the first objective will be selected.
        MCDM stands for multi-criteria decision making, and tries to get the
        knee of the pareto frontier of the final population.
        The criteria is based on validation partition if `0<validation_size<=1` ---
        otherwise it will use training partition. 
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
        max_gen=100,
        verbosity=0,
        max_depth=3,
        max_size=20,
        n_islands=5,
        mig_prob=0.05,
        cx_prob= 1/7,
        mutation_options = {"point":1/6, "insert":1/6, "delete":1/6, "subtree":1/6,
                            "toggle_weight_on":1/6, "toggle_weight_off":1/6},
        functions: list[str]|dict[str,float] = {},
        initialization="uniform",
        selection="e-lexicase",
        algorithm="nsga2",
        pick_criteria="error",
        objectives=["error", "size"],
        random_state=None,
        weights_init=True,
        validation_size: float = 0.0,
        batch_size: float = 1.0
        ):
        self.pop_size=pop_size
        self.max_gen=max_gen
        self.verbosity=verbosity
        self.algorithm=algorithm
        self.pick_criteria=pick_criteria
        self.mode=mode
        self.max_depth=max_depth
        self.max_size=max_size
        self.n_islands=n_islands
        self.mig_prob=mig_prob
        self.cx_prob=cx_prob
        self.mutation_options=mutation_options
        self.functions=functions
        self.objectives=objectives
        self.initialization=initialization
        self.selection=selection
        self.random_state=random_state
        self.batch_size=batch_size
        self.weights_init=weights_init
        self.validation_size=validation_size

    def _fitness_validation(self, ind, data: _brush.Dataset):
        # Fitness without fitting the expression, used with validation data

        ind_objectives = {
            "error"     : self._error(ind, data),
            "size"      : ind.prg.size(),
            "complexity": ind.prg.complexity()
        }

        return [ ind_objectives[obj] for obj in self.objectives ]


    def _fitness_function(self, ind, data: _brush.Dataset):
        # fit the expression, then evaluate.

        ind.prg.fit(data)

        # Setting individual errors for lexicase
        ind.errors = np.nan_to_num(
            np.abs( data.y - ind.prg.predict(data) ), nan=np.inf)

        return self._fitness_validation(ind, data)


    def _setup_toolbox(self, data_train, data_validation):
        """Setup the deap toolbox"""
        toolbox: base.Toolbox = base.Toolbox()

        # creator.create is used to "create new functions", and takes at least
        # 2 arguments: the name of the newly created class and a base class

        # Cleaning possible previous classes that are model-dependent (clf and reg are differente)
        if hasattr(creator, "FitnessMulti"):
            del creator.FitnessMulti
        if hasattr(creator, "Individual"):
            del creator.Individual

        # Minimizing/maximizing problem: negative/positive weight, respectively.
        # Our classification is using the error as a metric
        # Comparing fitnesses: https://deap.readthedocs.io/en/master/api/base.html#deap.base.Fitness
        creator.create("FitnessMulti", base.Fitness, weights=self.weights)

        # create Individual class, inheriting from self.Individual with a fitness attribute
        creator.create("Individual", DeapIndividual, fitness=creator.FitnessMulti)  

        toolbox.register("Clone", lambda ind: creator.Individual(ind.prg.copy()))
        
        toolbox.register("mate", self._crossover)
        toolbox.register("mutate", self._mutate)

        # When solving multi-objective problems, selection and survival must
        # support this feature. This means that these selection operators must
        # accept a tuple of fitnesses as argument)
        if self.algorithm=="nsga2" or self.algorithm=="nsga2island":
            if self.selection == "e-lexicase":
                toolbox.register("select", e_lexicase) 
            elif self.selection == "tournament":
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
        toolbox.register("getBatch", data_train.get_batch)
        toolbox.register("evaluate", self._fitness_function, data=data_train)
        toolbox.register("evaluateValidation", self._fitness_validation, data=data_validation)

        return toolbox


    def _crossover(self, ind1, ind2):
        offspring = [] 

        for i,j in [(ind1,ind2),(ind2,ind1)]:
            attempts = 0
            child = None
            while (attempts < 3 and child is None):
                child = i.prg.cross(j.prg)

                if child is not None:
                    child = creator.Individual(child)
                attempts = attempts + 1

            offspring.extend([child])

        # so we always need to have two elements to unpack inside `offspring`
        return offspring[0], offspring[1]
    

    def _mutate(self, ind1):
        # offspring = (creator.Individual(ind1.prg.mutate(self.search_space_)),)
        attempts = 0
        offspring = None
        while (attempts < 3 and offspring is None):
            offspring = ind1.prg.mutate()
            
            if offspring is not None:
                return creator.Individual(offspring)
            attempts = attempts + 1
        
        return None


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
        _brush.set_params(self.get_params())
        
        if self.random_state is not None:
            _brush.set_random_state(self.random_state)

        self.feature_names_ = []
        if isinstance(X, pd.DataFrame):
            self.feature_names_ = X.columns.to_list()

        # OBS: validation data is going to be the same as train if no split is set
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

        # Weight of each objective (+ for maximization, - for minimization)
        obj_weight = {
            "error"      : +1.0 if self.mode=="classification" else -1.0,
            "size"       : -1.0,
            "complexity" : -1.0
        }
        self.weights = [obj_weight[w] for w in self.objectives]

        # These have a default behavior to return something meaningfull if 
        # no values are set
        self.train_ = self.data_.get_training_data()
        self.train_.set_batch_size(self.batch_size)
        self.validation_ = self.data_.get_validation_data()

        self.search_space_ = _brush.SearchSpace(self.train_, self.functions_, self.weights_init)
        self.toolbox_ = self._setup_toolbox(data_train=self.train_, data_validation=self.validation_)

        if self.algorithm=="nsga2island" or self.algorithm=="gaisland":
            self.archive_, self.logbook_ = nsga2island(
                self.toolbox_, self.max_gen, self.pop_size, self.n_islands,
                self.mig_prob, self.cx_prob, 
                (0.0<self.batch_size<1.0), self.verbosity, _brush.rnd_flt)
        elif self.algorithm=="nsga2" or self.algorithm=="ga":
            # nsga2 and ga differ in the toolbox
            self.archive_, self.logbook_ = nsga2(
                self.toolbox_, self.max_gen, self.pop_size, self.cx_prob, 
                (0.0<self.batch_size<1.0), self.verbosity, _brush.rnd_flt)

        final_ind_idx = 0

        # Each individual is a point in the Multi-Objective space. We multiply
        # the fitness by the weights so greater numbers are always better
        points = np.array([self.toolbox_.evaluateValidation(ind) for ind in self.archive_])
        points = points*np.array(self.weights)

        # Using the multi-criteria decision making on:
        # - test data if pick_criteria is MCDM
        if self.pick_criteria=="MCDM":
            # Selecting the best estimator using training data
            # (train data==val data if validation_size is set to 0.0)
            # and multi-criteria decision making

            # Normalizing
            min_vals = np.min(points, axis=0)
            max_vals = np.max(points, axis=0)
            points = (points - min_vals) / (max_vals - min_vals)
            points = np.nan_to_num(points, nan=0)
            
            # Reference should be best value each obj. can have (after normalization)
            reference = [1 for _ in range(len(self.weights))]

            # closest to the reference
            final_ind_idx = np.argmin( np.linalg.norm(points - reference, axis=1) )
        else: # Best in obj.1 (loss) in validation data (or training data)
            final_ind_idx = max(
                range(len(points)),
                key=lambda index: (points[index][0], points[index][1]) )

        self.best_estimator_ = self.archive_[final_ind_idx].prg

        if self.verbosity > 0:
            print(f'best model {self.best_estimator_.get_model()}'      +
                  f' with size {self.best_estimator_.size()}, '         +
                  f' depth {self.best_estimator_.depth()}, '            +
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

        if isinstance(y, None):
            return _brush.Dataset(X=X,
                    feature_names=feature_names, validation_size=validation_size)

        return _brush.Dataset(X=X, y=y,
            feature_names=feature_names, validation_size=validation_size)


    def predict(self, X):
        """Predict using the best estimator in the archive. """

        check_is_fitted(self)

        if isinstance(X, pd.DataFrame):
            X = X.values

        assert isinstance(X, np.ndarray)

        data = _brush.Dataset(X=X, ref_dataset=self.data_, 
                              feature_names=self.feature_names_)
        
        # data = self._make_data(X, feature_names=self.feature_names_)

        return self.best_estimator_.predict(data)

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
    

class BrushClassifier(BrushEstimator,ClassifierMixin):
    """Brush for classification.

    For options, see :py:class:`BrushEstimator <brush.estimator.BrushEstimator>`. 

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.read_csv('docs/examples/datasets/d_analcatdata_aids.csv')
    >>> X = df.drop(columns='target')
    >>> y = df['target']
    >>> from brush import BrushClassifier
    >>> est = BrushClassifier()
    >>> est.fit(X,y)
    >>> print('score:', est.score(X,y))
    """
    def __init__( self, **kwargs):
        super().__init__(mode='classification',**kwargs)

    def _error(self, ind, data: _brush.Dataset):
        #return (data.y==ind.prg.predict(data)).sum() / data.y.shape[0]
        return average_precision_score(data.y, ind.prg.predict(data))
    
    def _make_individual(self):
        # C++'s PTC2-based `make_individual` will create a tree of at least
        # the given size. By uniformly sampling the size, we can instantiate a
        # population with more diversity
        
        if self.initialization not in ["uniform", "max_size"]:
            raise ValueError(f"Invalid argument value for `initialization`. "
                             f"expected 'max_size' or 'uniform'. got {self.initialization}")

        return creator.Individual(
            self.search_space_.make_classifier(
                self.max_depth, (0 if self.initialization=='uniform' else self.max_size))
        if self.n_classes_ == 2 else
            self.search_space_.make_multiclass_classifier(
                self.max_depth, (0 if self.initialization=='uniform' else self.max_size))
        )

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

        prob = self.best_estimator_.predict_proba(data)

        if self.n_classes_ <= 2:
            prob = np.hstack( (np.ones(X.shape[0]).reshape(-1,1), prob.reshape(-1,1)) )  
            prob[:, 0] -= prob[:, 1]

        return prob


class BrushRegressor(BrushEstimator, RegressorMixin):
    """Brush for regression.

    For options, see :py:class:`BrushEstimator <brush.estimator.BrushEstimator>`. 

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.read_csv('docs/examples/datasets/d_enc.csv')
    >>> X = df.drop(columns='label')
    >>> y = df['label']
    >>> from brush import BrushRegressor
    >>> est = BrushRegressor()
    >>> est.fit(X,y)
    >>> print('score:', est.score(X,y))
    """
    def __init__(self, **kwargs):
        super().__init__(mode='regressor',**kwargs)

    def _error(self, ind, data: _brush.Dataset):
        MSE = np.mean( (data.y-ind.prg.predict(data))**2 )
        if not np.isfinite(MSE): # numeric erros, np.nan, +-np.inf
            MSE = np.inf

        return MSE

    def _make_individual(self):
        if self.initialization not in ["uniform", "max_size"]:
            raise ValueError(f"Invalid argument value for `initialization`. "
                             f"expected 'max_size' or 'uniform'. got {self.initialization}")
        
        # No arguments (or zero): brush will use PARAMS passed in set_params.
        # max_size is sampled between 1 and params['max_size'] if zero is provided
        return creator.Individual(
            self.search_space_.make_regressor(
                self.max_depth, (0 if self.initialization=='uniform' else self.max_size))
        )

# Under development
# class BrushRepresenter(BrushEstimator, TransformerMixin):
#     """Brush for representation learning.

#     For options, see :py:class:`BrushEstimator <brush.estimator.BrushEstimator>`. 

#     Examples
#     --------
#     >>> import pandas as pd
#     >>> df = pd.read_csv('docs/examples/datasets/d_enc.csv')
#     >>> X = df.drop(columns='label')
#     >>> y = df['label']
#     >>> from brush import BrushRegressor
#     >>> est = BrushRegressor()
#     >>> est.fit(X,y)
#     >>> print('score:', est.score(X,y))
#     """
#     def __init__(self, **kwargs):
#         super().__init__(mode='regressor',**kwargs)

#     def _fitness_function(self, ind, data: _brush.Dataset):
#         ind.prg.fit(data)
#         return (
#             # todo: need to return a matrix from X for this
#             np.sum((data.get_X()- ind.prg.predict(data))**2),
#             ind.prg.size()
#         )

#     def _make_individual(self):
#         return creator.Individual(
#             self.search_space_.make_representer(self.max_depth, self.max_size)
#         )

#     def transform(self, X):
#         """Transform X using the best estimator in the archive. """
#         return self.predict(X)
