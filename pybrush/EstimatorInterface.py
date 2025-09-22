"""
Estimator interface for GP implementations.

This interface defines all the hyperparameters for Brush estimators and
provides documentation for the hyperparameters.
"""

import numpy as np
import pandas as pd
from pybrush import Parameters, Dataset
from typing import Union, List, Dict
from collections.abc import Callable

class EstimatorInterface():
    """
    Interface class for all estimators in pybrush.

    Parameters
    ----------
    mode : str, default 'classification'
        The mode of the estimator. Used by subclasses
    pop_size : int, default 100
        Population size.
    max_gens : int, default 100
        Maximum iterations of the algorithm.
    max_time: int, optional (default: -1)
        Maximum time terminational criterion in seconds. If -1, not used.
    max_stall: int, optional (default: 0)
        How many generations to continue after the validation loss has
        stalled. If 0, not used.
    verbosity : int, default 0
        Controls level of printouts. Set 0 to disable all printouts,
        1 for basic information, and 2 or more for detailed information.
    max_depth : int, default 10
        Maximum depth of GP trees in the GP program. Use 0 for no limit.
    max_size : int, default 100
        Maximum number of nodes in a tree. Use 0 for no limit.
    num_islands : int, default 5
        Number of independent islands to use in evolutionary framework. 
        This also corresponds to the number of parallel threads in the c++
        engine.
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
        A non-positive value will disable the mutation, even if the multi armed
        bandit strategy is turned on (the mutation will be hidden from the bandit 
        at initialization).
    functions: dict[str,float] or list[str], default {}
        A dictionary with keys naming the function set and values giving the probability
        of sampling them, or a list of functions which will be weighted uniformly.
        If empty, all available functions are included in the search space.
    initialization : {"uniform", "max_size"}, default "uniform" 
        Distribution of sizes on the initial population. If `max_size`, then every
        expression is created with `max_size` nodes. If `uniform`, size will be
        uniformly distributed between 1 and `max_size`.
    objectives : list[str], default ["scorer", "linear_complexity"]
        list with one or more objectives to use. The first objective is the main.
        If `"scorer"` is used, then the metric in `scorer` will be used as objective.
        Possible values are "scorer", "size", "complexity", "linear_complexity",
        and "depth". The first objective will be used as criteria for Pareto
        sorting when creating the archive, and the recursive complexity will be
        used as secondary objective.
    scorer : str, default None
        The metric to use for the "scorer" objective. If None, it will be set to
        "mse" for regression and "log" for binary classification.
        Available options are `["mse", "log", "accuracy", "balanced_accuracy", "average_precision_score"]`
    algorithm : {"nsga2island", "nsga2", "gaisland", "ga"}, default "nsga2"
        Which Evolutionary Algorithm framework to use to evolve the population.
        This is used only in DeapEstimators.
    weights_init : bool, default True
        Whether the search space should initialize the sampling weights of terminal nodes
        based on the correlation with the output y. If `False`, then all terminal nodes
        will have the same probability of 1.0. This parameter is ignored if the bandit
        strategy is used, and weights will be learned dynamically during the run.
    validation_size : float, default 0.2
        Percentage of samples to use as a hold-out partition. These samples are used
        to calculate statistics during evolution, but not used to train the models.
        The `best_estimator_` will be selected using this partition. If zero, then
        the same data used for training is used for validation.
    val_from_arch: boolean, optional (default: True)
        Validates the final model using the archive rather than the whole 
        population.
    constants_simplification: boolean, optional (default: True)
        Whether if the program should check for constant sub-trees and replace
        them with a single terminal with constant value or not.
    inexact_simplification: boolean, optional (default: True)
        Whether if the program should use the inexact simplification proposed in:

        Guilherme Seidyo Imai Aldeia, Fabrício Olivetti de França, and William
        G. La Cava. 2024. Inexact Simplification of Symbolic Regression Expressions
        with Locality-sensitive Hashing. In Genetic and Evolutionary Computation
        Conference (GECCO '24), July 14-18, 2024, Melbourne, VIC, Australia. ACM,
        New York, NY, USA, 9 pages. https://doi.org/10.1145/3638529.3654147

        The inexact simplification algorithm works by mapping similar expressions
        to the same hash, and retrieving the simplest one when doing the
        simplification of an expression.
    use_arch: boolean, optional (default: True)
        Determines if we should save pareto front of the entire evolution
        (when set to  True) or just the final population (True).
    batch_size : float, default 1.0
        Percentage of training data to sample every generation. If `1.0`, then
        all data is used. Very small values can improve execution time, but 
        also lead to underfit.
    sel : str, default 'lexicase'
        The selection method to perform parent selection. When using lexicase,
        the selection is done as if it was a single-objective problem, based on 
        absolute error for regression, and log loss for classification.
    surv : str, default 'nsga2'
        The survival method for selecting the next generation from parents and offspring.
    save_population: str, optional (default "")
        string containing the path to save the final population. Ignored if
        not provided.
    load_population: str, optional (default "")
        string containing the path to load the initial population. Ignored
        if not provided.
    final_model_selection : str or function, optional (default "")
        specifies how the final model should be selected. If a function is 
        passed, then it will be applied over the population to select the
        final model. If a string is passed, then it should be one of the
        available options:
        * `""`: the model selected by the C++ engine is used. The C++ picks the
        model with best `scorer` objective value on the inner validation
        partition. If `validation_size` is set to zero, then the training 
        partition is used;
        * `"smallest_complexity"`: the non-dominated individual with the 
        smallest complexity, and more than one node in size (asserting it is
        a non-constant solution);
        * `"best_validation_ci"`: The less complex solution that is within
        the 95% confidence interval of the best solution's validation loss, with
        the confidence interval estimated with the inner validation partition of
        the data passed to `fit` or `fit_partial`;

        If a custom function is passed, then it should hhave the signature
        `Callable[[List[Dict], List[Dict]], Dict]]`, which means that it takes
        as arguments two lists of dicts (the entire population and the 
        pareto front represented as a list of individuals serialized as
        dictionaries, respectively), and returns a single individual (one 
        individual from any of the lists).
    bandit : str, optional (default: "dynamic_thompson")
        The bandit strategy to use for the estimator. Options are `"dummy"` that
        does not change the probabilities; `"thompson"` that uses static Thompson
        sampling to update sampling probabilities for terminals in search space with
        a static implementation; and `"dynamic_thompson" that implements a Thompson 
        strategy that weights more recent rewards and applies exponential decay
        to older observed rewards.
    shuffle_split: boolean, optional (default False)
        whether if the engine should shuffle the data before splitting it
        into train and validation partitions. Ignored if `validation_size`
        is set to zero.
    class_weights : list of float, default []
        List of weights to assign to each class in classification problems. 
        The length of the list should match the number of classes. If empty, all
        classes are assumed to have equal weight. This can be useful to handle
        imbalanced datasets by assigning weights to underrepresented classes.
    logfile: str, optional (default: "")
        If specified, spits statistics into a logfile. "" means don't log.
    random_state: int or None, default None
        If int, then the value is used to seed the c++ random generator; if None,
        then a seed will be generated using a non-deterministic generator. It is
        important to notice that, even if the random state is fixed, it is
        unlikely that running brush using multiple threads will have the same
        results. This happens because the Operating System's scheduler is
        responsible to choose which thread will run at any given time, thus 
        reproductibility is not guaranteed.
    """

    def __init__(self,
        mode: str = 'classification',
        pop_size: int = 100,
        max_gens: int = 100,
        max_time: int = -1,
        max_stall: int = 0,
        verbosity: int = 0,
        max_depth: int = 10,
        max_size: int = 100,
        num_islands: int = 5,
        n_jobs: int = 1,
        mig_prob: float = 0.05,
        cx_prob: float = 1/7,
        mutation_probs: Dict[str, float] = {"point":1/6, "insert":1/6, "delete":1/6, "subtree":1/6,
                          "toggle_weight_on":1/6, "toggle_weight_off":1/6},
        functions: Union[List[str], Dict[str, float]] = {},
        initialization: str = "uniform",
        objectives: List[str] = ["scorer", "linear_complexity"],
        scorer: str = None,
        algorithm: str = "nsga2",
        weights_init: bool = True,
        validation_size: float = 0.2,
        use_arch: bool = True,
        val_from_arch: bool = True,
        constants_simplification=True,
        inexact_simplification=True,
        batch_size: float = 1.0,
        sel: str = "lexicase",
        surv: str = "nsga2",
        final_model_selection: Union[str, Callable[[List[Dict], List[Dict]], Dict]] = "",
        save_population: str = "",
        load_population: str = "",
        bandit: str = 'dynamic_thompson',
        shuffle_split: bool = False,
        logfile: str = "",
        random_state: int = None,
        # class_weights: List[float] = [] # TODO: should we allow the user to set it?
    ) -> None:
        self.pop_size = pop_size
        self.max_gens = max_gens
        self.max_stall = max_stall
        self.max_time = max_time
        self.verbosity = verbosity
        self.algorithm = algorithm
        self.mode = mode
        self.max_depth = max_depth
        self.max_size = max_size
        self.num_islands = num_islands
        self.mig_prob = mig_prob
        self.n_jobs = n_jobs
        self.cx_prob = cx_prob
        self.bandit = bandit
        self.logfile = logfile
        self.final_model_selection = final_model_selection
        self.save_population = save_population
        self.load_population = load_population
        self.mutation_probs = mutation_probs
        self.val_from_arch = val_from_arch
        self.use_arch = use_arch
        self.functions = functions
        self.objectives = objectives
        self.constants_simplification=constants_simplification
        self.inexact_simplification=inexact_simplification
        self.scorer = scorer
        self.shuffle_split = shuffle_split
        self.initialization = initialization
        self.random_state = random_state
        self.batch_size = batch_size
        self.sel = sel
        self.surv = surv
        self.weights_init = weights_init
        self.validation_size = validation_size
        # self.class_weights = class_weights

    def _wrap_parameters(self, y, **extra_kwargs):
        """
        Creates a `Parameters` class to send to c++ backend the settings for
        the algorithm to use.
        """
        
        if isinstance(self.functions, list):
            self.functions_ = {k:1.0 for k in self.functions}
        else:
            self.functions_ = self.functions

        params = Parameters()

        # Setting up the classification or regression problem
        if self.mode == "classification":
            params.classification = True
            params.set_n_classes(y)
            params.set_class_weights(y)
            params.set_sample_weights(y)

        for obj in self.objectives:
            if obj not in ['scorer', 'size', 'complexity', 'linear_complexity', 'depth']:
                raise ValueError(f"Invalid objective {obj}. "
                                 "Valid objectives are: 'scorer', 'size', "
                                 "'complexity', 'linear_complexity', and 'depth'.")
                          
        params.objectives = self.objectives
        params.n_jobs = self.n_jobs

        # logging
        params.verbosity = self.verbosity
        params.logfile = self.logfile
        params.save_population = self.save_population
        params.load_population = self.load_population

        # Pop and archive
        params.use_arch = self.use_arch
        params.val_from_arch = self.val_from_arch

        # Simplification
        params.constants_simplification = self.constants_simplification
        params.inexact_simplification = self.inexact_simplification

        # Evolutionary loop
        params.pop_size = self.pop_size
        params.max_gens = self.max_gens
        params.num_islands = self.num_islands
        params.mig_prob = self.mig_prob
        params.max_depth = self.max_depth
        params.max_size = self.max_size
        params.cx_prob = self.cx_prob
        params.sel = self.sel
        params.surv = self.surv

        # Stop criteria 
        params.max_stall = self.max_stall
        params.max_time = self.max_time

        # Sampling probabilities
        params.weights_init = self.weights_init
        params.bandit = self.bandit
        params.mutation_probs = self.mutation_probs

        # Data management
        params.shuffle_split = self.shuffle_split
        params.functions = self.functions_
        params.validation_size = self.validation_size
        params.batch_size = self.batch_size
        params.feature_names = self.feature_names_
    
        # Scorer is the metric associated with "scorer" objective. To optimize
        # something else, set it in the objectives list.
        if self.scorer is None:
            scorer = "mse"
            if self.mode == "classification":
                scorer = "log" if params.n_classes == 2 else "multi_log"
            self.scorer = scorer
        else:
            if self.mode == "regression":
                assert self.scorer in ['mse'], \
                    "Invalid scorer for the regression mode"
            else:
                assert self.scorer in ['log', 'multi_log', 'balanced_accuracy',
                                       'accuracy', 'average_precision_score'], \
                    "Invalid scorer for the classification mode"
                
        params.scorer = self.scorer
       
        if self.random_state is not None:
            seed = 0
            if isinstance(self.random_state, np.random.Generator):
                seed = self.random_state.integers(1_000_000)
            elif isinstance(self.random_state, int):
                seed = self.random_state
            else:
                raise ValueError("random_state must be either a numpy random generator or an integer")

            params.random_state = seed

        for k, v in extra_kwargs.items():
            setattr(params, k, v)

        return params

    
    def _make_data(self, X, y=None,
                    feature_names=[], feature_types=[],
                    validation_size=0.0, shuffle_split=False):
        """
        Prepare the data for training or prediction.

        Parameters:
        - X: array-like or pandas DataFrame, shape (n_samples, n_features)
            The input features.
        - y: array-like or pandas Series, shape (n_samples,), optional (default=None)
            The target variable.
        - feature_names: list, optional (default=[])
            The names of the features.
        - validation_size: float, optional (default=0.0)
            The proportion of the data to be used for validation.
        - shuffle_split: bool, optional (default=False)
            Whether to shuffle and split the data.

        Returns:
        - dataset: Dataset
            The prepared dataset object containing the input features, target variable,
            feature names, and validation size.
        """
                
        # This function should not split the data (since it may be used in `predict`).
        # Partitioning is done by `fit`. Feature names should be inferred
        # before calling `_make_data` (so predict can be made with np arrays or
        # pd dataframes).

        if isinstance(y, pd.Series):
            y = y.values
        if isinstance(X, pd.DataFrame):
            X = X.values

        assert isinstance(X, np.ndarray)

        if y is None:
            return Dataset(X=X,
                    feature_names=feature_names,
                    feature_types=feature_types,
                    validation_size=validation_size,
                    shuffle_split=shuffle_split,
                    c=(self.mode=='classification') )

        return Dataset(X=X, y=y,
            feature_names=feature_names,
            feature_types=feature_types,
            validation_size=validation_size,
            shuffle_split=shuffle_split,
            c=(self.mode=='classification'))

    # Serializing only the stuff to make new predictions
    def __getstate__(self):
        state = self.__dict__.copy()

        # Serialization of data is not yet supported. TODO.
        del state["data_"]
        del state["train_"]
        del state["validation_"]
        del state["search_space_"]

        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # self.data_ = None
        # self.train_ = None
        # self.validation_ = None
        # self.search_space_ = None