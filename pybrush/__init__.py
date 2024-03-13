# Interfaces for Brush classes. Use to prototype with Brush
from _brush import Dataset
from _brush import SearchSpace
from _brush import Parameters

# geting random floats
from _brush import rng_flt as brush_rng

# Population modifiers
from _brush import RegressorEvaluator, ClassifierEvaluator, MultiClassifierEvaluator
from _brush import RegressorSelector, ClassifierSelector, MultiClassifierSelector
from _brush import RegressorVariator, ClassifierVariator, MultiClassifierVariator

# Individuals
from _brush.individual import RegressorIndividual, \
                              ClassifierIndividual, MultiClassifierIndividual

# Prototyping an EA using brush classes, but other EA framework
from pybrush.DeapEstimator import DeapClassifier, DeapRegressor

# c++ learning engines. These are wrapped into a scikit-learn-like estimator in the python side
from _brush.engine import RegressorEngine, ClassifierEngine, MultiClassifierEngine