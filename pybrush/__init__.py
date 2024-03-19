# Interfaces for Brush data structures. Use to prototype with Brush
from _brush import Dataset
from _brush import SearchSpace
from _brush import Parameters

# geting random floats with the same engine
from _brush import rnd_flt as brush_rng

# Individuals
from _brush import individual #RegressorIndividual, ClassifierIndividual, MultiClassifierIndividual

# c++ learning engines. These are wrapped into a scikit-learn-like estimator in the python side
from _brush import engine # RegressorEngine, ClassifierEngine, MultiClassifierEngine


# Population modifiers
from _brush import RegressorEvaluator, ClassifierEvaluator, MultiClassifierEvaluator
from _brush import RegressorSelector, ClassifierSelector, MultiClassifierSelector
from _brush import RegressorVariator, ClassifierVariator, MultiClassifierVariator
# --------------------

# --------------------
# Prototyping an EA using brush classes, but other EA framework
from pybrush.DeapEstimator import DeapClassifier, DeapRegressor
from pybrush.BrushEstimator import BrushClassifier, BrushRegressor
