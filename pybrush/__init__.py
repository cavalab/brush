from ._versionstr import __version__

# Interfaces for Brush data structures. Use to prototype with Brush
from ._brush import Dataset
from ._brush import SearchSpace
from ._brush import Parameters

# geting random floats with brush (avoid random state issues in parallel exec)
from ._brush import rnd_flt as brush_rng

from ._brush import individual # Individual classes (specific for each task)

# c++ learning engines
from ._brush.engine import *

# Evaluation, selection, and variation. used in python estimators
from ._brush import RegressorEvaluator, ClassifierEvaluator, MultiClassifierEvaluator
from ._brush import RegressorSelector, ClassifierSelector, MultiClassifierSelector
from ._brush import RegressorVariator, ClassifierVariator, MultiClassifierVariator

# full estimator implementations --------------------
from pybrush.DeapEstimator import DeapClassifier, DeapRegressor
from pybrush.BrushEstimator import BrushClassifier, BrushRegressor
