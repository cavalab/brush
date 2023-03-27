# Brush

<!-- start overview -->

Brush is a strongly-typed genetic programming language. 
It is designed for **b**ackpropagations and **r**ecursion **u**sing **s**earch **h**euristics.

This project is **very much** under active development. 
Expect api changes, bugs, breaking things, etc. 

## Goals

- flexibility to define n-ary trees of operators on data of variable types (singletons, arrays, matrices of floats, ints, and bools)
- support for gradient descent over these programs
- support for recursive splits that flow with gradients
- fast-ish in C++
- easy-to-use Python API

## Contact

Maintained by William La Cava @lacava (william.lacava@childrens.harvard.edu)

## Acknowledgments

This work is supported by grant R00-LM012926 from the National Library of Medicine. 
Brush is being developed to learn clinical diagnostics in the [Cava Lab at Harvard Medical School](http://cavalab.org). 

## License

GNU GPLv3, see [LICENSE](https://github.com/cavalab/brush/blob/master/LICENSE)

<!-- end overview -->

# Documentation

For the user guide and API, see the [docs](https://cavalab.org/brush).

# Installation 

<!-- start installation -->

## Install the brush environment

```
conda env create
```

If you are just using (not editing) the Python package:

```text
pip install .
```

from the repo root directory.

## Development 

```text
python setup.py develop
```

Gives you an editable install for messing with Python code in the project. 
(Any underyling cpp changes require this command to be re-run).

## Package Structure

There are a few different moving parts that can be built in this project:

- the cpp brush library (called `cbrush`)
- the cpp tests, written google tests (an executable named `tests`)
    - depends on `cbrush`
- the cpp-python bindings (a Python module written in cpp named `_brush`)
    - depends on `cbrush`
- the `brush` Python module
    - depends on `_brush`
- the docs (built with a combination of Sphinx and Doxygen)
    - depends on `brush`


Pip will install the `brush` module and call `CMake` to build the `_brush` extension.   
It will not build the docs or cpp tests. 

## Tests

## Python

The tests are run by calling pytest from the root directory. 

```bash
pytest 
```

## Cpp

If you are developing the cpp code and want to build the cpp tests, run the following: 

```
./configure
./install tests
```

<!-- end installation -->


## Contributing

<!-- start contributing -->

Please follow the [Github flow](https://guides.github.com/introduction/flow/) guidelines for contributing to this project.

In general, this is the approach:

-   Fork the repo into your own repository and clone it locally.

    ```
    git clone https://github.com/my_user_name/brush
    ```

-   Have an idea for a code change. Checkout a new branch with an
    appropriate name.

    ```
    git checkout -b my_new_change
    ```

-   Make your changes.
-   Commit your changes to the branch.

    ```
    git commit -m "adds my new change"
    ```

-   Check that your branch has no conflict with Brush's master branch by
    merging the master branch from the upstream repo.

    ```
    git remote add upstream https://github.com/cavalab/brush
    git fetch upstream
    git merge upstream/master
    ```

-   Fix any conflicts and commit.

    ```
    git commit -m "Merges upstream master"
    ```

-   Push the branch to your forked repo.

    ```
    git push origin my_new_change
    ```

-   Go to either Github repo and make a new Pull Request for your forked
    branch. Be sure to reference any relevant issues.

<!-- end contributing -->


## Basic Usage

<!-- start basics -->

Brush is designed to be used similarly to any [sklearn-style estimator](https://sklearn.org).
That means it should be compatible with sklearn pipelines, wrappers, and so forth. 

In addition, Brush provides functionality that allows you to feed in more complicated data types than just matrices of floating point values. 

## Regression

```python
# load data
import pandas as pd
df = pd.read_csv('docs/examples/datasets/d_enc.csv')
X = dfr.drop(columns='label')
y = dfr['label']

# import and make a regressor
import pandas as pd
from brush import BrushRegressor
est = BrushRegressor()

# use like you would a sklearn regressor
est.fit(X,y)
y_pred = est.predict(X)

print('score:', est.score(X,y))
```

## Classification

```python
# load data
import pandas as pd
df = pd.read_csv('docs/examples/datasets/d_analcatdata_aids.csv')
X = dfc.drop(columns='target')
y = dfc['target']

# import and make a classifier
import pandas as pd
from brush import BrushClassifier
est = BrushClassifier()
# use like you would a sklearn classifier
est.fit(X,y)
y_pred = est.predict(X)
y_pred_proba = est.predict_proba(X)

print('score:', est.score(X,y))
```

<!-- end basics -->

