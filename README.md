# Brush

<!-- start overview -->

Brush is an interpretable machine learning library for training symbolic models. 
It wraps multiple learning paradigms (gradient descent, decision trees, symbolic regression) into a strongly-typed genetic programming language (Montana, 1995 [PDF](http://davidmontana.net/papers/stgp.pdf)). 

This project is **very much** under active development. 
Expect api changes and broken things.  

For the user guide and API, see the [docs](https://cavalab.org/brush).

## Features / Design Goals

- Flexibility to define n-ary trees of operators on data of variable types (singletons, arrays, time series, matrices of floats, ints, and bools)
- Support for gradient descent over these programs
- Support for recursive splits that flow with gradients
- Fast-ish in C++
- Easy-to-use Python API with low-level bindings

## Contact

Brush is maintained by William La Cava ([@lacava](https://github.com/lacava), william.lacava@childrens.harvard.edu) and initially authored by him and Joseph D. Romano ([@JDRomano2](https://github.com/JDRomano2)). 

Special thanks to these contributors:

- Guilherme Aldeia ([@gAldeia](https://github.com/gAldeia))
- Fabricio Olivetti de Franca ([@folivetti](https://github.com/folivetti))
- Zongjun Liu ([@msnliu](https://github.com/msnliu))
- Daniel S. Herman ([@hermands](https://github.com/hermands))


## Acknowledgments

Brush is being developed to improve clinical diagnostics in the [Cava Lab at Harvard Medical School](http://cavalab.org). 
This work is partially funded by grant R00-LM012926 from the National Library of Medicine and a Patient-Centered Outcomes Research Institute (PCORI) Award (ME-2020C1D-19393).

## License

GNU GPLv3, see [LICENSE](https://github.com/cavalab/brush/blob/master/LICENSE)

<!-- end overview -->

# Quickstart 

## Installation

### Installation via Python wheel and `pip` (recommended)

First create an environment and install `gxx` and the `cpp` dependencies. You can easily do it by running `conda env create -f environment.yml` inside the root.

Then, `pip install .` will build and install the main package.

To install the extended functionalities that depends on `deap`, you can do `pip install .[deap]`. It may be necessary to wrap it in quotes in some cases (e.g. zsh, `pip install '.[deap]`.)

### Manual installation

<!-- start installation -->
Clone the repo:

```
git clone https://github.com/cavalab/brush.git
```

Install the brush environment:

```
cd brush
conda env create -f environment.yml
```

Install brush: 

```text
pip install .
```

from the repo root directory.
If you are just planning to develop, see [Development](#development).

<!-- end installation -->



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
X = df.drop(columns='label')
y = df['label']

# import and make a regressor
from pybrush import BrushRegressor

# you can set verbosity=1 to see the progress bar
est = BrushRegressor(verbosity=1)

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
X = df.drop(columns='target')
y = df['target']

# import and make a classifier
from pybrush import BrushClassifier
est = BrushClassifier(verbosity=1)

# use like you would a sklearn classifier
est.fit(X,y)

y_pred = est.predict(X)
y_pred_proba = est.predict_proba(X)

print('score:', est.score(X,y))
```

<!-- end basics -->


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

# Development 
<!-- start development -->

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

### Python

The tests are run by calling pytest from the root directory. You may need to install brush with `pip install '.[test]'` to install all required dependencies


```bash
pytest 
```

### Cpp

If you are developing the cpp code and want to build the cpp tests, run the following: 

```
./configure
./install tests
```

## Building the docs locally

To build the documentation you will need some additional requirements.
Before proceeding, make sure you have the python wrapper installed, as the documentation have some sample notebooks that will run the code.

First go to the `docs` folder:

```bash
cd docs/
```

Then, install additional python packages in the same environemnt as brush is intalled with:

```bash
conda activate brush
pip install -r requirements.txt
```

Now just run:

```bash
make html
```

The static website is located in `-build/html`

## Building the python docs

First you need to modify the `CMakeLists.txt` to enable building the docs by changing `option(DOCS "build docs" OFF)` to `ON`, as well as `"-DDOCS=OFF"` to `ON` in the `pyproject.toml`.

Then you can install brush with `pip install '.[docs]'` and it will build the docs as it install the module.

<!-- end development -->
