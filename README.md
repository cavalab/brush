# brush

Brush is a strongly-typed genetic programming language. 
It is designed for **b**ackpropagations and **r**ecursion **u**sing **s**earch **h**euristics.

Goals:

- flexibility to define n-ary trees of operators on data of variable types (singletons, arrays, matrices of floats, ints, and bools)
- support for gradient descent over these programs
- support for recursive splits that flow with gradients
- fast-ish in C++
- easy-to-use Python API

# Installation 

## Install the brush environment

```
conda env create
```

## Overview

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

If you are just working on the Python package, you can install by running

```text
pip install .
```

from the repo root directory.
Pip will install the `brush` module and call `CMake` to build the `_brush` extension.   
It will not build the docs or cpp tests. 

## Installing the cpp tests

If you are developing the cpp code and want to build the cpp tests, run the following: 

```
./configure
./install tests
```