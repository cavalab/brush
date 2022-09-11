# brush

Brush is a strongly-typed genetic programming language. 
It is designed for **b**ackpropagations and **r**ecursion **u**sing **s**earch **h**euristics.

Goals:

- flexibility to define n-ary trees of operators on data of variable types (singletons, arrays, matrices of floats, ints, and bools)
- support for gradient descent over these programs
- support for recursive splits that flow with gradients
- fast-ish in C++
- easy-to-use Python API

# Installation instructions

- Install the brush environment

```
conda env create
```

- Configure the build

```
./configure
```

- Install C++ library

```
./install
```

- Install tests (optional)

```
./install tests
```

- Install Python package (optional)

```
pip install .
```