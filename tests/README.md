## Running C++ tests

Compile and install `brush` the standard way. From the root directory:

`./configure`

`./install`

Then, simply run the tests:

`./build/tests`

## Running Python tests

Install the python wrapper from the root directory:

`pip install .`

Make sure that you have the python package `pytest` installed. From the root directory of this repository simply call:

`pytest`