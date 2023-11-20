/* Brush

module.h : Python interface to Brush classes and functions, using pybind11

copyright 2021 William La Cava
authors: William La Cava and Joseph D. Romano
license: GNU/GPL v3
*/

#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include <pybind11/iostream.h>

// json support
#include "pybind11_json/pybind11_json.hpp"
#include "nlohmann/json.hpp"