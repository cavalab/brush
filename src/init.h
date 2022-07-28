/* Brush
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#ifndef INIT_H
#define INIT_H

#ifdef _OPENMP
    #include <omp.h>
#else
    #define omp_get_thread_num() 0
    #define omp_get_num_threads() 1
    #define omp_get_max_threads() 1
    #define omp_set_num_threads( x ) 0
#endif
// stuff being used
#include "stdint.h"
#include <Eigen/Dense>
#include <memory>
#include <iostream>
#include <fstream>
#include <numeric>
#include <map>
#include <set>
#include <vector>
#include <string>
/* #include <fmt/core.h> */
#include <fmt/ostream.h> 
#include <fmt/format.h>
#include <fmt/ranges.h>

using Eigen::MatrixXf;
using Eigen::ArrayXXf;
using Eigen::ArrayXXi;
using Eigen::VectorXf;
using Eigen::VectorXi;
using Eigen::ArrayXf;
using Eigen::seq;
typedef Eigen::Array<bool,Eigen::Dynamic,1> ArrayXb;
typedef Eigen::Array<bool,Eigen::Dynamic,Eigen::Dynamic> ArrayXXb;
typedef Eigen::Matrix<bool,Eigen::Dynamic,1> VectorXb;
typedef Eigen::Matrix<long,Eigen::Dynamic,1> VectorXl;
// STD
using std::map;
using std::vector;
using std::set;
using std::string;
using std::unique_ptr;
using std::shared_ptr;
using std::make_shared;
using std::cout; 
typedef std::map<string, 
                 std::pair<vector<ArrayXf>, vector<ArrayXf>>
                > LongData;
using Eigen::Dynamic;
using Eigen::Map;
// internal includes
#include "thirdparty/json.hpp"
using nlohmann::json;

static float NEAR_ZERO = 0.0000001;
static float MAX_FLT = std::numeric_limits<float>::max();
static float MIN_FLT = std::numeric_limits<float>::lowest();

namespace Brush{
// helper constant for the visitor
template<class> inline constexpr bool always_false_v = false;
// explicit deduction guide (not needed as of C++20)
/* template<class... Ts> struct overloaded : Ts... { using Ts::operator()...; }; */
/* template<class... Ts> overloaded(Ts...) -> overloaded<Ts...>; */
}

#endif
