#ifndef TESTS_HEADER_H
#define TESTS_HEADER_H

#include <vector>
#include <iostream>
#include <Eigen/Dense>
#include <memory>
#include <omp.h>
#include <string>
#include <stack>
#include <gtest/gtest.h>	

// stuff being used

// using namespace std;

typedef Eigen::Array<bool,Eigen::Dynamic,1> ArrayXb;
using std::shared_ptr;
using std::make_shared;
using std::cout;
using std::stoi;
using std::to_string;
using std::stof;

#define private public

#include <cstdio>
#include "../init.h"
#include "../data/data.h"
#include "../search_space.h"
using namespace Brush;
using namespace Brush::data;

#endif
