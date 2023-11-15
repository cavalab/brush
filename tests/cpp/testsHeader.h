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
#include "../../src/init.h"
#include "../../src/params.h"
#include "../../src/data/data.h"
#include "../../src/program/operator.h"
#include "../../src/program/dispatch_table.h"
#include "../../src/program/program.h"
#include "../../src/individual.h"
#include "../../src/search_space.h"
#include "../../src/variation.cpp" // TODO: is this ok? (otherwise I would have to create a test separated file, or move the implementation to the header)

using namespace Brush;
using namespace Brush::Data;
using namespace Brush::Var;

#endif
