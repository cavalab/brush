/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/
#ifndef NODELIST_H
#define NODELIST_H
//internal includes
#include "node.h"
#include "operators.h"
#include "util/utils.h"
#include "util/rnd.h"
using std::vector;
/* TODO
* instead of specifying keys, values, just specify the values into a list of
* some kind, and then loop thru the list and construct a map using the node
* name as the key.
*/
/* template<typename R, typename Arg1, typename Arg2=Arg1> */
/* Defines the search space of Brush. 
 *  The search spaces consists of nodes and their accompanying probability
 *  distribution. 
 *  Nodes can be accessed by name using a string map. 
 *  Alternatively, the functions and terminal sets may be sampled separately
 *  or together. 
 *  You may also sample the search space by return type. 
 *  Sampling is done in proportion to the weight associated with 
 *  each node. By default, sampling is done uniform randomly.
*/
using Brush::NodeType; 
using Brush::DataType; 
using Brush::ExecType; 
namespace Brush {

auto BinaryFF  = {
    { DataType::ArrayF, { DataType::ArrayF,DataType::ArrayF } },
    { DataType::MatrixF, { DataType::MatrixF,DataType::MatrixF } },
    { DataType::TimeSeriesF, { DataType::TimeSeriesF,DataType::TimeSeriesF } },
};

auto BinaryBB  = {
    { DataType::ArrayB, { DataType::ArrayB,DataType::ArrayB } },
    { DataType::MatrixB, { DataType::MatrixB,DataType::MatrixB } },
    { DataType::TimeSeriesB, { DataType::TimeSeriesB,DataType::TimeSeriesB } },
};

auto BinaryBF  = {
    { DataType::ArrayB, { DataType::ArrayF, DataType::ArrayF } },
    { DataType::MatrixB, { DataType::MatrixF,DataType::MatrixF } },
    { DataType::ArrayB, { DataType::TimeSeriesF,DataType::TimeSeriesF } },
};

auto UnaryFF  = {
    { DataType::ArrayF, { DataType::ArrayF} },
    { DataType::MatrixF, { DataType::MatrixF } },
    { DataType::TimeSeriesF, { DataType::TimeSeriesF } },
};
auto UnaryBB  = {
    { DataType::ArrayB, { DataType::ArrayB } },
    { DataType::MatrixB, { DataType::MatrixB } },
    { DataType::TimeSeriesB, { DataType::TimeSeriesB } },
};

auto ReduceFF  = {
    { DataType::ArrayF, { DataType::MatrixF } },
    { DataType::ArrayF, { DataType::TimeSeriesF } },
};
auto ReduceIA  = {
    { DataType::ArrayI, { DataType::MatrixF } },
    { DataType::ArrayI, { DataType::MatrixI } },
    { DataType::ArrayI, { DataType::TimeSeriesI } },
    { DataType::ArrayI, { DataType::TimeSeriesF } },
};
auto ReduceAll  = {
    { DataType::ArrayF, { DataType::MatrixF } },
    { DataType::ArrayF, { DataType::TimeSeriesF } },
    { DataType::ArrayI, { DataType::MatrixI } },
    { DataType::ArrayI, { DataType::TimeSeriesI } },
};


auto NodeSchema = {

//arithmetic
    {NodeType::Add, {
        {"ExecType", ExecType::Mapper},
        {"Mapping", BinaryFF},
                    },
    },
    {NodeType::Sub, {
        {"ExecType", ExecType::Mapper},
        {"Mapping", BinaryFF},
                    },
    },
    {NodeType::Mul, {
        {"ExecType", ExecType::Mapper},
        {"Mapping", BinaryFF},
                    },
    },
    {NodeType::Div, {
        {"ExecType", ExecType::Mapper},
        {"Mapping", BinaryFF},
                    },
    },
    {NodeType::Aq, {
        {"ExecType", ExecType::Mapper},
        {"Mapping", BinaryFF},
                   },
    },
    {NodeType::Abs, {
        {"ExecType", ExecType::Mapper},
        {"Mapping", UnaryFF},
                    },
    },
    {NodeType::Cbrt, {
            {"ExecType", ExecType::Mapper},
            {"Mapping", UnaryFF},
                     },
    },
    {NodeType::Ceil, {
            {"ExecType", ExecType::Mapper},
            {"Mapping", UnaryIF},
                     },
    },
    {NodeType::Floor, {
            {"ExecType", ExecType::Mapper},
            {"Mapping", UnaryIF},
                      },
    },
    {NodeType::Exp, {
        {"ExecType", ExecType::Mapper},
        {"Mapping", UnaryFF},
                    },
    },
    {NodeType::Log, {
        {"ExecType", ExecType::Mapper},
        {"Mapping", UnaryFF},
                    },
    },
    {NodeType::Logabs, {
            {"ExecType", ExecType::Mapper},
            {"Mapping", UnaryFF},
                       },
    },
    {NodeType::Log1p, {
            {"ExecType", ExecType::Mapper},
            {"Mapping", UnaryFF},
                      },
    },
    {NodeType::Sqrt, {
            {"ExecType", ExecType::Mapper},
            {"Mapping", UnaryFF},
                     },
    },
    {NodeType::Sqrtabs, {
            {"ExecType", ExecType::Mapper},
            {"Mapping", UnaryFF},
                        },
    },
    {NodeType::Square, {
            {"ExecType", ExecType::Mapper},
            {"Mapping", UnaryFF},
                       },
    },
    {NodeType::Pow, {
        {"ExecType", ExecType::Mapper},
        {"Mapping", UnaryFF},
                    },
    },
// trigonometry
    {NodeType::Acos, {
            {"ExecType", ExecType::Mapper},
            {"Mapping", UnaryFF},
                     },
    },
    {NodeType::Asin, {
            {"ExecType", ExecType::Mapper},
            {"Mapping", UnaryFF},
                     },
    },
    {NodeType::Atan, {
            {"ExecType", ExecType::Mapper},
            {"Mapping", UnaryFF},
                     },
    },
    {NodeType::Cos, {
        {"ExecType", ExecType::Mapper},
        {"Mapping", UnaryFF},
                    },
    },
    {NodeType::Cosh, {
            {"ExecType", ExecType::Mapper},
            {"Mapping", UnaryFF},
                     },
    },
    {NodeType::Sin, {
        {"ExecType", ExecType::Mapper},
        {"Mapping", UnaryFF},
                    },
    },
    {NodeType::Sinh, {
            {"ExecType", ExecType::Mapper},
            {"Mapping", UnaryFF},
                     },
    },
    {NodeType::Tan, {
        {"ExecType", ExecType::Mapper},
        {"Mapping", UnaryFF},
                    },
    },
    {NodeType::Tanh, {
            {"ExecType", ExecType::Mapper},
            {"Mapping", UnaryFF},
                     },
    },

// logic; not sure these will make it in
    {NodeType::And, {
        {"ExecType", ExecType::Mapper},
        {"Mapping", BinaryBB},
                    },
    },
    {NodeType::Or, {
        {"ExecType", ExecType::Mapper},
        {"Mapping", BinaryBB},
                   },
    },
    {NodeType::Not, {
        {"ExecType", ExecType::Mapper},
        {"Mapping", UnaryBB},
                    },
    },
    {NodeType::Xor, {
        {"ExecType", ExecType::Mapper},
        {"Mapping", BinaryBB},
                    },
    },

// decision (same)
    {NodeType::Equals, {
            {"ExecType", ExecType::Mapper},
            {"Mapping", BinaryBF},
                       },
    },
    {NodeType::LessThan, {
                {"ExecType", ExecType::Mapper},
                {"Mapping", BinaryBF},
                         },
    },
    {NodeType::GreaterThan, {
                {"ExecType", ExecType::Mapper},
                {"Mapping", BinaryBF},
                            },
    },
    {NodeType::Leq, {
        {"ExecType", ExecType::Mapper},
        {"Mapping", BinaryBF},
                    },
    },
    {NodeType::Geq, {
        {"ExecType", ExecType::Mapper},
        {"Mapping", BinaryBF},
                    },
    },

// summary stats
    {NodeType::Min, {
        {"ExecType", ExecType::Mapper},
        {"Mapping", ReduceAll},
                    },
    },
    {NodeType::Max, {
        {"ExecType", ExecType::Mapper},
        {"Mapping", ReduceAll},
                    },
    },
    {NodeType::Mean, {
            {"ExecType", ExecType::Mapper},
            {"Mapping", ReduceFF},
                     },
    },
    {NodeType::Median, {
            {"ExecType", ExecType::Mapper},
            {"Mapping", ReduceFF},
                       },
    },
    {NodeType::Count, {
            {"ExecType", ExecType::Mapper},
            {"Mapping", ReduceIA},
                      },
    },
    {NodeType::Sum, {
        {"ExecType", ExecType::Mapper},
        {"Mapping", ReduceAll},
                    },
    },

// timing masks
    {NodeType::Before, {
            {"ExecType", ExecType::Mapper},
            {"Mapping", BinaryFF},
                       },
    },
    {NodeType::After, {
            {"ExecType", ExecType::Mapper},
            {"Mapping", BinaryFF},
                      },
    },
    {NodeType::During, {
            {"ExecType", ExecType::Mapper},
            {"Mapping", BinaryFF},
                       },
    },

//split
    {NodeType::SplitBest, {
                {"ExecType", ExecType::Mapper},
                {"Mapping", {DataType::ArrayF, {DataType::ArrayF,DataType::ArrayF}},
                          },
    },
    {NodeType::SplitOn, {
            {"ExecType", ExecType::Mapper},
            {"Mapping", {DataType::ArrayF, {DataType::ArrayF,
                                            DataType::ArrayF,
                                            DataType::ArrayF}
                        },
                        },
    },

// leaves
    {NodeType::Constant, {
                {"ExecType", ExecType::Mapper},
                {"Mapping", BinaryFF},
                         },
    },
    {NodeType::Variable, {
                {"ExecType", ExecType::Mapper},
                {"Mapping", BinaryFF},
                         },
    },

// custom
    {NodeType::CustomOp, {
                {"ExecType", ExecType::Mapper},
                {"Mapping", BinaryFF},
                         },
    },
    {NodeType::CustomSplit {
                {"ExecType", ExecType::Mapper},
                {"Mapping", BinaryFF},
                           },
    },
};
}
