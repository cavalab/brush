/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/
#ifndef NODELIST_H
#define NODELIST_H
//internal includes
#include "init.h"
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
using std::tuple;
using std::array;
namespace Brush {

auto BinaryFF  = {
    { DataType::ArrayF, {
        {"InType", {ArrayF,ArrayF}},
        {"Signature", array<Eigen::ArrayXf, 2>() },
                        }
    },
    { DataType::MatrixF, tuple<Eigen::ArrayXXf,Eigen::ArrayXXf>() },
    { DataType::TimeSeriesF, tuple<TimeSeriesf,TimeSeriesf>() },
};
/* auto BinaryFF  = { */
/*     { DataType::ArrayF, tuple<ArrayF,ArrayF>() }, */
/*     { DataType::MatrixF, tuple<MatrixF,MatrixF>() }, */
/*     { DataType::TimeSeriesF, tuple<TimeSeriesF,TimeSeriesF>() }, */
/* }; */

auto BinaryBB  = {
    { DataType::ArrayB, tuple<ArrayXb,ArrayXb>() },
    { DataType::MatrixB, tuple<ArrayXXb,ArrayXXb>() },
    { DataType::TimeSeriesB, tuple<TimeSeriesb,TimeSeriesb>() },
};

auto BinaryBF  = {
    { DataType::ArrayB, tuple<Eigen::ArrayXf, Eigen::ArrayXf>() },
    { DataType::MatrixB, tuple<Eigen::ArrayXXf,Eigen::ArrayXXf>() },
    { DataType::ArrayB, tuple<TimeSeriesf,TimeSeriesf>() },
};

auto UnaryFF  = {
    { DataType::ArrayF, tuple<Eigen::ArrayXf>() },
    { DataType::MatrixF, tuple<Eigen::ArrayXXf>() },
    { DataType::TimeSeriesF, tuple<TimeSeriesXf>() },
};
auto UnaryBB  = {
    { DataType::ArrayB, tuple<Eigen::ArrayXb>() },
    { DataType::MatrixB, tuple<Eigen::ArrayXXb>() },
    { DataType::TimeSeriesB, tuple<TimeSeriesb>() },
};

auto ReduceFF  = {
    { DataType::ArrayF, tuple<Eigen::ArrayXXf>() },
    { DataType::ArrayF, tuple<TimeSeriesXf>() },
};
auto ReduceIA  = {
    { DataType::ArrayI, tuple<Eigen::ArrayXXf>() },
    { DataType::ArrayI, tuple<MatrixI>() },
    { DataType::ArrayI, tuple<TimeSeriesI>() },
    { DataType::ArrayI, tuple<TimeSeriesXf>() },
};
auto ReduceAll  = {
    { DataType::ArrayF, tuple<Eigen::ArrayXXf>() },
    { DataType::ArrayF, tuple<TimeSeriesXf>() },
    { DataType::ArrayI, tuple<Eigen::ArrayXXi>() },
    { DataType::ArrayI, tuple<TimeSeriesI>() },
};


auto NodeSchema = {

//arithmetic
    {NodeType::Add, {
        {"Arity", 2},
        {"ExecType", ExecType::Mapper},
        {"Mapping", BinaryFF},
        {"Signature", BinaryFFSig},
                    },
    },
    {NodeType::Sub, {
        {"Arity", 2},
        {"ExecType", ExecType::Mapper},
        {"Mapping", BinaryFF},
                    },
    },
    {NodeType::Mul, {
        {"Arity", 2},
        {"ExecType", ExecType::Mapper},
        {"Mapping", BinaryFF},
                    },
    },
    {NodeType::Div, {
        {"Arity", 2},
        {"ExecType", ExecType::Mapper},
        {"Mapping", BinaryFF},
                    },
    },
    {NodeType::Aq, {
        {"Arity", 2},
        {"ExecType", ExecType::Mapper},
        {"Mapping", BinaryFF},
                   },
    },
    {NodeType::Abs, {
        {"Arity", 1},
        {"ExecType", ExecType::Mapper},
        {"Mapping", UnaryFF},
                    },
    },
    {NodeType::Cbrt, {
        {"Arity", 1},
        {"ExecType", ExecType::Mapper},
        {"Mapping", UnaryFF},
                     },
    },
    {NodeType::Ceil, {
        {"Arity", 1},
            {"ExecType", ExecType::Mapper},
            {"Mapping", UnaryIF},
                     },
    },
    {NodeType::Floor, {
        {"Arity", 1},
            {"ExecType", ExecType::Mapper},
            {"Mapping", UnaryIF},
                      },
    },
    {NodeType::Exp, {
        {"Arity", 1},
        {"ExecType", ExecType::Mapper},
        {"Mapping", UnaryFF},
                    },
    },
    {NodeType::Log, {
        {"Arity", 1},
        {"ExecType", ExecType::Mapper},
        {"Mapping", UnaryFF},
                    },
    },
    {NodeType::Logabs, {
        {"Arity", 1},
            {"ExecType", ExecType::Mapper},
            {"Mapping", UnaryFF},
                       },
    },
    {NodeType::Log1p, {
        {"Arity", 1},
            {"ExecType", ExecType::Mapper},
            {"Mapping", UnaryFF},
                      },
    },
    {NodeType::Sqrt, {
        {"Arity", 1},
            {"ExecType", ExecType::Mapper},
            {"Mapping", UnaryFF},
                     },
    },
    {NodeType::Sqrtabs, {
        {"Arity", 1},
            {"ExecType", ExecType::Mapper},
            {"Mapping", UnaryFF},
                        },
    },
    {NodeType::Square, {
        {"Arity", 1},
            {"ExecType", ExecType::Mapper},
            {"Mapping", UnaryFF},
                       },
    },
    /* {NodeType::Pow, { */
    /*     {"Arity", 1}, */
    /*     {"ExecType", ExecType::Mapper}, */
    /*     {"Mapping", UnaryFF}, */
    /*                 }, */
    /* }, */
// trigonometry
    {NodeType::Acos, {
        {"Arity", 1},
            {"ExecType", ExecType::Mapper},
            {"Mapping", UnaryFF},
                     },
    },
    {NodeType::Asin, {
        {"Arity", 1},
            {"ExecType", ExecType::Mapper},
            {"Mapping", UnaryFF},
                     },
    },
    {NodeType::Atan, {
        {"Arity", 1},
            {"ExecType", ExecType::Mapper},
            {"Mapping", UnaryFF},
                     },
    },
    {NodeType::Cos, {
        {"Arity", 1},
        {"ExecType", ExecType::Mapper},
        {"Mapping", UnaryFF},
                    },
    },
    {NodeType::Cosh, {
        {"Arity", 1},
            {"ExecType", ExecType::Mapper},
            {"Mapping", UnaryFF},
                     },
    },
    {NodeType::Sin, {
        {"Arity", 1},
        {"ExecType", ExecType::Mapper},
        {"Mapping", UnaryFF},
                    },
    },
    {NodeType::Sinh, {
        {"Arity", 1},
            {"ExecType", ExecType::Mapper},
            {"Mapping", UnaryFF},
                     },
    },
    {NodeType::Tan, {
        {"Arity", 1},
        {"ExecType", ExecType::Mapper},
        {"Mapping", UnaryFF},
                    },
    },
    {NodeType::Tanh, {
        {"Arity", 1},
            {"ExecType", ExecType::Mapper},
            {"Mapping", UnaryFF},
                     },
    },

// logic; not sure these will make it in
    {NodeType::And, {
        {"Arity", 1},
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
                {"Mapping", {DataType::ArrayF, tuple<Eigen::ArrayXf,Eigen::ArrayXf>},
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
