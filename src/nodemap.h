/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/
#ifndef NODELIST_H
#define NODELIST_H
//internal includes
#include "init.h"
/* #include "node.h" */
#include "operators.h"
#include "data/data.h"
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
using namespace Brush;
/* using Brush::NodeType; */ 
/* using Brush::DataType; */ 
/* using Brush::ExecType; */ 
using std::tuple;
using std::array;
namespace Brush {

enum class NodeType : uint32_t {
    //arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Aq,
    Abs,

    Acos,
    Asin,
    Atan,
    Cos,
    Cosh,
    Sin,
    Sinh,
    Tan,
    Tanh,
    Cbrt,
    Ceil,
    Floor,
    Exp,
    Log,
    Logabs,
    Log1p,
    Sqrt,
    Sqrtabs,
    Square,
    Pow,

    // logic; not sure these will make it in
    And,
    Or,
    Not,
    Xor,

    // decision (same)
    Equals,
    LessThan,
    GreaterThan,
    Leq,
    Geq,

    // summary stats
    Min,
    Max,
    Mean,
    Median,
    Count,
    Sum,

    // timing masks
    Before,
    After,
    During,

    //split
    SplitBest,
    SplitOn,

    // leaves
    Constant,
    Variable,

    // custom
    CustomOp,
    CustomSplit
};

// OK: the signatures should be Tuples, just so the map has all the same types. 
// when grabbing the signature to get children (for things where Array types are needed),
// just grab the first element and make an array. 
// Otherwise we won't be able to support arguments with multiple types. 
// in Node(), we can make the arg_types vector of enum DataTypes using a mapping from 
// typeids to DataTypes. It will be a little clunky but also totally fine. 
//
auto BinaryFF  = {
    { DataType::ArrayF, tuple<Eigen::ArrayXf, Eigen::ArrayXf>() },
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
    { DataType::ArrayB, tuple<ArrayXb>() },
    { DataType::MatrixB, tuple<Eigen::ArrayXXb>() },
    { DataType::TimeSeriesB, tuple<TimeSeriesb>() },
};

auto ReduceFF  = {
    { DataType::ArrayF, tuple<Eigen::ArrayXXf>() },
    { DataType::ArrayF, tuple<TimeSeriesf>() },
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
        {"ExecType", ExecType::Reducer},
        {"Mapping", BinaryF},
                    },
    },
    {NodeType::Sub, {
        {"Arity", 2},
        {"ExecType", ExecType::Applier},
        {"Mapping", BinaryFF},
                    },
    },
    {NodeType::Mul, {
        {"Arity", 2},
        {"ExecType", ExecType::Reducer},
        {"Mapping", BinaryFF},
                    },
    },
    {NodeType::Div, {
        {"Arity", 2},
        {"ExecType", ExecType::Applier},
        {"Mapping", BinaryFF},
                    },
    },
    {NodeType::Aq, {
        {"Arity", 2},
        {"ExecType", ExecType::Applier},
        {"Mapping", BinaryFF},
                   },
    },
    {NodeType::Abs, {
        {"Arity", 1},
        {"ExecType", ExecType::Transformer},
        {"Mapping", UnaryFF},
                    },
    },
    {NodeType::Cbrt, {
        {"Arity", 1},
        {"ExecType", ExecType::Transformer},
        {"Mapping", UnaryFF},
                     },
    },
    {NodeType::Ceil, {
        {"Arity", 1},
            {"ExecType", ExecType::Transformer},
            {"Mapping", UnaryIF},
                     },
    },
    {NodeType::Floor, {
        {"Arity", 1},
            {"ExecType", ExecType::Transformer},
            {"Mapping", UnaryIF},
                      },
    },
    {NodeType::Exp, {
        {"Arity", 1},
        {"ExecType", ExecType::Transformer},
        {"Mapping", UnaryFF},
                    },
    },
    {NodeType::Log, {
        {"Arity", 1},
        {"ExecType", ExecType::Transformer},
        {"Mapping", UnaryFF},
                    },
    },
    {NodeType::Logabs, {
        {"Arity", 1},
            {"ExecType", ExecType::Transformer},
            {"Mapping", UnaryFF},
                       },
    },
    {NodeType::Log1p, {
        {"Arity", 1},
            {"ExecType", ExecType::Transformer},
            {"Mapping", UnaryFF},
                      },
    },
    {NodeType::Sqrt, {
        {"Arity", 1},
            {"ExecType", ExecType::Transformer},
            {"Mapping", UnaryFF},
                     },
    },
    {NodeType::Sqrtabs, {
        {"Arity", 1},
            {"ExecType", ExecType::Transformer},
            {"Mapping", UnaryFF},
                        },
    },
    {NodeType::Square, {
        {"Arity", 1},
            {"ExecType", ExecType::Transformer},
            {"Mapping", UnaryFF},
                       },
    },
    /* {NodeType::Pow, { */
    /*     {"Arity", 1}, */
    /*     {"ExecType", ExecType::Transformer}, */
    /*     {"Mapping", UnaryFF}, */
    /*                 }, */
    /* }, */
// trigonometry
    {NodeType::Acos, {
        {"Arity", 1},
            {"ExecType", ExecType::Transformer},
            {"Mapping", UnaryFF},
                     },
    },
    {NodeType::Asin, {
        {"Arity", 1},
            {"ExecType", ExecType::Transformer},
            {"Mapping", UnaryFF},
                     },
    },
    {NodeType::Atan, {
        {"Arity", 1},
            {"ExecType", ExecType::Transformer},
            {"Mapping", UnaryFF},
                     },
    },
    {NodeType::Cos, {
        {"Arity", 1},
        {"ExecType", ExecType::Transformer},
        {"Mapping", UnaryFF},
                    },
    },
    {NodeType::Cosh, {
        {"Arity", 1},
            {"ExecType", ExecType::Transformer},
            {"Mapping", UnaryFF},
                     },
    },
    {NodeType::Sin, {
        {"Arity", 1},
        {"ExecType", ExecType::Transformer},
        {"Mapping", UnaryFF},
                    },
    },
    {NodeType::Sinh, {
        {"Arity", 1},
            {"ExecType", ExecType::Transformer},
            {"Mapping", UnaryFF},
                     },
    },
    {NodeType::Tan, {
        {"Arity", 1},
        {"ExecType", ExecType::Transformer},
        {"Mapping", UnaryFF},
                    },
    },
    {NodeType::Tanh, {
        {"Arity", 1},
            {"ExecType", ExecType::Transformer},
            {"Mapping", UnaryFF},
                     },
    },

// logic; not sure these will make it in
    {NodeType::And, {
        {"Arity", 1},
        {"ExecType", ExecType::Reducer},
        {"Mapping", BinaryBB},
                    },
    },
    {NodeType::Or, {
        {"ExecType", ExecType::Reducer},
        {"Mapping", BinaryBB},
                   },
    },
    {NodeType::Not, {
        {"ExecType", ExecType::Transformer},
        {"Mapping", UnaryBB},
                    },
    },
    {NodeType::Xor, {
        {"ExecType", ExecType::Applier},
        {"Mapping", BinaryBB},
                    },
    },

// decision (same)
    {NodeType::Equals, {
            {"ExecType", ExecType::Reducer},
            {"Mapping", BinaryBF},
                       },
    },
    {NodeType::LessThan, {
                {"ExecType", ExecType::Applier},
                {"Mapping", BinaryBF},
                         },
    },
    {NodeType::GreaterThan, {
                {"ExecType", ExecType::Applier},
                {"Mapping", BinaryBF},
                            },
    },
    {NodeType::Leq, {
        {"ExecType", ExecType::Applier},
        {"Mapping", BinaryBF},
                    },
    },
    {NodeType::Geq, {
        {"ExecType", ExecType::Applier},
        {"Mapping", BinaryBF},
                    },
    },

// summary stats
    {NodeType::Min, {
        {"ExecType", ExecType::Reducer},
        {"Mapping", ReduceAll},
                    },
    },
    {NodeType::Max, {
        {"ExecType", ExecType::Reducer},
        {"Mapping", ReduceAll},
                    },
    },
    {NodeType::Mean, {
            {"ExecType", ExecType::Applier},
            {"Mapping", ReduceFF},
                     },
    },
    {NodeType::Median, {
            {"ExecType", ExecType::Applier},
            {"Mapping", ReduceFF},
                       },
    },
    {NodeType::Count, {
            {"ExecType", ExecType::Applier},
            {"Mapping", ReduceIA},
                      },
    },
    {NodeType::Sum, {
        {"ExecType", ExecType::Applier},
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

#endif
