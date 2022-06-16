/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/
#ifndef NODELIST_H
#define NODELIST_H
// external includes
#include <bitset>
//internal includes
#include "init.h"
/* #include "node.h" */
/* #include "operators.h" */
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
/* using Brush::ExecType; */ 
using std::tuple;
using std::array;
using Brush::DataType; 
using Brush::data::TimeSeriesb; 
using Brush::data::TimeSeriesi; 
using Brush::data::TimeSeriesf; 

namespace Brush {

enum class ExecType : uint32_t {
    Unary, // no weights, just a unary call to apply
    Binary, // no weights, just a binary call to apply
    Transformer, // maps child nodes to output via function call.
    Reducer, // maps child nodes to output via reduction call.
    Applier, // maps child nodes to output via function call.
    Splitter, // splits data and returns the output of the children on each split.
    Terminal,    // returns a data element.
}; 
enum class NodeType : uint64_t {
    // arithmetic
    Add                 = 1UL << 1UL,
    Sub                 = 1UL << 2UL,
    Mul                 = 1UL << 3UL,
    Div                 = 1UL << 4UL,
    Aq                  = 1UL << 5UL,
    Abs                 = 1UL << 6UL,
    // trig
    Acos                = 1UL << 7UL,
    Asin                = 1UL << 8UL,
    Atan                = 1UL << 9UL,
    Cos                 = 1UL << 10UL,
    Cosh                = 1UL << 11UL,
    Sin                 = 1UL << 12UL,
    Sinh                = 1UL << 13UL,
    Tan                 = 1UL << 14UL,
    Tanh                = 1UL << 15UL,
    Cbrt                = 1UL << 16UL,
    Ceil                = 1UL << 17UL,
    Floor               = 1UL << 18UL,
    Exp                 = 1UL << 19UL,
    Log                 = 1UL << 20UL,
    Logabs              = 1UL << 21UL,
    Log1p               = 1UL << 22UL,
    Sqrt                = 1UL << 23UL,
    Sqrtabs             = 1UL << 24UL,
    Square              = 1UL << 25UL,
    Pow                 = 1UL << 26UL,
    // logic 
    And                 = 1UL << 27UL,
    Or                  = 1UL << 28UL,
    Not                 = 1UL << 29UL,
    Xor                 = 1UL << 30UL,
    // decision (same)
    Equals              = 1UL << 31UL,
    LessThan            = 1UL << 32UL,
    GreaterThan         = 1UL << 33UL,
    Leq                 = 1UL << 34UL,
    Geq                 = 1UL << 35UL,
    // summary stats
    Min                 = 1UL << 36UL,
    Max                 = 1UL << 37UL,
    Mean                = 1UL << 38UL,
    Median              = 1UL << 39UL,
    Count               = 1UL << 40UL,
    Sum                 = 1UL << 41UL,
    // timing masks
    Before              = 1UL << 42UL,
    After               = 1UL << 43UL,
    During              = 1UL << 44UL,
    //split
    SplitBest           = 1UL << 45UL,
    SplitOn             = 1UL << 46UL,
    // leaves
    Constant            = 1UL << 47UL,
    Terminal            = 1UL << 48UL,
    // custom
    CustomOp            = 1UL << 49UL,
    CustomSplit         = 1UL << 50UL,
    // Special Op
    _END_               = 1UL << 51UL,
};


using UnderlyingNodeType = std::underlying_type_t<NodeType>;
struct NodeTypes {
    // magic number keeping track of the number of different node types
    static constexpr size_t Count = 51;

    // returns the index of the given type in the NodeType enum
    static auto GetIndex(NodeType type) -> size_t
    {
        return std::bitset<Count>(static_cast<std::underlying_type_t<NodeType>>(type) - 1).count();
    }
};

inline constexpr auto operator&(NodeType lhs, NodeType rhs) -> NodeType { return static_cast<NodeType>(static_cast<UnderlyingNodeType>(lhs) & static_cast<UnderlyingNodeType>(rhs)); }
inline constexpr auto operator|(NodeType lhs, NodeType rhs) -> NodeType { return static_cast<NodeType>(static_cast<UnderlyingNodeType>(lhs) | static_cast<UnderlyingNodeType>(rhs)); }
inline constexpr auto operator^(NodeType lhs, NodeType rhs) -> NodeType { return static_cast<NodeType>(static_cast<UnderlyingNodeType>(lhs) ^ static_cast<UnderlyingNodeType>(rhs)); }
inline constexpr auto operator~(NodeType lhs) -> NodeType { return static_cast<NodeType>(~static_cast<UnderlyingNodeType>(lhs)); }
inline auto operator&=(NodeType& lhs, NodeType rhs) -> NodeType&
{
    lhs = lhs & rhs;
    return lhs;
}
inline auto operator|=(NodeType& lhs, NodeType rhs) -> NodeType&
{
    lhs = lhs | rhs;
    return lhs;
}
inline auto operator^=(NodeType& lhs, NodeType rhs) -> NodeType&
{
    lhs = lhs ^ rhs;
    return lhs;
}

std::map<std::string, NodeType> NodeNameType = {
    //arithmetic
    {"Add", NodeType::Add},
    {"Sub", NodeType::Sub},
    {"Mul", NodeType::Mul},
    {"Div", NodeType::Div},
    {"Aq", NodeType::Aq},
    {"Abs", NodeType::Abs},

    {"Acos", NodeType::Acos},
    {"Asin", NodeType::Asin},
    {"Atan", NodeType::Atan},
    {"Cos", NodeType::Cos},
    {"Cosh", NodeType::Cosh},
    {"Sin", NodeType::Sin},
    {"Sinh", NodeType::Sinh},
    {"Tan", NodeType::Tan},
    {"Tanh", NodeType::Tanh},
    {"Cbrt", NodeType::Cbrt},
    {"Ceil", NodeType::Ceil},
    {"Floor", NodeType::Floor},
    {"Exp", NodeType::Exp},
    {"Log", NodeType::Log},
    {"Logabs", NodeType::Logabs},
    {"Log1p", NodeType::Log1p},
    {"Sqrt", NodeType::Sqrt},
    {"Sqrtabs", NodeType::Sqrtabs},
    {"Square", NodeType::Square},
    {"Pow", NodeType::Pow},

    // logic; not sure these will make it in
    {"And", NodeType::And},
    {"Or", NodeType::Or},
    {"Not", NodeType::Not},
    {"Xor", NodeType::Xor},

    // decision (same)
    {"Equals", NodeType::Equals},
    {"LessThan", NodeType::LessThan},
    {"GreaterThan", NodeType::GreaterThan},
    {"Leq", NodeType::Leq},
    {"Geq", NodeType::Geq},

    // summary stats
    {"Min", NodeType::Min},
    {"Max", NodeType::Max},
    {"Mean", NodeType::Mean},
    {"Median", NodeType::Median},
    {"Count", NodeType::Count},
    {"Sum", NodeType::Sum},

    // timing masks
    {"Before", NodeType::Before},
    {"After", NodeType::After},
    {"During", NodeType::During},

    //split
    {"SplitBest", NodeType::SplitBest},
    {"SplitOn", NodeType::SplitOn},

    // leaves
    {"Constant", NodeType::Constant},
    {"Terminal", NodeType::Terminal},

    // custom
    {"CustomOp", NodeType::CustomOp},
    {"CustomSplit", NodeType::CustomSplit},
};

auto NodeTypeName = Util::reverse_map(NodeNameType);
// OK: the signatures should be Tuples, just so the map has all the same types. 
// when grabbing the signature to get children (for things where Array types are needed),
// just grab the first element and make an array. 
// Otherwise we won't be able to support arguments with multiple types. 
// in Node(), we can make the arg_types vector of enum DataTypes using a mapping from 
// typeids to DataTypes. It will be a little clunky but also totally fine. 
//
//
//
/* struct TupleOf */
/* { */
/*     // given a tuple of DataTypes (enums), return a tuple of the actual types */ 
/*     // */
/*     /1* using TupleArgs = std::tuple<Args...>; *1/ */
/*     /1* static constexpr std::size_t ArgCount = sizeof...(Args); *1/ */
/*     /1* using StateArrayArgs = std::array<State,ArgCount>; *1/ */
/*     /1* template <std::size_t N> *1/ */
/*     /1* using NthType = typename std::tuple_element<N, TupleArgs>::type; *1/ */

/*     /1* template<size_t... Is> *1/ */
/*     /1* vector<std::type_index> get_arg_types(std::index_sequence<Is...>) const *1/ */
/*     /1* { *1/ */
/*     /1*     return vector<type_index>{typeid(NthType<Is>)...}; *1/ */
/*     /1* } *1/ */
/*     template<size_t... Is, typename... Args> */
/*     auto tupleize(const tuple<Args...>& in, std::index_sequence<Is...>) */
/*     { */ 
/*         return std::make_tuple(data::DataMap<std::get<Is>(in)>()...); */
/*     }; */

/*     template<typename ... Args> */
/*     auto operator()(const tuple<Args...>& in) */
/*     { */
/*         static constexpr std::size_t ArgCount = sizeof...(Args); */
/*         return this->tupleize(in, std::make_index_sequence<ArgCount>{}); */
/*     }; */

/*     template<typename ... Args> */
/*     template<DataType T, DataType L, DataType V> */
/*     auto tupleof(T */
/* }; */

/* template<ExecType E = ExecType::Unary> */ 
/* auto TupleArgs(json& args) */ 
/* { */
/*     return make_tuple(DataMap<args.at(0)>()); */
/* }; */

/* template<> */ 
/* auto TupleArgs<ExecType::Binary>(json& args) */ 
/* { */
/*     return make_tuple(DataMap<args.at(0)>(), DataMap<args.at(1)>()); */
/* }; */
/* template<> */ 
/* auto TupleArgs<ExecType::Reduce>(json& args) */ 
/* { */
/*     return TupleArgs<ExecType::Binary>(args); */
/* }; */


/* template<> */ 
/* auto TupleArgs<ExecType::Transform>(json& args) */ 
/* { */
/*     return make_tuple(DataMap<args.at(0)>()); */
/* }; */


/* json BinaryFF  = { */
/*     { "ArrayF", tuple<Eigen::ArrayXf, Eigen::ArrayXf>() }, */
/*     { "MatrixF", tuple<Eigen::ArrayXXf,Eigen::ArrayXXf>() }, */
/*     { TimeSeriesF, tuple<TimeSeriesf,TimeSeriesf>() }, */
/* }; */
/* std::map<DataType,std::vector<DataType>> BinaryFFtoF  = { */
json BinaryFFtoF  = {
    { "ArrayF", {DataType::ArrayF, DataType::ArrayF }},
    { "MatrixF", {DataType::MatrixF,DataType::MatrixF }},
    { "TimeSeriesF", {DataType::TimeSeriesF,DataType::TimeSeriesF }},
};

json UnaryFtoF  = {
    { "ArrayF", {DataType::ArrayF }},
    { "MatrixF", {DataType::MatrixF }},
    { "TimeSeriesF", {DataType::TimeSeriesF }},
};

json NodeSchema = {
//arithmetic
    {"Add", { 
                {"ExecType", ExecType::Binary}, 
                {"Signature", BinaryFFtoF}, 
            },
    },
    {"Abs", { 
                {"ExecType", ExecType::Unary}, 
                {"Signature", UnaryFtoF}, 
            },
    },
};
/* auto BinaryFF  = { */
/*     { DataType::ArrayF, tuple<Eigen::ArrayXf, Eigen::ArrayXf>() }, */
/*     { DataType::MatrixF, tuple<Eigen::ArrayXXf,Eigen::ArrayXXf>() }, */
/*     { DataType::TimeSeriesF, tuple<TimeSeriesf,TimeSeriesf>() }, */
/* }; */
/* auto BinaryFF  = { */
/*     { DataType::ArrayF, tuple<ArrayF,ArrayF>() }, */
/*     { DataType::MatrixF, tuple<MatrixF,MatrixF>() }, */
/*     { DataType::TimeSeriesF, tuple<TimeSeriesF,TimeSeriesF>() }, */
/* }; */

/* auto BinaryBB  = { */
/*     { DataType::ArrayB, tuple<Eigen::ArrayXb,Eigen::ArrayXb>() }, */
/*     { DataType::MatrixB, tuple<ArrayXXb,ArrayXXb>() }, */
/*     { DataType::TimeSeriesB, tuple<TimeSeriesb,TimeSeriesb>() }, */
/* }; */

/* auto BinaryBF  = { */
/*     { DataType::ArrayB, tuple<Eigen::ArrayXf, Eigen::ArrayXf>() }, */
/*     { DataType::MatrixB, tuple<Eigen::ArrayXXf,Eigen::ArrayXXf>() }, */
/*     { DataType::ArrayB, tuple<TimeSeriesf,TimeSeriesf>() }, */
/* }; */

/* auto UnaryFF  = { */
/*     { DataType::ArrayF, tuple<Eigen::ArrayXf>() }, */
/*     { DataType::MatrixF, tuple<Eigen::ArrayXXf>() }, */
/*     { DataType::TimeSeriesF, tuple<TimeSeriesXf>() }, */
/* }; */
/* auto UnaryBB  = { */
/*     { DataType::ArrayB, tuple<ArrayXb>() }, */
/*     { DataType::MatrixB, tuple<Eigen::ArrayXXb>() }, */
/*     { DataType::TimeSeriesB, tuple<TimeSeriesb>() }, */
/* }; */

/* auto ReduceFF  = { */
/*     { DataType::ArrayF, tuple<Eigen::ArrayXXf>() }, */
/*     { DataType::ArrayF, tuple<TimeSeriesf>() }, */
/* }; */
/* auto ReduceIA  = { */
/*     { DataType::ArrayI, tuple<Eigen::ArrayXXf>() }, */
/*     { DataType::ArrayI, tuple<MatrixI>() }, */
/*     { DataType::ArrayI, tuple<TimeSeriesI>() }, */
/*     { DataType::ArrayI, tuple<TimeSeriesXf>() }, */
/* }; */
/* auto ReduceAll  = { */
/*     { DataType::ArrayF, tuple<Eigen::ArrayXXf>() }, */
/*     { DataType::ArrayF, tuple<TimeSeriesXf>() }, */
/*     { DataType::ArrayI, tuple<Eigen::ArrayXXi>() }, */
/*     { DataType::ArrayI, tuple<TimeSeriesI>() }, */
/* }; */


/* auto NodeSchema = { */

//arithmetic
    /* {NodeType::Add, { */
    /*     {"ExecType", ExecType::Reducer}, */
    /*     {"Signature", BinaryF}, */
    /*                 }, */
    /* }, */
    /* {NodeType::Sub, { */
    /*     {"Arity", 2}, */
    /*     {"ExecType", ExecType::Applier}, */
    /*     {"Signature", BinaryFF}, */
    /*                 }, */
    /* }, */
    /* {NodeType::Mul, { */
    /*     {"Arity", 2}, */
    /*     {"ExecType", ExecType::Reducer}, */
    /*     {"Signature", BinaryFF}, */
    /*                 }, */
    /* }, */
    /* {NodeType::Div, { */
    /*     {"Arity", 2}, */
    /*     {"ExecType", ExecType::Applier}, */
    /*     {"Signature", BinaryFF}, */
    /*                 }, */
    /* }, */
    /* {NodeType::Aq, { */
    /*     {"Arity", 2}, */
    /*     {"ExecType", ExecType::Applier}, */
    /*     {"Signature", BinaryFF}, */
    /*                }, */
    /* }, */
    /* {NodeType::Abs, { */
    /*     {"Arity", 1}, */
    /*     {"ExecType", ExecType::Transformer}, */
    /*     {"Signature", UnaryFF}, */
    /*                 }, */
    /* }, */
    /* {NodeType::Cbrt, { */
    /*     {"Arity", 1}, */
    /*     {"ExecType", ExecType::Transformer}, */
    /*     {"Signature", UnaryFF}, */
    /*                  }, */
    /* }, */
    /* {NodeType::Ceil, { */
    /*     {"Arity", 1}, */
    /*         {"ExecType", ExecType::Transformer}, */
    /*         {"Signature", UnaryIF}, */
    /*                  }, */
    /* }, */
    /* {NodeType::Floor, { */
    /*     {"Arity", 1}, */
    /*         {"ExecType", ExecType::Transformer}, */
    /*         {"Signature", UnaryIF}, */
    /*                   }, */
    /* }, */
    /* {NodeType::Exp, { */
    /*     {"Arity", 1}, */
    /*     {"ExecType", ExecType::Transformer}, */
    /*     {"Signature", UnaryFF}, */
    /*                 }, */
    /* }, */
    /* {NodeType::Log, { */
    /*     {"Arity", 1}, */
    /*     {"ExecType", ExecType::Transformer}, */
    /*     {"Signature", UnaryFF}, */
    /*                 }, */
    /* }, */
    /* {NodeType::Logabs, { */
    /*     {"Arity", 1}, */
    /*         {"ExecType", ExecType::Transformer}, */
    /*         {"Signature", UnaryFF}, */
    /*                    }, */
    /* }, */
    /* {NodeType::Log1p, { */
    /*     {"Arity", 1}, */
    /*         {"ExecType", ExecType::Transformer}, */
    /*         {"Signature", UnaryFF}, */
    /*                   }, */
    /* }, */
    /* {NodeType::Sqrt, { */
    /*     {"Arity", 1}, */
    /*         {"ExecType", ExecType::Transformer}, */
    /*         {"Signature", UnaryFF}, */
    /*                  }, */
    /* }, */
    /* {NodeType::Sqrtabs, { */
    /*     {"Arity", 1}, */
    /*         {"ExecType", ExecType::Transformer}, */
    /*         {"Signature", UnaryFF}, */
    /*                     }, */
    /* }, */
    /* {NodeType::Square, { */
    /*     {"Arity", 1}, */
    /*         {"ExecType", ExecType::Transformer}, */
    /*         {"Signature", UnaryFF}, */
    /*                    }, */
    /* }, */
    /* /1* {NodeType::Pow, { *1/ */
    /* /1*     {"Arity", 1}, *1/ */
    /* /1*     {"ExecType", ExecType::Transformer}, *1/ */
    /* /1*     {"Signature", UnaryFF}, *1/ */
    /* /1*                 }, *1/ */
    /* /1* }, *1/ */
/* // trigonometry */
    /* {NodeType::Acos, { */
    /*     {"Arity", 1}, */
    /*         {"ExecType", ExecType::Transformer}, */
    /*         {"Signature", UnaryFF}, */
    /*                  }, */
    /* }, */
    /* {NodeType::Asin, { */
    /*     {"Arity", 1}, */
    /*         {"ExecType", ExecType::Transformer}, */
    /*         {"Signature", UnaryFF}, */
    /*                  }, */
    /* }, */
    /* {NodeType::Atan, { */
    /*     {"Arity", 1}, */
    /*         {"ExecType", ExecType::Transformer}, */
    /*         {"Signature", UnaryFF}, */
    /*                  }, */
    /* }, */
    /* {NodeType::Cos, { */
    /*     {"Arity", 1}, */
    /*     {"ExecType", ExecType::Transformer}, */
    /*     {"Signature", UnaryFF}, */
    /*                 }, */
    /* }, */
    /* {NodeType::Cosh, { */
    /*     {"Arity", 1}, */
    /*         {"ExecType", ExecType::Transformer}, */
    /*         {"Signature", UnaryFF}, */
    /*                  }, */
    /* }, */
    /* {NodeType::Sin, { */
    /*     {"Arity", 1}, */
    /*     {"ExecType", ExecType::Transformer}, */
    /*     {"Signature", UnaryFF}, */
    /*                 }, */
    /* }, */
    /* {NodeType::Sinh, { */
    /*     {"Arity", 1}, */
    /*         {"ExecType", ExecType::Transformer}, */
    /*         {"Signature", UnaryFF}, */
    /*                  }, */
    /* }, */
    /* {NodeType::Tan, { */
    /*     {"Arity", 1}, */
    /*     {"ExecType", ExecType::Transformer}, */
    /*     {"Signature", UnaryFF}, */
    /*                 }, */
    /* }, */
    /* {NodeType::Tanh, { */
    /*     {"Arity", 1}, */
    /*         {"ExecType", ExecType::Transformer}, */
    /*         {"Signature", UnaryFF}, */
    /*                  }, */
    /* }, */

/* // logic; not sure these will make it in */
    /* {NodeType::And, { */
    /*     {"Arity", 1}, */
    /*     {"ExecType", ExecType::Reducer}, */
    /*     {"Signature", BinaryBB}, */
    /*                 }, */
    /* }, */
    /* {NodeType::Or, { */
    /*     {"ExecType", ExecType::Reducer}, */
    /*     {"Signature", BinaryBB}, */
    /*                }, */
    /* }, */
    /* {NodeType::Not, { */
    /*     {"ExecType", ExecType::Transformer}, */
    /*     {"Signature", UnaryBB}, */
    /*                 }, */
    /* }, */
    /* {NodeType::Xor, { */
    /*     {"ExecType", ExecType::Applier}, */
    /*     {"Signature", BinaryBB}, */
    /*                 }, */
    /* }, */

/* // decision (same) */
    /* {NodeType::Equals, { */
    /*         {"ExecType", ExecType::Reducer}, */
    /*         {"Signature", BinaryBF}, */
    /*                    }, */
    /* }, */
    /* {NodeType::LessThan, { */
    /*             {"ExecType", ExecType::Applier}, */
    /*             {"Signature", BinaryBF}, */
    /*                      }, */
    /* }, */
    /* {NodeType::GreaterThan, { */
    /*             {"ExecType", ExecType::Applier}, */
    /*             {"Signature", BinaryBF}, */
    /*                         }, */
    /* }, */
    /* {NodeType::Leq, { */
    /*     {"ExecType", ExecType::Applier}, */
    /*     {"Signature", BinaryBF}, */
    /*                 }, */
    /* }, */
    /* {NodeType::Geq, { */
    /*     {"ExecType", ExecType::Applier}, */
    /*     {"Signature", BinaryBF}, */
    /*                 }, */
    /* }, */

/* // summary stats */
    /* {NodeType::Min, { */
    /*     {"ExecType", ExecType::Reducer}, */
    /*     {"Signature", ReduceAll}, */
    /*                 }, */
    /* }, */
    /* {NodeType::Max, { */
    /*     {"ExecType", ExecType::Reducer}, */
    /*     {"Signature", ReduceAll}, */
    /*                 }, */
    /* }, */
    /* {NodeType::Mean, { */
    /*         {"ExecType", ExecType::Applier}, */
    /*         {"Signature", ReduceFF}, */
    /*                  }, */
    /* }, */
    /* {NodeType::Median, { */
    /*         {"ExecType", ExecType::Applier}, */
    /*         {"Signature", ReduceFF}, */
    /*                    }, */
    /* }, */
    /* {NodeType::Count, { */
    /*         {"ExecType", ExecType::Applier}, */
    /*         {"Signature", ReduceIA}, */
    /*                   }, */
    /* }, */
    /* {NodeType::Sum, { */
    /*     {"ExecType", ExecType::Applier}, */
    /*     {"Signature", ReduceAll}, */
    /*                 }, */
    /* }, */

/* // timing masks */
    /* {NodeType::Before, { */
    /*         {"ExecType", ExecType::Mapper}, */
    /*         {"Signature", BinaryFF}, */
    /*                    }, */
    /* }, */
    /* {NodeType::After, { */
    /*         {"ExecType", ExecType::Mapper}, */
    /*         {"Signature", BinaryFF}, */
    /*                   }, */
    /* }, */
    /* {NodeType::During, { */
    /*         {"ExecType", ExecType::Mapper}, */
    /*         {"Signature", BinaryFF}, */
    /*                    }, */
    /* }, */

/* //split */
    /* {NodeType::SplitBest, { */
    /*             {"ExecType", ExecType::Mapper}, */
    /*             {"Signature", {DataType::ArrayF, tuple<Eigen::ArrayXf,Eigen::ArrayXf>}, */
    /*                       }, */
    /* }, */
    /* {NodeType::SplitOn, { */
    /*         {"ExecType", ExecType::Mapper}, */
    /*         {"Signature", {DataType::ArrayF, {DataType::ArrayF, */
    /*                                         DataType::ArrayF, */
    /*                                         DataType::ArrayF} */
    /*                     }, */
    /*                     }, */
    /* }, */

/* // leaves */
    /* {NodeType::Constant, { */
    /*             {"ExecType", ExecType::Mapper}, */
    /*             {"Signature", BinaryFF}, */
    /*                      }, */
    /* }, */
    /* {NodeType::Variable, { */
    /*             {"ExecType", ExecType::Mapper}, */
    /*             {"Signature", BinaryFF}, */
    /*                      }, */
    /* }, */

/* // custom */
    /* {NodeType::CustomOp, { */
    /*             {"ExecType", ExecType::Mapper}, */
    /*             {"Signature", BinaryFF}, */
    /*                      }, */
    /* }, */
    /* {NodeType::CustomSplit { */
    /*             {"ExecType", ExecType::Mapper}, */
    /*             {"Signature", BinaryFF}, */
    /*                        }, */
    /* }, */
/* }; */
}

#endif
