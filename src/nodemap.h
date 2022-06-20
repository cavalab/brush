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
    // leaves
    _LEAF_              = 1UL << 1UL,
    Constant            = 1UL << 2UL,
    Terminal            = 1UL << 3UL,
    // Unary
    _UNARY_             = 1UL << 4UL,
    Abs                 = 1UL << 5UL,
    Acos                = 1UL << 6UL,
    Asin                = 1UL << 7UL,
    Atan                = 1UL << 8UL,
    Cos                 = 1UL << 9UL,
    Cosh                = 1UL << 10UL,
    Sin                 = 1UL << 11UL,
    Sinh                = 1UL << 12UL,
    Tan                 = 1UL << 13UL,
    Tanh                = 1UL << 14UL,
    /* Cbrt                = 1UL 15 14UL, */
    Ceil                = 1UL << 16UL,
    Floor               = 1UL << 17UL,
    Exp                 = 1UL << 18UL,
    Log                 = 1UL << 19UL,
    Logabs              = 1UL << 20UL,
    Log1p               = 1UL << 21UL,
    Sqrt                = 1UL << 22UL,
    Sqrtabs             = 1UL << 23UL,
    Square              = 1UL << 24UL,
    // timing masks
    Before              = 1UL << 25UL,
    After               = 1UL << 26UL,
    During              = 1UL << 27UL,
    CustomUnaryOp       = 1UL << 28UL,
    // Reducers
    _REDUCER_           = 1UL << 29UL,
    Min                 = 1UL << 30UL,
    Max                 = 1UL << 31UL,
    Mean                = 1UL << 32UL,
    Median              = 1UL << 33UL,
    Count               = 1UL << 34UL,
    Sum                 = 1UL << 35UL,
    // Binary
    _BINARY_            = 1UL << 36UL,
    Add                 = 1UL << 37UL,
    Sub                 = 1UL << 38UL,
    Mul                 = 1UL << 39UL,
    Div                 = 1UL << 40UL,
    Aq                  = 1UL << 41UL,
    Pow                 = 1UL << 42UL,
    And                 = 1UL << 43UL,
    Or                  = 1UL << 44UL,
    Not                 = 1UL << 45UL,
    Xor                 = 1UL << 46UL,
    // these ones change type
    _COMPARATOR_
    Equals              = 1UL << 47UL,
    LessThan            = 1UL << 48UL,
    GreaterThan         = 1UL << 49UL,
    Leq                 = 1UL << 50UL,
    Geq                 = 1UL << 51UL,
    CustomBinaryOp      = 1UL << 52UL,
    //split
    _SPLITTER_          = 1UL << 53UL,
    SplitBest           = 1UL << 54UL,
    SplitOn             = 1UL << 55UL,
    CustomSplit         = 1UL << 56UL,
    // Special Op
    // _END_            = 1UL << 5UL
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

}

extern std::map<std::string, NodeType> NodeNameType;
extern std::map<NodeType,std::string> NodeTypeName;
extern json BinaryFFtoF;
extern json UnaryFtoF;
extern json NodeSchema;

#endif
