/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3

This section adapted heavily from Operon: https://github.com/heal-research/operon/
*/
#ifndef NODEMAP_H
#define NODEMAP_H
// external includes
#include <bitset>
#include <type_traits>
#include <utility>
//internal includes
#include "../init.h"
#include "../data/data.h"
#include "../util/utils.h"
#include "../util/rnd.h"
using std::vector;
using namespace Brush;
/* using Brush::NodeType; */ 
/* using Brush::ExecType; */ 
using std::tuple;
using std::array;
using Brush::DataType; 
using Brush::Data::TimeSeriesb; 
using Brush::Data::TimeSeriesi; 
using Brush::Data::TimeSeriesf; 

namespace Brush {

enum class NodeType : uint64_t {
    // Unary
    Abs                 = 1UL << 0UL,
    Acos                = 1UL << 1UL,
    Asin                = 1UL << 2UL,
    Atan                = 1UL << 3UL,
    Cos                 = 1UL << 4UL,
    Cosh                = 1UL << 5UL,
    Sin                 = 1UL << 6UL,
    Sinh                = 1UL << 7UL,
    Tan                 = 1UL << 8UL,
    Tanh                = 1UL << 9UL,
    Ceil                = 1UL << 10UL,
    Floor               = 1UL << 11UL,
    Exp                 = 1UL << 12UL,
    Log                 = 1UL << 13UL,
    Logabs              = 1UL << 14UL,
    Log1p               = 1UL << 15UL,
    Sqrt                = 1UL << 16UL,
    Sqrtabs             = 1UL << 17UL,
    Square              = 1UL << 18UL,
    Logistic            = 1UL << 19UL,
    // timing masks
    Before              = 1UL << 20UL,
    After               = 1UL << 21UL,
    During              = 1UL << 22UL,
    // Reducers
    Min                 = 1UL << 23UL,
    Max                 = 1UL << 24UL,
    Mean                = 1UL << 25UL,
    Median              = 1UL << 26UL,
    Sum                 = 1UL << 27UL,
    Prod                = 1UL << 28UL,
    // Transformers 
    Softmax             = 1UL << 29UL,
    // Binary
    Add                 = 1UL << 30UL,
    Sub                 = 1UL << 31UL,
    Mul                 = 1UL << 32UL,
    Div                 = 1UL << 33UL,
    Pow                 = 1UL << 34UL,
    //split
    SplitBest           = 1UL << 35UL,
    SplitOn             = 1UL << 36UL,
    // these ones change type
    /* Equals              = 1UL << 39UL, */
    /* LessThan            = 1UL << 40UL, */
    /* GreaterThan         = 1UL << 41UL, */
    /* Leq                 = 1UL << 42UL, */
    /* Geq                 = 1UL << 43UL, */
    // leaves
    Constant            = 1UL << 37UL,
    Terminal            = 1UL << 38UL,
    ArgMax              = 1UL << 39UL,
    Count               = 1UL << 40UL,
    // custom
    CustomUnaryOp       = 1UL << 41UL,
    CustomBinaryOp      = 1UL << 42UL,
    CustomSplit         = 1UL << 43UL
    // boolean
    // And                 = 1UL << 37UL,
    // Or                  = 1UL << 38UL,
    // Xor                 = 1UL << 39UL,
    // Not                 = 1UL << 19UL,
};


using UnderlyingNodeType = std::underlying_type_t<NodeType>;
struct NodeTypes {
    // magic number keeping track of the number of different node types
    static constexpr size_t Count = 39;
    static constexpr size_t OpCount = Count-2;

    // returns the index of the given type in the NodeType enum
    static auto GetIndex(NodeType type) -> size_t
    {
        return std::bitset<Count>(static_cast<UnderlyingNodeType>(type)).count();
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



extern std::map<std::string, NodeType> NodeNameType;
extern std::map<NodeType,std::string> NodeTypeName;

// map NodeType values to JSON as strings
NLOHMANN_JSON_SERIALIZE_ENUM( NodeType, {
    //arithmetic
    {NodeType::Add,"Add" },
    {NodeType::Sub,"Sub" },
    {NodeType::Mul,"Mul" },
    {NodeType::Div,"Div" },
    /* {NodeType::Aq,"Aq" }, */
    {NodeType::Abs,"Abs" },
    {NodeType::Acos,"Acos" },
    {NodeType::Asin,"Asin" },
    {NodeType::Atan,"Atan" },
    {NodeType::Cos,"Cos" },
    {NodeType::Cosh,"Cosh" },
    {NodeType::Sin,"Sin" },
    {NodeType::Sinh,"Sinh" },
    {NodeType::Tan,"Tan" },
    {NodeType::Tanh,"Tanh" },
    {NodeType::Ceil,"Ceil" },
    {NodeType::Floor,"Floor" },
    {NodeType::Exp,"Exp" },
    {NodeType::Log,"Log" },
    {NodeType::Logabs,"Logabs" },
    {NodeType::Log1p,"Log1p" },
    {NodeType::Sqrt,"Sqrt" },
    {NodeType::Sqrtabs,"Sqrtabs" },
    {NodeType::Square,"Square" },
    {NodeType::Pow,"Pow" },
    {NodeType::Logistic,"Logistic" },

    // logic; not sure these will make it in
    // {NodeType::And,"And" },
    // {NodeType::Or,"Or" },
    // {NodeType::Not,"Not" },
    // {NodeType::Xor,"Xor" },

    // decision (same)
    /* {NodeType::Equals,"Equals" }, */
    /* {NodeType::LessThan,"LessThan" }, */
    /* {NodeType::GreaterThan,"GreaterThan" }, */
    /* {NodeType::Leq,"Leq" }, */
    /* {NodeType::Geq,"Geq" }, */

    // reductions
    {NodeType::Min,"Min" },
    {NodeType::Max,"Max" },
    {NodeType::Mean,"Mean" },
    {NodeType::Median,"Median" },
    {NodeType::Count,"Count" },
    {NodeType::Sum,"Sum" },
    {NodeType::Prod,"Prod" },
    {NodeType::ArgMax,"ArgMax" },

    // transforms
    {NodeType::Softmax,"Softmax" },

    // timing masks
    {NodeType::Before,"Before" },
    {NodeType::After,"After" },
    {NodeType::During,"During" },

    //split
    {NodeType::SplitBest,"SplitBest" },
    {NodeType::SplitOn,"SplitOn" },

    // leaves
    {NodeType::Constant,"Constant" },
    {NodeType::Terminal,"Terminal" },

    // custom
    {NodeType::CustomUnaryOp,"CustomUnaryOp" },
    {NodeType::CustomBinaryOp,"CustomBinaryOp" },
    {NodeType::CustomSplit,"CustomSplit" },
})   

} // namespace Brush

// format overload for NodeTypes
template <> struct fmt::formatter<Brush::NodeType>: formatter<string_view> {
  // parse is inherited from formatter<string_view>.
  template <typename FormatContext>
  auto format(Brush::NodeType x, FormatContext& ctx) const {
    return formatter<string_view>::format(Brush::NodeTypeName.at(x), ctx);
  }
};
////////////////////////////////////////////////////////////////////////////////
template <NodeType T, NodeType... Ts> 
struct is_in
{
    static constexpr bool value = ((T == Ts) || ...);
};

template<NodeType T, NodeType... Ts> 
static constexpr bool is_in_v = is_in<T, Ts...>::value;

using NT = NodeType;
// NodeType concepts
template<NT nt>
static constexpr bool UnaryOp = is_in_v<nt, 
    NT::Abs,
    NT::Acos,
    NT::Asin,
    NT::Atan,
    NT::Cos,
    NT::Cosh,
    NT::Sin,
    NT::Sinh,
    NT::Tan,
    NT::Tanh,
    NT::Ceil,
    NT::Floor,
    NT::Exp,
    NT::Log,
    NT::Logabs,
    NT::Log1p,
    NT::Sqrt,
    NT::Sqrtabs,
    NT::Square,
    NT::Logistic
>;

template<NT nt>
static constexpr bool BinaryOp = is_in_v<nt, 
    NT::Add,
    NT::Sub,
    NT::Mul,
    NT::Div,
    NT::Pow
>;
template<NT nt>
static constexpr bool AssociativeBinaryOp = is_in_v<nt, 
    NT::Add,
    NT::Mul
>;

template<NT nt>
static constexpr bool NaryOp = is_in_v<nt, 
    NT::Min,
    NT::Max,
    NT::Mean,
    NT::Median,
    NT::Sum,
    NT::Prod,
    NT::Softmax
>;
// // TODO: make this work 
// template<typename NT, size_t ArgCount>
// concept Transformer = requires(NT n, size_t ArgCount) 
// {
//     UnaryOp<n> && ArgCount > 1;
// }

// template<typename NT, size_t ArgCount>
// concept Reducer = requires(NT n, size_t ArgCount) 
// {
//     BinaryOp<n> && ArgCount > 2;
// }

   
   
#include "signatures.h"
#endif