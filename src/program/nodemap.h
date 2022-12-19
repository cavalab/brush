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
using Brush::Util::is_tuple;

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
    Not                 = 1UL << 19UL,
    // timing masks
    Before              = 1UL << 20UL,
    After               = 1UL << 21UL,
    During              = 1UL << 22UL,
    // Reducers
    Min                 = 1UL << 23UL,
    Max                 = 1UL << 24UL,
    Mean                = 1UL << 25UL,
    Median              = 1UL << 26UL,
    Count               = 1UL << 27UL,
    Sum                 = 1UL << 28UL,
    // Binary
    Add                 = 1UL << 29UL,
    Sub                 = 1UL << 30UL,
    Mul                 = 1UL << 31UL,
    Div                 = 1UL << 32UL,
    Pow                 = 1UL << 33UL,
    And                 = 1UL << 34UL,
    Or                  = 1UL << 35UL,
    Xor                 = 1UL << 36UL,
    //split
    SplitBest           = 1UL << 37UL,
    SplitOn             = 1UL << 38UL,
    // these ones change type
    /* Equals              = 1UL << 39UL, */
    /* LessThan            = 1UL << 40UL, */
    /* GreaterThan         = 1UL << 41UL, */
    /* Leq                 = 1UL << 42UL, */
    /* Geq                 = 1UL << 43UL, */
    // leaves
    Constant            = 1UL << 39UL,
    Terminal            = 1UL << 40UL,
    // custom
    CustomUnaryOp       = 1UL << 41UL,
    CustomBinaryOp      = 1UL << 42UL,
    CustomSplit         = 1UL << 43UL
};


using UnderlyingNodeType = std::underlying_type_t<NodeType>;
struct NodeTypes {
    // magic number keeping track of the number of different node types
    static constexpr size_t Count = 41;
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

} // namespace Brush
// format overload for Nodes
template <> struct fmt::formatter<Brush::NodeType>: formatter<string_view> {
  // parse is inherited from formatter<string_view>.
  template <typename FormatContext>
  auto format(Brush::NodeType x, FormatContext& ctx) const {
    return formatter<string_view>::format(Brush::NodeTypeName.at(x), ctx);
  }
};
#include "signatures.h"
#endif
