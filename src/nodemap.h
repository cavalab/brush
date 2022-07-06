/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/
#ifndef NODEMAP_H
#define NODEMAP_H
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




enum class NodeType : uint64_t {
    // leaves
    Constant            = 1UL << 0UL,
    Terminal            = 1UL << 1UL,
    // Unary
    Abs                 = 1UL << 2UL,
    Acos                = 1UL << 3UL,
    Asin                = 1UL << 4UL,
    Atan                = 1UL << 5UL,
    Cos                 = 1UL << 6UL,
    Cosh                = 1UL << 7UL,
    Sin                 = 1UL << 8UL,
    Sinh                = 1UL << 9UL,
    Tan                 = 1UL << 10UL,
    Tanh                = 1UL << 11UL,
    Ceil                = 1UL << 13UL,
    Floor               = 1UL << 14UL,
    Exp                 = 1UL << 15UL,
    Log                 = 1UL << 16UL,
    Logabs              = 1UL << 17UL,
    Log1p               = 1UL << 18UL,
    Sqrt                = 1UL << 19UL,
    Sqrtabs             = 1UL << 20UL,
    Square              = 1UL << 21UL,
    Not                 = 1UL << 22UL,
    // timing masks
    Before              = 1UL << 23UL,
    After               = 1UL << 24UL,
    During              = 1UL << 25UL,
    CustomUnaryOp       = 1UL << 26UL,
    // Reducers
    Min                 = 1UL << 28UL,
    Max                 = 1UL << 29UL,
    Mean                = 1UL << 30UL,
    Median              = 1UL << 31UL,
    Count               = 1UL << 32UL,
    Sum                 = 1UL << 33UL,
    // Binary
    Add                 = 1UL << 35UL,
    Sub                 = 1UL << 36UL,
    Mul                 = 1UL << 37UL,
    Div                 = 1UL << 38UL,
    Pow                 = 1UL << 39UL,
    And                 = 1UL << 40UL,
    Or                  = 1UL << 41UL,
    Xor                 = 1UL << 42UL,
    // these ones change type
    Equals              = 1UL << 43UL,
    LessThan            = 1UL << 44UL,
    GreaterThan         = 1UL << 45UL,
    Leq                 = 1UL << 46UL,
    Geq                 = 1UL << 47UL,
    CustomBinaryOp      = 1UL << 48UL,
    //split
    SplitBest           = 1UL << 50UL,
    SplitOn             = 1UL << 51UL,
    CustomSplit         = 1UL << 52UL
};


using UnderlyingNodeType = std::underlying_type_t<NodeType>;
struct NodeTypes {
    // magic number keeping track of the number of different node types
    static constexpr size_t Count = 10;

    // returns the index of the given type in the NodeType enum
    static auto GetIndex(NodeType type) -> size_t
    {
        return std::bitset<Count>(static_cast<std::underlying_type_t<NodeType>>(type)).count();
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
extern json BinaryFFtoF;
extern json UnaryFtoF;
extern json NodeSchema;

////////////////////////////////////////////////////////////////////////////////
// Exec types
enum class ExecType: uint32_t {
    Unary, // no weights, just a unary call to apply
    Binary, // no weights, just a binary call to apply
    /* Transformer, // maps child nodes to output via transform call. */
    Reducer, // maps child nodes to output via reduction call.
    /* Applier, // maps child nodes to output via function call. */
    Comparator, // maps inputs to a decision / comparison boolean-derived type
    BestSplitter, // splits data at the best spot, and returns the output of the children on each split.
    ArgSplitter, // splits data on first arg, and returns the output of the children on each split.
    Terminal,    // returns a data element.
};

// https://stackoverflow.com/questions/25958259/how-do-i-find-out-if-a-tuple-contains-a-type
// https://stackoverflow.com/questions/34111060/c-check-if-the-template-type-is-one-of-the-variadic-template-types
template <NodeType T, NodeType... Ts> struct is_one_of
{
    static constexpr bool value = ((T == Ts) || ...);
    /* { */ 
    /*     return true; */ 
    /* } */ 
    /* else */
    /*     return false; */
};

template<NodeType T, NodeType... Ts> 
static constexpr bool is_one_of_v = is_one_of<T, Ts...>::value;


enum class SigType : uint32_t {
    Terminal,
    ArrayFtoArrayF,
    ArrayItoArrayI,
    ArrayBtoArrayB,
    ArrayFtoArrayB,
    ArrayFtoArrayI,
    ArrayItoArrayB,
    ArrayFArrayFtoArrayF,
    ArrayIArrayItoArrayI,
    ArrayBArrayBtoArrayB,
    ArrayFArrayFtoArrayB,
    ArrayFArrayFtoArrayI,
    ArrayIArrayItoArrayB,
    MatrixFtoMatrixF,
    MatrixItoMatrixI,
    MatrixBtoMatrixB,
    MatrixFtoMatrixB,
    MatrixFtoMatrixI,
    MatrixItoMatrixB,
    MatrixItoMatrixF,
    MatrixFMatrixFtoMatrixF,
    MatrixIMatrixItoMatrixI,
    MatrixBMatrixBtoMatrixB,
    MatrixFMatrixFtoMatrixB,
    MatrixFMatrixFtoMatrixI,
    MatrixIMatrixItoMatrixB,
    MatrixFtoArrayF,
    MatrixItoArrayI,
    MatrixBtoArrayB,
    MatrixFtoArrayB,
    MatrixFtoArrayI,
    MatrixItoArrayB,
    MatrixItoArrayF,
    MatrixBtoArrayF,
    ArrayFArrayFArrayFtoArrayF
};
template<SigType S = SigType::ArrayFtoArrayF> struct Signature{ 
    using RetType = ArrayXf; 
    using ArgTypes = std::array<ArrayXf,1>; 
    static constexpr size_t ArgCount=1;
};
template<> struct Signature<SigType::ArrayItoArrayI>{ 
    using RetType = ArrayXi; 
    using ArgTypes = std::array<ArrayXi,1>;
    static constexpr size_t ArgCount=1;
};
template<> struct Signature<SigType::ArrayBtoArrayB>{ 
    using RetType = ArrayXb; 
    using ArgTypes = std::array<ArrayXb,1>;
    static constexpr size_t ArgCount=1;
};

// Signatures gives a set of SigType for each node

// Store the signatures that each Node can handle
//
template<NodeType N, typename T = void> struct Signatures; 

/* template<NodeType N> */
/* struct Signatures<N> */
/* { */ 
/*     static constexpr array<SigType,1> value = { */ 
/*         SigType::Terminal */
/*     }; */ 
/* }; */ 
template<NodeType N>
struct Signatures<N, enable_if_t<is_one_of_v<N, NodeType::Constant, NodeType::Terminal>>>
{ 
    static constexpr array<SigType,1> value = { 
        SigType::Terminal
    }; 
}; 
template<NodeType N>
struct Signatures<N, enable_if_t<is_one_of_v<N,
    NodeType::Add,
    NodeType::Sub,
    NodeType::Mul,
    NodeType::Div,
    /* NodeType::Aq, */
    NodeType::Pow
    >>>{ 
        static constexpr array<SigType,2> value = { 
            SigType::ArrayFArrayFtoArrayF, 
            SigType::MatrixFMatrixFtoMatrixF
        }; 
    }; 

template<NodeType N>
struct Signatures<N, enable_if_t<is_one_of_v<N,
    NodeType::And,
    NodeType::Or,
    NodeType::Xor
    >>>{ 
        /* static constexpr ExecType type = ExecType::Binary; */ 
        static constexpr array<SigType,2> value = { 
            SigType::ArrayBArrayBtoArrayB, 
            SigType::MatrixBMatrixBtoMatrixB
        }; 
    }; 

template<> 
struct Signatures<NodeType::Not> { 
    static constexpr array<SigType, 2> value = {
            SigType::ArrayBtoArrayB, 
            SigType::MatrixBtoMatrixB
    };
};

template<NodeType N> 
struct Signatures<N, enable_if_t<is_one_of_v<N,
    NodeType::Abs,
    NodeType::Acos,
    NodeType::Asin,
    NodeType::Atan,
    NodeType::Cos,
    NodeType::Cosh,
    NodeType::Sin,
    NodeType::Sinh,
    NodeType::Tan,
    NodeType::Tanh,
    NodeType::Ceil,
    NodeType::Floor,
    NodeType::Exp,
    NodeType::Log,
    NodeType::Logabs,
    NodeType::Log1p,
    NodeType::Sqrt,
    NodeType::Sqrtabs,
    NodeType::Square 
    >>>{ 
        static constexpr array<SigType,2> value = { 
            SigType::ArrayFtoArrayF, 
            SigType::MatrixFtoMatrixF 
        };
    };

template<NodeType N>
struct Signatures<N, enable_if_t<is_one_of_v<N, 
    NodeType::Before,
    NodeType::After,
    NodeType::During,
    NodeType::CustomUnaryOp
    >>>{ 
        //TODO: Fix
        static constexpr array<SigType,3> value = { 
            SigType::MatrixFtoArrayF, 
            SigType::MatrixItoArrayI,
            SigType::MatrixBtoArrayB,
        };
    }; 
template<NodeType N>
struct Signatures<N, enable_if_t<is_one_of_v<N, 
    NodeType::Min, 
    NodeType::Max
    >>>{ 
        static constexpr array<SigType,3> value = { 
            SigType::MatrixFtoArrayF, 
            SigType::MatrixItoArrayI,
            SigType::MatrixBtoArrayB,
        };
    }; 

template<NodeType N>
struct Signatures<N, enable_if_t<is_one_of_v<N, 
    NodeType::Mean,
    NodeType::Median
    >>>{ 
        static constexpr array<SigType,1> value = { 
            SigType::MatrixFtoArrayF
        };
    }; 

template<NodeType N>
struct Signatures<N, enable_if_t<is_one_of_v<N, 
    NodeType::Count,
    NodeType::Sum
    >>>{ 
        static constexpr array<SigType,3> value = { 
            SigType::MatrixFtoArrayF, 
            SigType::MatrixItoArrayF, 
            SigType::MatrixBtoArrayF, 
        };
    }; 

template<NodeType N>
struct Signatures<N, enable_if_t<is_one_of_v<N, 
    NodeType::Equals,
    NodeType::LessThan,
    NodeType::GreaterThan,
    NodeType::Leq,
    NodeType::Geq,
    NodeType::CustomBinaryOp
    >>>{ 
        static constexpr array<SigType,4> value = { 
            SigType::ArrayFArrayFtoArrayB, 
            SigType::ArrayIArrayItoArrayB, 
            SigType::MatrixFMatrixFtoMatrixB, 
            SigType::MatrixFMatrixFtoMatrixB, 
        };
    }; 

template<NodeType N>
struct Signatures<N, enable_if_t<is_one_of_v<N, NodeType::SplitBest, NodeType::CustomSplit>>>
    { 
        static constexpr array<SigType,3> value = { 
            SigType::ArrayFArrayFtoArrayF, 
            SigType::ArrayIArrayItoArrayI, 
            SigType::ArrayBArrayBtoArrayB, 
            // TODO
            /* SigType::MatrixFMatrixFtoMatrixF, */
            /* SigType::MatrixFMatrixFtoMatrixF */
        }; 
    }; 
template<>
struct Signatures<NodeType::SplitOn>{ 
        static constexpr array<SigType,1> value = { 
            SigType::ArrayFArrayFArrayFtoArrayF
            /* SigType::ArrayFArrayIArrayItoArrayI, */ 
            /* SigType::ArrayFArrayBArrayBtoArrayB, */ 
            /* SigType::ArrayIArrayFArrayFtoArrayF, */ 
            /* SigType::ArrayIArrayIArrayItoArrayI, */ 
            /* SigType::ArrayIArrayBArrayBtoArrayB, */ 
            /* SigType::ArrayBArrayFArrayFtoArrayF, */ 
            /* SigType::ArrayBArrayIArrayItoArrayI, */ 
            /* SigType::ArrayBArrayBArrayBtoArrayB, */ 
        }; 
    }; 
} // namespace Brush
#endif
