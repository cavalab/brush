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
};

template<NodeType T, NodeType... Ts> 
static constexpr bool is_one_of_v = is_one_of<T, Ts...>::value;


// Signatures gives a set of SigType for each node
template<typename First, typename ... Next>
struct all_same{
    static constexpr bool value {(std::is_same_v<First,Next> && ...)};
    /* static constexpr bool value = true; */
};
template<typename R, typename... Args>
struct SigBase  
{
    using RetType = R;
    static constexpr std::size_t ArgCount = sizeof...(Args);

    using FirstArg = std::tuple_element_t<0, std::tuple<Args...>>;

    /// ArgTypes is a std::array if the types are shared, otherwise it is a tuple. 
    // (using std::array allows begin() and end() ops like transform to be applied)
    using ArgTypes = conditional_t<(std::is_same_v<FirstArg,Args> && ...),
                                   std::array<FirstArg,ArgCount>,
                                   std::tuple<Args...>
                                  >;
    template <std::size_t N>
    using NthType = conditional_t<!is_tuple<ArgTypes>::value, 
                                  FirstArg,
                                  typename std::tuple_element<N, ArgTypes>::type
                                 >;
    // currently unused
    using Function = std::function<R(Args...)>;


    template<size_t... Is>
    static constexpr auto get_arg_types(std::index_sequence<Is...>) 
    {
        return vector<DataType>{(DataTypeEnum<NthType<Is>>::value) ...};
    }

    static constexpr auto get_arg_types() {
        return get_arg_types(make_index_sequence<ArgCount>());
    } 
    static constexpr auto get_args_type() {
        if constexpr (!is_tuple<ArgTypes>::value)
            return "Array";
        else
            return "Tuple";
    };

    // template<typename T>
    // static constexpr bool contains() { return is_one_of_v<T, Args...>; }       // This is ill-formed on clang; returns an error

    static constexpr std::size_t hash_args(){ return typeid(ArgTypes).hash_code();}

    static constexpr std::size_t hash(){ return typeid(tuple<R,Args...>).hash_code();};
};
/// specialization for terminals
template<typename R>
struct SigBase<R>
{
    using RetType = R;
    using ArgTypes = void;
    static constexpr std::size_t ArgCount = 0;
    static constexpr auto get_arg_types() { return vector<DataType>{}; } 
    static constexpr auto get_args_type() { return "None"; } 
    static constexpr std::size_t hash(){ return typeid(R).hash_code(); };
};

template<typename T> struct Signature;
template<typename R, typename... Args>
struct Signature<R(Args...)> : SigBase<R, Args...>
{
    using base = SigBase<R, Args...>;
    using RetType = typename base::RetType;
    using ArgTypes = typename base::ArgTypes;
    static constexpr auto ArgCount = base::ArgCount;
    /* using Function = base::Function; */
};
// Store the signatures that each Node can handle
//
template<NodeType N, typename T = void> struct Signatures; 

template<NodeType N>
struct Signatures<N, enable_if_t<is_one_of_v<N, NodeType::Constant, NodeType::Terminal>>>{ 
    using type = std::tuple< 
          Signature<ArrayXf()>, 
          Signature<ArrayXb()>, 
          Signature<ArrayXi()> >; 
}; 

template<NodeType N>
struct Signatures<N, enable_if_t<is_one_of_v<N, 
    NodeType::Add,
    NodeType::Sub,
    NodeType::Mul,
    NodeType::Div,
    NodeType::Pow
    >>>{
        using type = std::tuple< 
            Signature<ArrayXf(ArrayXf,ArrayXf)>,
            Signature<ArrayXXf(ArrayXXf,ArrayXXf)>
        >; 
    }; 

template<NodeType N>
struct Signatures<N, enable_if_t<is_one_of_v<N,
    NodeType::And,
    NodeType::Or,
    NodeType::Xor
    >>>{ 
        using type = std::tuple< 
            Signature<ArrayXb(ArrayXb,ArrayXb)>,
            Signature<ArrayXXb(ArrayXXb,ArrayXXb)>
        >; 
    }; 

template<> 
struct Signatures<NodeType::Not> { 
    using type = std::tuple<
        Signature<ArrayXb(ArrayXb)>,
        Signature<ArrayXXb(ArrayXXb)>
    >;
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
        using type = std::tuple< 
            Signature<ArrayXf(ArrayXf)>,
            Signature<ArrayXXf(ArrayXXf)>
        >;
    };

template<NodeType N>
struct Signatures<N, enable_if_t<is_one_of_v<N, 
    NodeType::Before,
    NodeType::After,
    NodeType::During
    >>>{ 
        //TODO: Fix
        using type = std::tuple<
            Signature<TimeSeriesf(TimeSeriesf,TimeSeriesf)>,
            Signature<TimeSeriesi(TimeSeriesi,TimeSeriesi)>,
            Signature<TimeSeriesb(TimeSeriesb,TimeSeriesb)>
        >;
    }; 
template<NodeType N>
struct Signatures<N, enable_if_t<is_one_of_v<N, 
    NodeType::Min, 
    NodeType::Max
    >>>{ 
        using type = std::tuple<
            Signature<ArrayXf(ArrayXXf)>,
            Signature<ArrayXi(ArrayXXi)>
            /* Signature<ArrayXb(ArrayXXb)> */
        >;
    }; 

template<NodeType N>
struct Signatures<N, enable_if_t<is_one_of_v<N, 
    NodeType::Mean,
    NodeType::Median
    >>>{ 
        using type = std::tuple<
            Signature<ArrayXf(ArrayXXf)>,
            Signature<ArrayXf(TimeSeriesf)>
            /* Signature<ArrayXf(ArrayXXi)>, */
            /* Signature<ArrayXf(ArrayXXb)> */
        >;
    }; 

template<>
struct Signatures<NodeType::Sum>{ 
        using type = std::tuple<
            Signature<ArrayXf(ArrayXXf)>,
            /* Signature<ArrayXi(ArrayXXi)>, */
            Signature<ArrayXf(TimeSeriesf)>
            /* Signature<ArrayXf(TimeSeriesi)>, */
            /* Signature<ArrayXf(TimeSeriesb)> */
        >;
    }; 

template<> 
struct Signatures<NodeType::Count>{
        using type = std::tuple<
            /* Signature<ArrayXf(ArrayXXb)>, */
            Signature<ArrayXf(TimeSeriesf)>,
            Signature<ArrayXf(TimeSeriesi)>,
            Signature<ArrayXf(TimeSeriesb)>
        >;
};

/* template<NodeType N> */
/* struct Signatures<N, enable_if_t<is_one_of_v<N, */ 
/*     NodeType::Equals, */
/*     NodeType::LessThan, */
/*     NodeType::GreaterThan, */
/*     NodeType::Leq, */
/*     NodeType::Geq, */
/*     NodeType::CustomBinaryOp */
/*     >>>{ */ 
/*         using type = std::tuple< */
/*             Signature<ArrayXb(ArrayXf,ArrayXf)> */
/*             /1* Signature<ArrayXXf(ArrayXXf,ArrayXXf)>, *1/ */
/*         >; */
/*     }; */ 

template<NodeType N>
struct Signatures<N, enable_if_t<is_one_of_v<N, NodeType::SplitBest, NodeType::CustomSplit>>>
    { 
        using type = std::tuple<
            Signature<ArrayXf(ArrayXf,ArrayXf)>, 
            Signature<ArrayXi(ArrayXi,ArrayXi)>, 
            Signature<ArrayXb(ArrayXb,ArrayXb)> 
            // TODO
            /* Signature<ArrayXXf,ArrayXXf,,ArrayXXf,, */
            /* Signature<ArrayXXf,ArrayXXf,,ArrayXXf, */
        >; 
    }; 
template<>
struct Signatures<NodeType::SplitOn>{ 
        using type = std::tuple< 
            Signature<ArrayXf(ArrayXf,ArrayXf,ArrayXf)>,
            Signature<ArrayXf(ArrayXi,ArrayXf,ArrayXf)>,
            /* Signature<ArrayXf(ArrayXb,ArrayXf,ArrayXf)>, */
            Signature<ArrayXi(ArrayXf,ArrayXi,ArrayXi)>,
            Signature<ArrayXi(ArrayXi,ArrayXi,ArrayXi)>
            /* Signature<ArrayXi(ArrayXb,ArrayXi,ArrayXi)>, */
            /* Signature<ArrayXb(ArrayXf,ArrayXb,ArrayXb)>, */
            /* Signature<ArrayXb(ArrayXi,ArrayXb,ArrayXb)>, */
            /* Signature<ArrayXb(ArrayXb,ArrayXb,ArrayXb)> */
        >;
    }; 
} // namespace Brush
// format overload for Nodes
template <> struct fmt::formatter<Brush::NodeType>: formatter<string_view> {
  // parse is inherited from formatter<string_view>.
  template <typename FormatContext>
  auto format(Brush::NodeType x, FormatContext& ctx) const {
    return formatter<string_view>::format(Brush::NodeTypeName.at(x), ctx);
  }
};
#endif
