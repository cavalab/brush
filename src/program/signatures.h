
/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3

This section adapted heavily from Operon: https://github.com/heal-research/operon/
*/
#ifndef SIGNATURES_H
#define SIGNATURES_H
// external includes
// using namespace Brush;
// using std::tuple;
// using std::array;
// using Brush::DataType; 
// using Brush::Data::TimeSeriesb; 
// using Brush::Data::TimeSeriesi; 
// using Brush::Data::TimeSeriesf; 
// using Brush::Util::is_tuple;

namespace Brush {
////////////////////////////////////////////////////////////////////////////////

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

    template<typename T>
    static constexpr bool contains() { return is_one_of_v<T, Args...>; }

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
    using RetType = base::RetType;
    using ArgTypes = base::ArgTypes;
    static constexpr auto ArgCount = base::ArgCount;
    /* using Function = base::Function; */
};
////////////////////////////////////////////////////////////////////////////////
// Signatures
// - store the signatures that each Node can handle
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

// TODO: specialize for variable arity operators that take a vector of inputs
    template <>
    struct Signatures<NodeType::Softmax>
    {
        using type = std::tuple<
            Signature<ArrayXXf(ArrayXXf)>,
            Signature<ArrayXXf(ArrayXf, ArrayXf)>,
            Signature<ArrayXXf(ArrayXf, ArrayXf, ArrayXf)>,
            Signature<ArrayXXf(ArrayXf, ArrayXf, ArrayXf, ArrayXf)>,
            Signature<ArrayXXf(ArrayXf, ArrayXf, ArrayXf, ArrayXf, ArrayXf)>,
            Signature<ArrayXXf(ArrayXf, ArrayXf, ArrayXf, ArrayXf, ArrayXf, ArrayXf)>,
            Signature<ArrayXXf(ArrayXf, ArrayXf, ArrayXf, ArrayXf, ArrayXf, ArrayXf, ArrayXf)>,
            Signature<ArrayXXf(ArrayXf, ArrayXf, ArrayXf, ArrayXf, ArrayXf, ArrayXf, ArrayXf,
                ArrayXf)>,
            Signature<ArrayXXf(ArrayXf, ArrayXf, ArrayXf, ArrayXf, ArrayXf, ArrayXf, ArrayXf,
                ArrayXf, ArrayXf)>,
            Signature<ArrayXXf(ArrayXf, ArrayXf, ArrayXf, ArrayXf, ArrayXf, ArrayXf, ArrayXf,
                ArrayXf, ArrayXf)>
            >;
    };
} // namespace Brush
#endif