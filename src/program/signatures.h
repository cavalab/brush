/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3

*/
#ifndef SIGNATURES_H
#define SIGNATURES_H

namespace Brush {
////////////////////////////////////////////////////////////////////////////////
// refs:
// https://stackoverflow.com/questions/25958259/how-do-i-find-out-if-a-tuple-contains-a-type
// https://stackoverflow.com/questions/34111060/c-check-if-the-template-type-is-one-of-the-variadic-template-types


// TODO: potentially improve this with something like
// template <typename T, typename S> struct Jetify; 
// template<typename T, 
//     typename S=T::Scalar, 
//     typename R=T::RowsAtCompileTime,
//     typename C=T::ColsAtCompileTime,
//     > 
// struct Jetify<Eigen::ArrayBase<T>> { 
//     using Scalar = std::conditional_t<is_same_v<S,int>, iJet, 
//         conditional_t<is_same_v<S,bool>,bJet, 
//         conditional_t<is_same_v<S,float>,fJet, void>>>;
//     using type = Array<Scalar,R,C>;
// };

static constexpr size_t MAX_ARGS = 5;

template <typename T> struct Jetify { using type = T;};
template<> struct Jetify<ArrayXf> { using type = ArrayXfJet;};
template<> struct Jetify<ArrayXi> { using type = ArrayXiJet;};
template<> struct Jetify<ArrayXb> { using type = ArrayXbJet;};
template<> struct Jetify<ArrayXXf> { using type = ArrayXXfJet;};
template<> struct Jetify<ArrayXXi> { using type = ArrayXXiJet;};
template<> struct Jetify<ArrayXXb> { using type = ArrayXXbJet;};
template<> struct Jetify<Data::TimeSeriesf> { using type = Data::TimeSeriesfJet;};
template<> struct Jetify<Data::TimeSeriesi> { using type = Data::TimeSeriesiJet;};
template<> struct Jetify<Data::TimeSeriesb> { using type = Data::TimeSeriesbJet;};
template <typename T> 
using Jetify_t = typename Jetify<T>::type;

template <typename T> struct UnJetify { using type = T;};
template<> struct UnJetify<ArrayXfJet> { using type = ArrayXf;};
template<> struct UnJetify<ArrayXiJet> { using type = ArrayXi;};
template<> struct UnJetify<ArrayXbJet> { using type = ArrayXb;};
template<> struct UnJetify<ArrayXXfJet> { using type = ArrayXXf;};
template<> struct UnJetify<ArrayXXiJet> { using type = ArrayXXi;};
template<> struct UnJetify<ArrayXXbJet> { using type = ArrayXXb;};
template<> struct UnJetify<Data::TimeSeriesfJet> { using type = Data::TimeSeriesf;};
template<> struct UnJetify<Data::TimeSeriesiJet> { using type = Data::TimeSeriesi;};
template<> struct UnJetify<Data::TimeSeriesbJet> { using type = Data::TimeSeriesb;};
template <typename T> 
using UnJetify_t = typename UnJetify<T>::type;


template<typename R, typename... Args>
struct SigBase  
{
    using RetType = R;
    static constexpr std::size_t ArgCount = sizeof...(Args);

    using FirstArg = std::tuple_element_t<0, std::tuple<Args...>>;

    /// ArgTypes is a std::array if the types are shared, otherwise it is a tuple. 
    // (using std::array allows begin() and end() ops like transform to be applied)
    // TODO: add an option to have argtypes be an ArrayX<T,-1,ArgCount> if the nodetype
    // is associative and the ArgCount is greater than the operator's arg count
    // (i.e., add is a BinaryOp and associative, so for Args>2 make the argtype an ArrayXX<T> )
    using ArgTypes = conditional_t<(std::is_same_v<FirstArg,Args> && ...),
                                   std::array<FirstArg,ArgCount>,
                                   std::tuple<Args...>
                                  >;
    template <std::size_t N>
    using NthType = conditional_t<!is_tuple<ArgTypes>::value, 
                                  FirstArg,
                                  typename std::tuple_element<N, ArgTypes>::type
                                 >;
    using WeightType = typename WeightType<FirstArg>::type; 
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
    static constexpr bool contains() { return is_in_v<T, Args...>; }

    static constexpr std::size_t hash_args(){ return typeid(ArgTypes).hash_code();}

    static constexpr std::size_t hash(){ return typeid(tuple<R,Args...>).hash_code();};
};
/// specialization for terminals
template<typename R>
struct SigBase<R>
{
    using RetType = R;
    using ArgTypes = void;
    using FirstArg = void;
    using WeightType = typename WeightType<R>::type;
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
    using FirstArg = base::FirstArg;
    using WeightType = base::WeightType;
    static constexpr auto ArgCount = base::ArgCount;

    using Dual = SigBase<Jetify_t<RetType>, Jetify_t<Args>... >;
    using DualArgs = SigBase<RetType, Jetify_t<Args>... >;
};

template<typename R, typename Arg, size_t ArgCount, 
    typename Indices = std::make_index_sequence<ArgCount> >
struct NarySignature
{
    template <std::size_t N>
    using NthType = Arg; 

    template<size_t ...Is>
    static constexpr auto make_signature(std::index_sequence<Is...>)
    {
        return Signature<R(NthType<Is>...)>{};
    }

    using type = decltype(make_signature(Indices{})); 

};
template<typename R, typename Arg, size_t ArgCount> 
using NarySignature_t = typename NarySignature<R,Arg,ArgCount>::type;

template<typename R, typename Arg, size_t MaxArgCount>
struct NarySignatures
{
    template <std::size_t N>
    using NthType = Arg; 
    static constexpr size_t Min = 2;
    static constexpr size_t Max = MaxArgCount-2;
    static constexpr auto Indices = std::make_index_sequence<Max>();

    template<size_t ...Is>
    static constexpr auto make_signatures(std::index_sequence<Is...>)
    {
        // return std::make_tuple(NarySignature<R,Arg,Is+Min>() ...);
        return std::tuple<NarySignature_t<R,Arg,(Is+Min)> ...>();
    }

    using type = decltype(make_signatures(Indices)); 

};
template<typename R, typename Arg, size_t MaxArgCount> 
using NarySignatures_t = typename NarySignatures<R,Arg,MaxArgCount>::type;

////////////////////////////////////////////////////////////////////////////////
// Signatures
// - store the signatures that each Node can handle
//
template<NodeType N, typename T = void> struct Signatures; 

template<NodeType N>
struct Signatures<N, enable_if_t<is_in_v<N, NodeType::Constant, NodeType::Terminal>>>{ 
    using type = std::tuple< 
          Signature<ArrayXf()>, 
          Signature<ArrayXi()> 
          >; 
}; 

template<NodeType N>
struct Signatures<N, enable_if_t<is_in_v<N, 
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

// template<NodeType N>
// struct Signatures<N, enable_if_t<is_in_v<N,
//     NodeType::And,
//     NodeType::Or,
//     NodeType::Xor
//     >>>{ 
//         using type = std::tuple< 
//             Signature<ArrayXb(ArrayXb,ArrayXb)>,
//             Signature<ArrayXXb(ArrayXXb,ArrayXXb)>
//         >; 
//     }; 

// template<> 
// struct Signatures<NodeType::Not> { 
//     using type = std::tuple<
//         Signature<ArrayXb(ArrayXb)>,
//         Signature<ArrayXXb(ArrayXXb)>
//     >;
// };

template<NodeType N> 
struct Signatures<N, enable_if_t<is_in_v<N,
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
    NodeType::Square,
    NodeType::Logistic
    >>>{ 
        // using type = std::tuple< 
        //     Signature<ArrayXf(ArrayXf)>,
        //     Signature<ArrayXXf(ArrayXXf)>
        // >;
        using unaryTuple = std::tuple<
            Signature<ArrayXf(ArrayXf)>,
            Signature<ArrayXXf(ArrayXXf)>
        >;

        using naryTuple = NarySignatures_t<ArrayXXf,ArrayXf,MAX_ARGS>;

        using type = decltype(std::tuple_cat(unaryTuple(), naryTuple()));
    };

template<NodeType N>
struct Signatures<N, enable_if_t<is_in_v<N, 
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
// template<NodeType N>
// struct Signatures<N, enable_if_t<is_in_v<N, 
//     NodeType::Min, 
//     NodeType::Max
//     >>>{ 
//         using unaryTuple = std::tuple<
//             Signature<ArrayXf(ArrayXXf)>,
//             Signature<ArrayXi(ArrayXXi)>
//         >;

//         using naryTupleF = NarySignatures_t<ArrayXf,ArrayXf,MAX_ARGS>;
//         using naryTupleI = NarySignatures_t<ArrayXi,ArrayXi,MAX_ARGS>;

//         using type = decltype(std::tuple_cat(unaryTuple(), naryTupleF(), naryTupleI()));
//     }; 

template<NodeType N>
struct Signatures<N, enable_if_t<is_in_v<N, 
    NodeType::Min, 
    NodeType::Max,
    NodeType::Sum,
    NodeType::Mean,
    NodeType::Median
    >>>{ 
        using unaryTuple = std::tuple<
            Signature<ArrayXf(ArrayXXf)>,
            Signature<ArrayXf(TimeSeriesf)>
        >;

        using naryTuple = NarySignatures_t<ArrayXf,ArrayXf,MAX_ARGS>;

        using type = decltype(std::tuple_cat(unaryTuple(), naryTuple()));
    }; 

// template<>
// struct Signatures<NodeType::Sum>{ 
//         using type = std::tuple<
//             Signature<ArrayXf(ArrayXXf)>,
//             /* Signature<ArrayXi(ArrayXXi)>, */
//             Signature<ArrayXf(TimeSeriesf)>
//             /* Signature<ArrayXf(TimeSeriesi)>, */
//             /* Signature<ArrayXf(TimeSeriesb)> */
//         >;
//     }; 

template<> 
struct Signatures<NodeType::Count>{
        using type = std::tuple<
            /* Signature<ArrayXf(ArrayXXb)>, */
            Signature<ArrayXf(TimeSeriesf)>,
            Signature<ArrayXf(TimeSeriesi)>,
            Signature<ArrayXf(TimeSeriesb)>
        >;
};

template<NodeType N>
struct Signatures<N, enable_if_t<is_in_v<N, 
    NodeType::ArgMax 
    >>>{ 
        using type = std::tuple<
            Signature<ArrayXi(ArrayXXf)>,
            Signature<ArrayXi(ArrayXXi)>,
            Signature<ArrayXi(ArrayXXb)>
        >;
    }; 

/* template<NodeType N> */
/* struct Signatures<N, enable_if_t<is_in_v<N, */ 
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
struct Signatures<N, enable_if_t<is_in_v<N, NodeType::SplitBest, NodeType::CustomSplit>>>
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

    template <>
    struct Signatures<NodeType::Softmax>
    {
        using unaryTuple = std::tuple< Signature<ArrayXXf(ArrayXXf)> >;
        using naryTuple = NarySignatures_t<ArrayXXf,ArrayXf,MAX_ARGS>;

        using type = decltype(std::tuple_cat(unaryTuple(), naryTuple()));
    };
} // namespace Brush
#endif