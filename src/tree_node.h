#ifndef tree_node_h
#define tree_node_h
#include <tuple>

#include "init.h"
#include "data/data.h"
#include "node.h"
#include "functions.h"
#include "operator.h"
#include "thirdparty/eternal.hpp"
/* #include "interpreter.h" */

using std::string;
using Brush::data::Data;
using Brush::ExecType;
using Brush::Node;

namespace Brush {
/// A node in the tree, combining links to other nodes as well as the actual data.
template<class T> class tree_node_; 

// /**
//  * @brief tree node specialization for Node.
//  * 
//  */
//
template<>
class tree_node_<Node> { // size: 5*4=20 bytes (on 32 bit arch), can be reduced by 8.
	public:
        tree_node_()
            : parent(0), first_child(0), last_child(0), prev_sibling(0), next_sibling(0)
            {}

        tree_node_(const Node& val)
            : parent(0), first_child(0), last_child(0), prev_sibling(0), next_sibling(0), n(val)
            {}

        tree_node_(Node&& val)
            : parent(0), first_child(0), last_child(0), prev_sibling(0), next_sibling(0), n(val)
            {}

		tree_node_<Node> *parent;
	    tree_node_<Node> *first_child, *last_child;
		tree_node_<Node> *prev_sibling, *next_sibling;
		Node n;

        /* template<typename T> */
        /* auto eval(const Data& d); */
        /* template<typename T> */
        auto fit(const Data& d){ State s; return std::get<T>(s);};
        /* template<typename T> */
        auto predict(const Data& d){ State s; return std::get<T>(s);};
        /* auto predict(const Data& d) const; */ 
        /* /1* void grad_descent(const ArrayXf&, const Data&); *1/ */
		string get_model(bool pretty=false);
		string get_tree_model(bool pretty=false, string offset="");
    private:
        

        /* template<ExecType E> */
        /* auto eval(const Data& d); */

        /* template<ExecType E> */
        /* auto _predict(const Data& d); */

        /* auto _dispatch(ExecType E, bool train, const Data& d); */

        template<ExecType E, typename T> struct GetKids; 
        template<ExecType E, typename T> struct GetKidsFit; 
        template<ExecType E, typename T> struct GetKidsPredict; 

}; 
typedef class tree_node_<Node> TreeNode; 

namespace detail {
    /* class tree_node_<Node> */
    // dispatching mechanism
    // stolen from Operon 
    /* template<NodeType Type, typename R, typename... Args> */
    template<NodeType Type, typename T> DispatchOpUnary(const Data& d, TreeNode& tn);

    template<NodeType Type, typename... T>
    auto DispatchOpUnary(const Data& d, TreeNode& tn)
    {
        Function<Type> f{};
        return f(tn.first_child->eval(d));
    }
    /* template<NodeType Type, typename R> */
    template<NodeType Type, typename T>
    T DispatchOpBinary(const Data& d, TreeNode& tn)
    {
        Function<Type> f{};
        return f(tn.first_child->eval<T>(d), tn.last_child->eval<T>(d));
    }
    /* State DispatchOpBinary(Data& d, TreeNode& tn) */
    /* { */
    /*     using Arg1 = decltype(nt.n.arg_types.at(0)); */
    /*     using Arg2 = decltype(nt.n.arg_types.at(1)); */
    /*     return Function<Type>{}(n.first_child->eval<Arg1>(d), n.last_child->eval<Arg1>(d)); */
    /* } */

    /* template<NodeType Type, typename R> */
    template<NodeType Type, typename T>
    T DispatchTerminal(const Data& d, TreeNode& tn)
    {
        /* using T = decltype(DataMap<tn.n.ret_type>{}()); */
        return std::get<T>(d[tn.n.feature]);
        /* return d.features[tn.n.feature]; */
    }

    template<NodeType Type, typename T>
    T NoOp(const Data& d, TreeNode& tn)
    {
        return T();
    }

    /* template<typename X, typename Tuple> */
    /* class tuple_index; */

    /* template<typename X, typename... T> */
    /* class tuple_index<X, std::tuple<T...>> { */
    /*     template<std::size_t... Idx> */
    /*     static constexpr auto FindIdx(std::index_sequence<Idx...> ) -> int64_t */
    /*     { */
    /*         return -1 + ((std::is_same<X, T>::value ? Idx + 1 : 0) + ...); */
    /*     } */

    /* public: */
    /*     static constexpr int64_t value = FindIdx(std::index_sequence_for<T...>{}); */
    /* }; */

    template<typename T>
    using Callable = typename std::function<T(const Data&,TreeNode&)>;

    template<NodeType Type, typename T>
    static constexpr auto MakeCall() -> Callable<T>
    {

        if constexpr (Type > NodeType::_SPLITTER_) { 
            return Callable<T>(detail::DispatchOpBinary<Type, T>);
        }
        else if constexpr (Type > NodeType::_BINARY_) { 
            return Callable<T>(detail::DispatchOpBinary<Type, T>);
        }
        else if constexpr (Type > NodeType::_UNARY_) { 
            return Callable<T>(detail::DispatchOpUnary<Type, T>);
        } 
        else if constexpr (Type > NodeType::_LEAF_) { 
            return Callable<T>(detail::DispatchTerminal<Type, T>);
        } 
        return Callable<T>(detail::NoOp<Type, T>);
    }

    /* template<NodeType Type, typename... Ts, std::enable_if_t<sizeof...(Ts) != 0, bool> = true> */
    template<NodeType Type, typename... Ts> 
    static constexpr auto MakeTuple()
    {
        return std::make_tuple(MakeCall<Type, Ts>()...);
        /* return MakeCall<Type>(); */
    };

    template<typename Arg1, typename Arg2>
    using Signature = typename std::tuple<Arg1,Arg2>;

    template<NodeType Type>
    static constexpr auto MakeOperators()  
    {
        // NOTE: I should just handcarft this for each case for now. 
        // It would not be that much code. All this map shit is getting messy.
        // TODO
        // handle arg types for reducers and comparators, which are different than return type 

        // handle cases when node doesn't have that return type
        //
        using TupleRetTypes = RetTypes<Type>::RetTypes;
        //TODO
        // make map a map of maps, to map from node type to return type to operator. 
        // example: map < NodeType, map<DataType,Operator<RetType,ArgTypes>()>>
        (map_.insert({ nt(Is), detail::MakeTuple<nt(Is),Ts...>() }), ...);
    } 
    template<NodeType Type, typename T>
    static constexpr auto MakeOperator()  
        if constexpr (Type > NodeType::_SPLITTER_) { 
            //TODO
            /* return Callable<T>(detail::DispatchOpSplitter<Type, T>); */
            if constexpr (Type == NodeType::SplitOn)  
                //TODO: this one can be different types for FirstArg
                return Operator<Type, T, T, T, T>(); 
            else ifconstexpr (Type == NodeType::SplitBest)  
                return Operator<Type, T, T, T>(); 
            /* return new SplitOp<Type, T>(); */
        }
        else if constexpr (Type > NodeType::_COMPARATOR_) { 
            /* return Callable<T>(detail::DispatchOpBinary<Type, T>); */
            /* return new CompareOp<Type, T>(); */
            // TODO: handle matrix
            typedef ComparisonType<T>::type R; 
            return Operator<Type, R, T, T>();
        }
        else if constexpr (Type > NodeType::_BINARY_) { 
            /* return Callable<T>(detail::DispatchOpBinary<Type, T>); */
            return Operator<Type, T, T, T>();
        }
        else if constexpr (Type > NodeType::_REDUCER_) { 
            /* return Callable<T>(detail::DispatchOpBinary<Type, T>); */
            typedef ReducedType<T>::type R; 
            return Operator<Type, R, T>();
        }
        else if constexpr (Type > NodeType::_UNARY_) { 
            /* return Callable<T>(detail::DispatchOpUnary<Type, T>); */
            /* return new UnaryOp<Type, T>(); */
            return Operator<Type, T, T>();
        } 
        else if constexpr (Type > NodeType::_LEAF_) { 
            /* return Callable<T>(detail::DispatchTerminal<Type, T>); */
            return new Operator<Type, T>();
        } 
        /* return Callable<T>(detail::NoOp<Type, T>); */
        return Operator<Type,T>();
    }




    /* template<typename F, typename... Ts, std::enable_if_t<sizeof...(Ts) != 0 && (std::is_invocable_r_v<void, F, detail::Array<Ts>&, Vector<Node> const&, size_t, Operon::Range> && ...), bool> = true> */
    /* static constexpr auto MakeTuple(F&& f) */
    /* { */
    /*     return std::make_tuple(Callable<Ts>(std::forward<F&&>(f))...); */
    /* } */
    template<typename X, typename Tuple>
    class tuple_index;

    template<typename X, typename... T>
    class tuple_index<X, std::tuple<T...>> {
        template<std::size_t... Idx>
        static constexpr auto FindIdx(std::index_sequence<Idx...> /*unused*/) -> int64_t
        {
            return -1 + ((std::is_same<X, T>::value ? Idx + 1 : 0) + ...);
        }

    public:
        static constexpr int64_t value = FindIdx(std::index_sequence_for<T...>{});
    };
} // namespace detail

////////////////////////////////////////////////////////////////////////////////
// Dispatch Table
template<typename Signatures> DispatchTable;

template<typename... Ts> 
struct DispatchTable
{
    
    //TODO: this Ts specifies the return type in Operon, which is one of two values, an Eigen array
    // or a Dual. 
    // The callables are templated by this type. In our case, we have functions that return
    // different types but have arbitrary signature types as well. 
    // One option is that we have the dispatch table templated by these, but then we have
    // disparate dispatch tables for every function signature (maybe this ok...). 
    // However, the logic around which node types have which function signatures might get
    // complicated. We'd have to avoid adding nodes to the map for a given type if they don't
    // have that return type. (might not be that hard... but wouldn't happen at compile time.)
    // Right now, I'm trying to use State returns, so that the Callable signature is the same for
    // different dispatch functions. Not sure if this will work.  
    template<typename T>
    using Callable = detail::Callable<T>;

    using Tuple    = std::tuple<Callable<Ts>...>;
    /* using Map      = robin_hood::unordered_flat_map<Operon::Hash, Tuple>; */
    using Map      = std::unordered_map<NodeType, Tuple>;

private:
    Map map_;

    template<std::size_t... Is>
    void InitMap(std::index_sequence<Is...> /*unused*/)
    {
        auto nt = [](auto i) { return static_cast<NodeType>(1UL << i); };
        /* auto et = [](auto i) { */ 
        /*     return static_cast<ExecType>(NodeSchema[NodeTypeName[i]]["ExecType"]); */
        /* }; */
        /* (map_.insert({ Node(f(Is)).HashValue, detail::MakeTuple<f(Is), Ts...>() }), ...); */
        //TODO: use MakeTuple to determine exec type from node type. We could implement
        // exectypes as an Is"blah"<> function in the Node class even. 
        (map_.insert({ nt(Is), detail::MakeTuple<nt(Is),Ts...>() }), ...);
        //TODO: this really should be a hash, if want to register other functions
    }

public:
    DispatchTable()
    {
        InitMap(std::make_index_sequence<NodeTypes::Count>{}); 
    }

    ~DispatchTable() = default;

    auto operator=(DispatchTable const& other) -> DispatchTable& {
        if (this != &other) {
            map_ = other.map_;
        }
        return *this;
    }

    auto operator=(DispatchTable&& other) noexcept -> DispatchTable& {
        map_ = std::move(other.map_);
        return *this;
    }

    DispatchTable(DispatchTable const& other) : map_(other.map_) { }
    DispatchTable(DispatchTable &&other) noexcept : map_(std::move(other.map_)) { }

    /* template<typename T> */
    /* inline auto Get(Operon::Hash const h) -> Callable<T>& */
    /* { */
    /*     return const_cast<Callable<T>&>(const_cast<DispatchTable<Ts...> const*>(*this)->Get(h)); // NOLINT */
    /* } */

    /* template<typename T> */
    /* [[nodiscard]] inline auto Get(Operon::Hash const h) const -> Callable<T> const& */
    /* { */
    /*     constexpr int64_t idx = detail::tuple_index<Callable<T>, Tuple>::value; */
    /*     static_assert(idx >= 0, "Tuple does not contain type T"); */
    /*     if (auto it = map_.find(h); it != map_.end()) { */
    /*         return std::get<static_cast<size_t>(idx)>(it->second); */
    /*     } */
    /*     throw std::runtime_error(fmt::format("Hash value {} is not in the map\n", h)); */
    /* } */
    template<typename T>
    [[nodiscard]] inline auto Get(NodeType h) const -> Callable<T> const&
    {
        constexpr int64_t idx = detail::tuple_index<Callable<T>, Tuple>::value;
        static_assert(idx >= 0, "Tuple does not contain type T");
        if (auto it = map_.find(h); it != map_.end()) {
            return std::get<static_cast<size_t>(idx)>(it->second);
        }
        /* throw std::runtime_error(fmt::format("Hash value {} is not in the map\n", h)); */
        throw std::runtime_error("Op not found");
    }

    /* template<typename F> */
    /* void RegisterCallable(Operon::Hash hash, F&& f) { */
    /*     map_[hash] = detail::MakeTuple<F, Ts...>(std::forward<F&&>(f)); */
    /* } */

    template<typename T>
    /* [[nodiscard]] inline auto TryGet(Operon::Hash const h) const noexcept -> std::optional<Callable<T>> */
    [[nodiscard]] inline auto TryGet(NodeType h) const noexcept -> std::optional<Callable<T>>
    {
        constexpr int64_t idx = detail::tuple_index<Callable<T>, Tuple>::value;
        static_assert(idx >= 0, "Tuple does not contain type T");
        if (auto it = map_.find(h); it != map_.end()) {
            return { std::get<static_cast<size_t>(idx)>(it->second) };
        }
        return {};
    }

    /* [[nodiscard]] auto Contains(Operon::Hash hash) const noexcept -> bool { return map_.contains(hash); } */
};

DispatchTable<
              ArrayXb,
              ArrayXi, 
              ArrayXf, 
              ArrayXXb,
              ArrayXXi, 
              ArrayXXf, 
              TimeSeriesb,
              TimeSeriesi,
              TimeSeriesf
             > dtable;
//////////////////////////////////////////////////////////////////////////////////
// fit, eval, predict
auto TreeNode::fit(const Data& d)
{ 
    auto F = dtable.template Get<T>(n.node_type);
    return F.fit(d, (*this));
};

auto TreeNode::predict(const Data& d)
{ 
    auto F = dtable.template Get<T>(n.node_type);
    return F.predict(d, (*this));
};

////////////////////////////////////////////////////////////////////////////////
// signature compile-time maps
//
using CTMap = mapblox::eternal::map;
using Signature = CTMap<DataType,std::array<DataType,N>>;

MAPBOX_ETERNAL_CONSTEXPR const auto UnaryFtoF = \
    Signature({
            {DataType::ArrayF, {DataType::ArrayF, DataType::_NONE_} },
            {DataType::MatrixF, {DataType::MatrixF, DataType::_NONE_} },
            });

MAPBOX_ETERNAL_CONSTEXPR const auto BinaryFFtoF = \
    Signature({
            {DataType::ArrayF, {DataType::ArrayF, DataType::ArrayF} },
            {DataType::MatrixF, {DataType::MatrixF, DataType::MatrixF} },
            });

MAPBOX_ETERNAL_CONSTEXPR const auto BinaryBBtoB = \
    Signature({
            {DataType::ArrayB, {DataType::ArrayB, DataType::ArrayB} },
            {DataType::MatrixB, {DataType::MatrixB, DataType::MatrixB} },
            });

/* template<NodeType Type, typename R, typename ...Args> */ 
/* struct Signature{ */
/*     using TupleArgs = std::tuple<Args...>; */
/* } */
/* template<NodeType Type, typename R> */ 
/* struct UnarySignature : Signature<Type, R, R> { */
/*     using TupleArgs = Signature::TupleArgs; */
/* }; */
/* template<NodeType Type, typename R> */ 
/* struct BinarySignature : Signature<Type, R, R, R> { */
/*     using TupleArgs = Signature::TupleArgs; */
/* }; */

using SigMap = CTmap<NodeType, Signature>;
MAPBOX_ETERNAL_CONSTEXPR const auto SignatureTable = SigMap({
    { NodeType::Abs, UnaryFtoF },
    { NodeType::Add, BinaryFFoF },
});

template<typename T=ArrayXXf> struct ReducedType{ using type=ArrayXf; };
template<> struct ReducedType<TimeSeriesf>{ using type=ArrayXf; };
template<> struct ReducedType<ArrayXXi>{ using type=ArrayXi; };
template<> struct ReducedType<TimeSeriesi>{ using type=ArrayXi; };
template<> struct ReducedType<ArrayXXb>{ using type=ArrayXb; };
template<> struct ReducedType<TimeSeriesb>{ using type=ArrayXb; };

template<typename T=ArrayXf> struct ComparisonType{ using type=ArrayXb; };
template<> struct ComparisonType<ArrayXi>{ using type=ArrayXb; };
template<> struct ComparisonType<ArrayXXf>{ using type=ArrayXXb; };
template<> struct ComparisonType<ArrayXXi>{ using type=ArrayXXb; };
}// Brush
#endif
