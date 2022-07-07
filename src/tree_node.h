#ifndef tree_node_h
#define tree_node_h
#include <tuple>
#include <unordered_map>

#include "init.h"
#include "data/data.h"
#include "node.h"
#include "functions.h"
#include "nodemap.h"
/* #include "operator.h" */
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
        template<typename T>
        auto fit(const Data& d); //{ State s; return std::get<T>(s);};
        template<typename T>
        auto predict(const Data& d); //{ State s; return std::get<T>(s);};
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

        /* template<ExecType E, typename T> struct GetKids; */ 
        /* template<ExecType E, typename T> struct GetKidsFit; */ 
        /* template<ExecType E, typename T> struct GetKidsPredict; */ 

}; 
using TreeNode = class tree_node_<Node>; 
//forward declarations
template<typename R, NodeType NT, SigType S> R DispatchPredict(const Data& d, TreeNode& tn) ;
template<typename R, NodeType NT, SigType S> R DispatchFit(const Data& d, TreeNode& tn) ;

namespace detail {

    template<typename T>
    using Callable = typename std::function<T(const Data&,TreeNode&)>;

    template <typename T, typename TupleCallables>
    struct has_type;

    template <typename T, typename... Us>
    struct has_type<T, std::tuple<Us...>> : std::disjunction<std::is_same<T, Us>...> {};

    /* template<NodeType NT, typename R> */
    template<NodeType N, SigType S>
    static constexpr auto MakeOperator()  
    {
        using R = typename Signature<S>::RetType;
        return Callable<R>(DispatchFit<R,N,S>);
    }




    /* template<typename F, typename... Ts, std::enable_if_t<sizeof...(Ts) != 0 && (std::is_invocable_r_v<void, F, detail::Array<Ts>&, Vector<Node> const&, size_t, Operon::Range> && ...), bool> = true> */
    /* static constexpr auto MakeTuple(F&& f) */
    /* { */
    /*     return std::make_tuple(Callable<Ts>(std::forward<F&&>(f))...); */
    /* } */
    template<typename X, typename TupleCallables>
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
/* template<typename T> struct DispatchTable; */

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

    using CallVariant    = std::variant<Callable<Ts>...>;
    /* using Map      = robin_hood::unordered_flat_map<Operon::Hash, Tuple>; */
    // could have fit map, predict map
    using SigMap = std::unordered_map<SigType,CallVariant>;
    using DTMap = std::unordered_map<NodeType, SigMap>;

private:
    DTMap map_;

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
        /* (map_.insert({ nt(Is), detail::MakeTuple<nt(Is),Ts...>() }), ...); */
        /* (MakeOperators<nt(Is),Ts>(), ...); */
        /* (MakeOperators<nt(Is)>(), ...); */
        (map_.insert({ nt(Is), MakeOperators<nt(Is)>() }), ...);
        //TODO: this really should be a hash, if want to register other functions
    }

    template<NodeType NT, typename Sigs, Sigs S, std::size_t... Is>
    static constexpr auto AddOperator(std::index_sequence<Is...>)
    {
        SigMap sm;
        (sm.insert({std:get<Is>(S), 
                    detail::MakeOperator<NT, std::get<Is>(S)>() 
                   }),
         ...);
        return sm;
    }
    template<NodeType NT>
    SigMap MakeOperators()  
    {
        constexpr auto signatures = Signatures<NT>::value;
        
        return AddOperator<NT, decltype(signatures), signatures>( 
                     std::make_index_sequence<std::tuple_size_v<decltype(signatures)>>()
                     );
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
    /*     constexpr int64_t idx = detail::tuple_index<Callable<T>, TupleCallables>::value; */
    /*     static_assert(idx >= 0, "TupleCallables does not contain type T"); */
    /*     if (auto it = map_.find(h); it != map_.end()) { */
    /*         return std::get<static_cast<size_t>(idx)>(it->second); */
    /*     } */
    /*     throw std::runtime_error(fmt::format("Hash value {} is not in the map\n", h)); */
    /* } */
    template<typename T>
    [[nodiscard]] inline auto Get(NodeType n, SigType s) const -> Callable<T> const&
    {
        return std::get<Callable<T>>(map_.at(n).at(s));
        /* constexpr auto typed_map = std::get<T>(map_); */
        /* constexpr int64_t idx = detail::tuple_index<Callable<T>, TupleCallables>::value; */
        /* static_assert(idx >= 0, "TupleCallables does not contain type T"); */
        /* if (auto it = map_.find(h); it != map_.end()) { */
        /*     return std::get<static_cast<size_t>(idx)>(it->second); */
        /* } */
        /* /1* throw std::runtime_error(fmt::format("Hash value {} is not in the map\n", h)); *1/ */
        /* throw std::runtime_error("Op not found"); */
    }

    /* template<typename F> */
    /* void RegisterCallable(Operon::Hash hash, F&& f) { */
    /*     map_[hash] = detail::MakeTuple<F, Ts...>(std::forward<F&&>(f)); */
    /* } */

    /* template<typename T> */
    /* /1* [[nodiscard]] inline auto TryGet(Operon::Hash const h) const noexcept -> std::optional<Callable<T>> *1/ */
    /* [[nodiscard]] inline auto TryGet(NodeType h) const noexcept -> std::optional<Callable<T>> */
    /* { */
    /*     constexpr int64_t idx = detail::tuple_index<Callable<T>, TupleCallables>::value; */
    /*     static_assert(idx >= 0, "TupleCallables does not contain type T"); */
    /*     if (auto it = map_.find(h); it != map_.end()) { */
    /*         return { std::get<static_cast<size_t>(idx)>(it->second) }; */
    /*     } */
    /*     return {}; */
    /* } */

    /* [[nodiscard]] auto Contains(Operon::Hash hash) const noexcept -> bool { return map_.contains(hash); } */
};

DispatchTable<
              ArrayXf
              /* ArrayXb, */
              /* ArrayXi, */ 
              /* ArrayXf, */ 
              /* ArrayXXb, */
              /* ArrayXXi, */ 
              /* ArrayXXf, */ 
              /* TimeSeriesb, */
              /* TimeSeriesi, */
              /* TimeSeriesf */
             > dtable;
//////////////////////////////////////////////////////////////////////////////////
// fit, eval, predict
template<typename T>
auto TreeNode::fit(const Data& d)
{ 
    auto F = dtable.template Get<T>(n.node_type, n.sig_type);
    return F(d, (*this));
};

template<typename T>
auto TreeNode::predict(const Data& d)
{ 
    auto F = dtable.template Get<T>(n.node_type, n.sig_type);
    return F(d, (*this));
};


}// Brush
#endif
