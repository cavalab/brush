#ifndef TREE_NODE_H
#define TREE_NODE_H
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
		string get_model(bool pretty=false) const;
		string get_tree_model(bool pretty=false, string offset="") const;

}; 
using TreeNode = class tree_node_<Node>; 
//forward declarations
/* template<typename R, NodeType NT, type_index S> R DispatchPredict(const Data& d, TreeNode& tn) ; */
/* template<typename R, NodeType NT, type_index S> R DispatchFit(const Data& d, TreeNode& tn) ; */
///////////////////////////////////////////////////////////////////////////////////////
// Operator class
template<NodeType NT, typename S> 
struct Operator 
{
    // TODO
    using Args = typename S::ArgTypes;
    using RetType = typename S::RetType;
    static constexpr size_t ArgCount = S::ArgCount;
    // get arg types from tuple by index
    template <std::size_t N>
    using NthType = conditional_t<is_array_v<Args>,
                                  typename Args::value_type,
                                  typename std::tuple_element<N, Args>::type
                                 >;
    
    static constexpr auto F = [](const auto& ...args){ Function<NT> f{}; return f(args...); }; 
    /* static constexpr Function<NT> F{}; */

    Operator() = default;
    /* Operator(NT node_type, RetType y, Args... args){}; */
    ////////////////////////////////////////////////////////////////////////////////
    /// Apply weights
    template<typename T=Args>
    enable_if_t<is_array_v<T,void>> 
    apply_weights(T& inputs, const Node& n) const
    {
        cout << "applying weights to " << n.name << " operator\n";
        std::transform(
                    inputs.begin(), 
                    inputs.end(),
                    n.W.begin(),
                    inputs.begin(), 
                    std::multiplies<>()
                    );
    };
    ////////////////////////////////////////////////////////////////////////////////
    /// Utilities to grab child outputs.
    /* template<typename T> T get_kids(const Data&, TreeNode&, bool fit) const; */

    // get a std::array of kids
    template<bool Fit,typename T=Args>
    enable_if_t<is_array_v<T>, T> 
    get_kids(const Data& d, TreeNode& tn) const
    {
        T child_outputs;
        using arg_type = typename T::value_type;

        TreeNode* sib = tn.first_child;
        for (int i = 0; i < this->get_arg_count(); ++i)
        {
            if constexpr (Fit)
                child_outputs.at(i) = sib->fit<arg_type>(d) ;
            else
                child_outputs.at(i) = sib->predict<arg_type>(d);
            sib = sib->next_sibling;
        }
        return child_outputs;
    };


    // get a std::tuple of kids
    template<int I,bool Fit>
    auto get_kid(const Data& d,TreeNode& tn ) const
    {
        auto sib = tn.first_child; 
        for (int i = 0 ; i < I; ++i)
            sib = sib->next_sibling;
        if constexpr(Fit)
            return sib->fit<NthType<I>>(d);
        else
            return sib->predict<NthType<I>>(d);
    };

    template<typename T, bool Fit, size_t ...Is>
    requires (!is_array_v<T>)
    auto get_kids_seq(const Data& d, TreeNode& tn, std::index_sequence<Is...>) const 
    { 
        return std::make_tuple(get_kid<Is,Fit>(d,tn)...);
    };

    // tuple get kids
    template<bool Fit, typename T=Args>
    requires (!is_array_v<T>)
    auto get_kids(const Data& d, TreeNode& tn) const
    {
        return get_kids_seq<T,Fit>(d, tn, std::make_index_sequence<ArgCount>{});
    };

    // fit and predict convenience functions
    auto get_kids_fit(const Data& d, TreeNode& tn) const { return get_kids<true>(d, tn); };
    auto get_kids_predict(const Data& d, TreeNode& tn) const { return get_kids<false>(d, tn); };

    ///////////////////////////////////////////////////////////////////////////
    // fit and predict
    template<bool Fit, typename T=Args>
    requires (is_array_v<T>)
    RetType eval(const Data& d, TreeNode& tn) const
    {
        auto inputs = get_kids<Fit>(d, tn);
        if (tn.n.is_weighted)
            this->apply_weights(inputs, tn.n);
        return std::apply(F, inputs);
    };

    template<bool Fit, typename T=Args>
    requires (!is_array_v<T>)
    RetType eval(const Data& d, TreeNode& tn) const
    {
        auto inputs = get_kids<Fit>(d, tn);
        return std::apply(F, inputs);
    };

    RetType fit(const Data& d, TreeNode& tn) const { return eval<true>(d,tn); };

    RetType predict(const Data& d, TreeNode& tn) const { return eval<false>(d,tn); };
};
/// Terminal Overload
template<typename S>
struct Operator<NodeType::Terminal, S>
{
    using RetType = typename S::RetType;
    RetType eval(const Data& d, TreeNode& tn) const { return std::get<RetType>(d[tn.n.feature]); };
    RetType fit(const Data& d, TreeNode& tn) const { return eval(d,tn); };
    RetType predict(const Data& d, TreeNode& tn) const { return eval(d,tn); };
};
template<typename S> 
struct Operator<NodeType::Constant, S>
{
    using RetType = typename S::RetType;

    template<typename T=RetType> requires same_as<T, ArrayXf>
    RetType eval(const Data& d, TreeNode& tn) const { 
        return tn.n.W.at(0)*RetType(d.n_samples); 
    };

    template<typename T=RetType> requires same_as<T, ArrayXi>
    RetType eval(const Data& d, TreeNode& tn) const { 
        return int(tn.n.W.at(0))*RetType(d.n_samples); 
    };

    template<typename T=RetType> requires same_as<T, ArrayXb>
    RetType eval(const Data& d, TreeNode& tn) const { 
        return RetType(d.n_samples) > tn.n.W.at(0); 
    };

    template<typename T=RetType> requires same_as<T, ArrayXXf>
    RetType eval(const Data& d, TreeNode& tn) const { 
        return tn.n.W.at(0)*RetType(d.n_samples, d.n_features); 
    };

    template<typename T=RetType> requires same_as<T, ArrayXXb>
    RetType eval(const Data& d, TreeNode& tn) const { 
        return RetType(d.n_samples, d.n_features) > tn.n.W.at(0);
    };
    
    RetType fit(const Data& d, TreeNode& tn) const { return eval(d,tn); };
    RetType predict(const Data& d, TreeNode& tn) const { return eval(d,tn); };
};
////////////////////////////////////////////////////////////////////////////
// fit and predict Dispatch functions
template<typename R, NodeType NT, typename S> //, typename ...Args>
R DispatchFit(const Data& d, TreeNode& tn) 
{
    const auto op = Operator<NT,S>{};
    return op.fit(d, tn);
};

template<typename R, NodeType NT, typename S>
R DispatchPredict(const Data& d, TreeNode& tn) 

{
    const auto op = Operator<NT,S>{};
    return op.predict(d, tn);
};

namespace detail {

    template<typename T>
    using Callable = typename std::function<T(const Data&,TreeNode&)>;

    template <typename T, typename TupleCallables>
    struct has_type;

    template <typename T, typename... Us>
    struct has_type<T, std::tuple<Us...>> : std::disjunction<std::is_same<T, Us>...> {};

    /* template<NodeType NT, typename R> */
    template<NodeType N, typename S>
    static constexpr auto MakeOperator()  
    {
        using R = typename S::RetType;
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
    using SigMap = std::unordered_map<std::size_t,CallVariant>;
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

    /* template<NodeType NT, typename Sigs, Sigs S, std::size_t... Is> */
    template<NodeType NT, typename Sigs, std::size_t... Is>
    static constexpr auto AddOperator(std::index_sequence<Is...>)
    {
        SigMap sm;
        (sm.insert({typeid(std::tuple_element_t<Is, Sigs>).hash_code(), 
                    detail::MakeOperator<NT, std::tuple_element_t<Is, Sigs>>()}), ...);
        return sm;
    }
    template<NodeType NT>
    SigMap MakeOperators()  
    {
        /* constexpr auto signatures = Signatures<NT>::value; */
        using signatures = typename Signatures<NT>::type;
        
        /* return AddOperator<NT, decltype(signatures)>( */ 
        return AddOperator<NT, signatures>( 
                     std::make_index_sequence<std::tuple_size_v<signatures>>()
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

    template<typename T>
    inline auto Get(NodeType n, std::size_t s) const -> Callable<T> const&
    {
        return std::get<Callable<T>>(map_.at(n).at(s));
    }

};

const DispatchTable<
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
template<typename T>
auto TreeNode::fit(const Data& d)
{ 
    auto F = dtable.template Get<T>(n.node_type, n.sig_hash);
    return F(d, (*this));
};

template<typename T>
auto TreeNode::predict(const Data& d)
{ 
    auto F = dtable.template Get<T>(n.node_type, n.sig_hash);
    return F(d, (*this));
};


}// Brush
#endif
