#ifndef tree_node_h
#define tree_node_h
#include <tuple>

#include "init.h"
#include "data/data.h"
#include "node.h"
#include "operators.h"
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

        template<typename T>
        auto eval(const Data& d);
        template<typename T>
        auto fit(const Data& d){ State s; return std::get<T>(s);};
        template<typename T>
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

        template<typename T>
        void apply_weights(T& inputs)
        {
            cout << "applying weights to " << this->n.name << " operator\n";
            std::transform(
                        inputs.begin(), 
                        inputs.end(),
                        n.W.begin(),
                        inputs.begin(), 
                        std::multiplies<>()
                        );
        }
}; 
typedef class tree_node_<Node> TreeNode; 

namespace detail {
    /* class tree_node_<Node> */
    // dispatching mechanism
    // stolen from Operon 
    /* template<NodeType Type, typename R, typename... Args> */
    template<NodeType Type, typename T>
    T DispatchOpUnary(const Data& d, TreeNode& tn)
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

    template<NodeType Type, typename T>
    static constexpr auto MakeSignature() -> Signature<T>
    {

        if constexpr (Type > NodeType::_SPLITTER_) { 
            return;
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
template<typename... Ts>
struct DispatchTable {
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
template<typename T>
auto TreeNode::eval(const Data& d)
{ 
    auto F = dtable.template Get<T>(n.node_type);
    return F(d, (*this));
};

/* template<> */
/* void TreeNode::grad_descent(const ArrayXf& gradient, const Data& d) */
/* { */
/*     /1* _grad_descent(gradient, d); *1/ */
/* }; */

/* template<> */
/* string TreeNode::get_model(bool pretty) */
/* { */
/*     return this->n.get_model(pretty, first_child, last_child); */
/* } */

/* template<> */
/* string TreeNode::get_tree_model(bool pretty, string offset) */
/* { */
/*     return this->n.get_tree_model(pretty, offset, first_child, last_child); */
/* } */

/* template <>*/ 
/* auto TreeNode::_fit<ExecType::Binary>(const Data& d)*/
/* {*/
/*     if (this->n.is_weighted)*/
/*         return Brush::Function<n.node_type>( n.W[0]*first_child->fit(d), n.W[1]*last_child->fit(d) ); */
/*         return dispatch_table.apply(n.node_type, n.W[0]*first_child->fit(d), n.W[1]*last_child->fit(d) );*/
/*     else*/
/*         return Brush::Function<n.node_type>( first_child->fit(d), last_child->fit(d) ); */
/*         return dispatch_table.apply(n.node_type, n.W[0]*first_child->fit(d), n.W[1]*last_child->fit(d) );*/
/* };*/

/* template <>*/ 
/* auto TreeNode::_fit<ExecType::Unary>(const Data& d)*/
/* {*/
/*     if (this->n.is_weighted)*/
/*         return Brush::Function<n.node_type>( n.W[0]*first_child->fit(d)) );*/
/*     else*/
/*         return Brush::Function<n.node_type>( first_child->fit(d) );*/
/* };*/

/* template <>*/ 
/* auto TreeNode::_fit<ExecType::Applier>(const Data& d)*/
/* {*/
    /* auto signature = NodeSchema[n.node_type]["Signature"][n.ret_type]; */ 
/*     typedef decltype(n.signature()) signature;*/

/*     auto inputs = GetKids<n.exec_type,signature>(d);*/
    
/*     if (this->n.is_weighted){*/
/*         apply_weights(inputs);*/
/*     }*/
/*     // State out = Util::apply(this->op, inputs);*/
/*     // cout << "returning " << std::get<R>(out) << endl;*/
/*     return std::apply(Brush::Function<n.node_type>, inputs);*/
/* };*/

/* template <>*/ 
/* auto TreeNode::_fit<ExecType::Transformer>(const Data& d)*/
/* {*/
/*     typedef decltype(n.signature().at(0)) signature;*/

/*     auto outputs = GetKidsFit<n.exec_type,signature>(d);*/
    
/*     if (this->n.is_weighted) {*/
/*         apply_weights(inputs);*/
/*     }*/
/*     // State out = Util::apply(this->op, inputs);*/
/*     // cout << "returning " << std::get<R>(out) << endl;*/
/*     std::transform(*/
/*                 outputs.begin(),*/ 
/*                 outputs.end(),*/
/*                 outputs.begin(),*/ 
/*                 Function<n.node_type>*/
/*                 );*/

/*     return outputs;*/ 
/* };*/

/* template <>*/ 
/* auto TreeNode::_fit<ExecType::Reducer>(const Data& d)*/
/* {*/
/*     typedef decltype(n.signature().at(0)) signature;*/

/*     auto inputs = GetKidsFit<n.exec_type,signature>(d);*/
    
/*     if (this->n.is_weighted) {*/
/*         apply_weights(inputs);*/
/*     }*/
/*     // State out = Util::apply(this->op, inputs);*/
/*     // cout << "returning " << std::get<R>(out) << endl;*/
/*     signature output = std::reduce(inputs.begin(), inputs.end(), signature(0),*/ 
/*                                    Function<n.node_type>);*/
/*     return output;*/ 
/* };*/


/* template <>*/ 
/* auto TreeNode::_fit<ExecType::Splitter>(const Data& d)*/
/* {*/
/*     // need to handle kind with fixed var and kind without*/

/*     // set feature and threshold*/
/*     if (this->n.fixed_variable)*/
/*     {*/
/*         tie(this->n.threshold, ignore) = best_threshold(*/
/*                                                      d[this->n.feature],*/
/*                                                      d.y,*/ 
/*                                                      d.classification*/
/*                                                      );*/
/*     }*/
/*     else*/
/*         set_variable_and_threshold(d);*/

/*     auto data_splits = Function<n.node_type>(d);*/ 
/*     ArrayXb mask = this->threshold_mask(d);*/
/*     array<Data, 2> data_splits = d.split(mask);*/

    /* array<State, base::ArgCount> kid_outputs; */
/*     //TODO: type for kids. also, handle scenario where first kid is the variable to split on.*/
    //
/*     typedef decltype(n.signature()) signature;*/

/*     auto kid_outputs = GetKidsFit<ExecType::Splitter,signature>(data_splits);*/ 
/*     // stitch together outputs*/
/*     State out = stitch(kid_outputs, d, mask);*/

/*     cout << "returning " << std::get<R>(out) << endl;*/

/*     return out;*/
/* };*/

/* template <>*/ 
/* auto TreeNode::_fit<ExecType::Terminal>(const Data& d)*/
/* {*/
/*     return this->predict(d);*/
/* };*/

/* ////////////////////////////////////////////////////////////////////////////////*/
/* // children fetching functions for nary operators*/

/* returns a fixed-sized array of arguments of the same type.
 */
/* template<typename T>*/
/* struct TreeNode::GetKids<ExecType::Applier, T>*/
/* {*/
    /* ArrayArgs */
/*     template <std::size_t N>*/
/*     using NthType = typename std::tuple_element<N, T>::type;*/

/*     T operator()(const Data& d, auto (TreeNode::*fn)(const Data&))*/
/*     {*/
/*         // why not make get kids return the tuple? because tuples suck with weights*/
/*         T kid_outputs;*/

/*         TreeNode* sib = first_child;*/
/*         for (int i = 0; i < kid_outputs.size(); ++i)*/
/*         {*/
/*             kid_outputs.at(i) = (sib->*fn)(d);*/
/*             sib = sib->next_sibling;*/
/*         }*/
/*         return kid_outputs;*/
/*     };*/
/* };*/

/* returns a vector of arguments of the same type. for nary children.
   should be used for ExectType::Transformer and Reducer.*/
/* template<ExecType E, typename T> */
/* struct TreeNode::GetKids */
/* { */
/*     auto operator()(const Data& d, auto (TreeNode::*fn)(const Data&) ) */
/*     { */
/*         vector<T> kid_outputs; */ 

/*         auto sib = first_child; */
/*         while(sib != last_child) */
/*         { */
/*             kid_outputs.push_back((sib->*fn)(d)); */
/*             sib = sib->next_sibling; */
/*         } */
/*         return kid_outputs; */
/*     }; */
/* }; */
/* template<typename T> */
/* struct TreeNode::GetKids<ExecType::Reducer, T> */
/* { */
/*     auto operator()(const Data& d, auto (TreeNode::*fn)(const Data&) ) */
/*     { */
/*         return GetKids<ExecType::Transformer, T>(d, fn); */
/*     }; */
/* }; */

/* template<ExecType E, typename T> */
/* struct TreeNode::GetKidsFit { */
/*     auto operator(const Data& d){ */
/*         return GetKids<E,T>(d, &TreeNode::fit); */
/*     }; */
/* }; */

/* template<ExecType E, typename T> */
/* struct TreeNode::GetKidsPredict { */
/*     auto operator(const Data& d) { */
/*         return GetKids<E,T>(d, &TreeNode::predict); */
/*     }; */
/* }; */
/* //////////////////////////////////////////////////////////////////////////////// */
/* template<> */
/* auto TreeNode::_dispatch(ExecType E, bool train, const Data& d) */
/* { */
/*     switch (E) { */
/*         case ExecType::Unary: */ 
/*             return train? _fit<ExecType::Unary>(d) : _predict<ExecType::Unary>(d); */
/*             break; */
/*         case ExecType::Binary: */
/*             return train? _fit<ExecType::Binary>(d) : _predict<ExecType::Binary>(d); */
/*             break; */
/*         case ExecType::Transformer: */ 
/*             return train? _fit<ExecType::Transformer>(d) : _predict<ExecType::Transformer>(d); */
/*             break; */
/*         case ExecType::Reducer: */ 
/*             return train? _fit<ExecType::Reducer>(d) : _predict<ExecType::Reducer>(d); */
/*             break; */
/*         case ExecType::Applier: */
/*             return train? _fit<ExecType::Applier>(d) : _predict<ExecType::Applier>(d); */
/*             break; */
/*         case ExecType::Splitter: */ 
/*              return train? _fit<ExecType::Splitter>(d) : _predict<ExecType::Splitter>(d); */
/*             break; */
/*         case ExecType::Terminal: */    
/*             return train? _fit<ExecType::Terminal>(d) : _predict<ExecType::Terminal>(d); */
/*             break; */
/*         default: */
/*             HANDLE_ERROR_THROW("ExecType not found"); */
/*     } */
/* }; */
/* template<> template<typename R> */
/* R TreeNode::eval(const Data& d) { */ 
/*     return _dispatch(n.exec_type, true, d); */ 
/* }; */ 
/* template<> template<typename R> */ 
/* R TreeNode::predict(const Data& d) const { return _dispatch(n.exec_type, false, d); }; */ 

}// Brush
#endif
