/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/

#ifndef INTERPRETER_H
#define INTERPRETER_H

#include "data/data.h"
#include "nodemap.h"
#include "node.h"
#include "operators.h"
#include "tree_node.h"
// from operon
/* #include <fmt/core.h> */
#include <optional>
/* #include <robin_hood.h> */
#include <cstddef>
#include <tuple>

/* #include "operon/core/node.hpp" */
/* #include "operon/core/range.hpp" */
/* #include "operon/core/types.hpp" */
/* #include "functions.hpp" */

namespace Brush{
    // forward declaration
    template<class T> class tree_node_; 
    using TreeNode = class tree_node_<Node>; 

namespace detail {
    /* class tree_node_<Node> */
    // dispatching mechanism
    // stolen from Operon 
    /* template<NodeType Type, typename R, typename... Args> */
    template<NodeType Type>
    State DispatchOpUnary(Data& d, TreeNode& n)
    {
        return Function<Type>{}(n.first->eval(d));
    }
    /* template<NodeType Type, typename R> */
    template<NodeType Type>
    State DispatchOpBinary(Data& d, TreeNode& n)
    {
        return Function<Type>{}(n.first->eval(d), n.last->eval(d));
    }

    /* template<NodeType Type, typename R> */
    template<NodeType Type>
    State DispatchTerminal(Data& d, TreeNode& n)
    {
        using T = decltype(DataMap[n.ret_type]);
        return std::get<T>(d.features[n.feature]);
    }


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

    using Callable = typename std::function<State(Data&,TreeNode&)>;

    template<NodeType Type, ExecType E>
    static constexpr auto MakeCall() -> 
    {
        switch (E) {
            case ExecType::Unary: 
                return Callable(detail::DispatchOpUnary<Type>);
                break;
            case ExecType::Binary:
                return Callable(detail::DispatchOpBinary<Type>);
                break;
            //TODO
            /* case ExecType::Transformer: */ 
            /*     break; */
            /* case ExecType::Reducer: */ 
            /*     break; */
            /* case ExecType::Applier: */
            /*     break; */
            /* case ExecType::Splitter: */ 
            /*     break; */
            /* case ExecType::Terminal: */    
                /* return train? _fit<ExecType::Terminal>(d) : _predict<ExecType::Terminal>(d); */
                /* break; */
            default:
                HANDLE_ERROR_THROW("ExecType not found");
        };
    }

    /* template<NodeType Type, typename... Ts, std::enable_if_t<sizeof...(Ts) != 0, bool> = true> */
    template<NodeType Type, ExecType E> 
    static constexpr auto MakeTuple()
    {
        /* return std::make_tuple(MakeCall<Type, Ts>()...); */
        return std::make_tuple(MakeCall<Type, E>());
    };

    /* template<typename F, typename... Ts, std::enable_if_t<sizeof...(Ts) != 0 && (std::is_invocable_r_v<void, F, detail::Array<Ts>&, Vector<Node> const&, size_t, Operon::Range> && ...), bool> = true> */
    /* static constexpr auto MakeTuple(F&& f) */
    /* { */
    /*     return std::make_tuple(Callable<Ts>(std::forward<F&&>(f))...); */
    /* } */
} // namespace detail

/* template<typename... Ts> */
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

    /* using Tuple    = std::tuple<Callable<Ts>...>; */
    /* using Map      = robin_hood::unordered_flat_map<Operon::Hash, Tuple>; */
    using Map      = std::unordered_flat_map<NodeType, Tuple>;

private:
    Map map_;

    template<std::size_t... Is>
    void InitMap(std::index_sequence<Is...> /*unused*/)
    {
        auto nt = [](auto i) { return static_cast<NodeType>(1U << i); };
        /* (map_.insert({ Node(f(Is)).HashValue, detail::MakeTuple<f(Is), Ts...>() }), ...); */
        (map_.insert({ nt(Is), detail::MakeTuple<nt(Is),NodeSchema[NodeTypeName[Is]]["ExecType"]>() }), ...);
        //TODO: this really should be a hash, if want to register other functions
    }

public:
    DispatchTable()
    {
        InitMap(std::make_index_sequence<NodeTypes::Count-3>{}); // exclude constant, variable, dynamic
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

    /* template<typename F> */
    /* void RegisterCallable(Operon::Hash hash, F&& f) { */
    /*     map_[hash] = detail::MakeTuple<F, Ts...>(std::forward<F&&>(f)); */
    /* } */

    /* template<typename T> */
    /* [[nodiscard]] inline auto TryGet(Operon::Hash const h) const noexcept -> std::optional<Callable<T>> */
    /* { */
    /*     constexpr int64_t idx = detail::tuple_index<Callable<T>, Tuple>::value; */
    /*     static_assert(idx >= 0, "Tuple does not contain type T"); */
    /*     if (auto it = map_.find(h); it != map_.end()) { */
    /*         return { std::get<static_cast<size_t>(idx)>(it->second) }; */
    /*     } */
    /*     return {}; */
    /* } */

    /* [[nodiscard]] auto Contains(Operon::Hash hash) const noexcept -> bool { return map_.contains(hash); } */
/* }; */

extern DispatchTable dispatch_table; 
} // namespace Brush
#endif
