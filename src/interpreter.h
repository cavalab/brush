/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/

#ifndef INTERPRETER_H
#define INTERPRETER_H

#include "data/data.h"
#include "nodemap.h"
#include "operators.h"

namespace Brush{

/* struct DispatchTable { */

/*     template<typename... Args> */
/*     auto apply(const NodeType NT, Args... args) */
/*     { */
/*         return Function<NT>(args); */ 
/*     } */
/* }; */
extern DispatchTable dispatch_table; 
// source: Operon
template<typename... Ts>
struct DispatchTable {
    /* template<typename T> */
    /* using Callable = Function<T>; */
    template<typename T>
    using Callable = typename std::function<void(Operon::Vector<Array<T>>&, Operon::Vector<Node> const&, size_t, Operon::Range)>;
    // should be the type of a DispatchOp fn call 
    using Callable = typename std::function<void(Operon::Vector<Array<T>>&, Operon::Vector<Node> const&, size_t, Operon::Range)>;

    using Tuple    = std::tuple<Callable<Ts>...>;
    /* using Map      = robin_hood::unordered_flat_map<Operon::Hash, Tuple>; */
    using Map      = std::unordered_flat_map<NodeType, Callable>;

private:
    Map map_;

    template<std::size_t... Is>
    void InitMap(std::index_sequence<Is...> /*unused*/)
    {
        auto f = [](auto i) { return static_cast<NodeType>(1U << i); };
        (map_.insert({ Node(f(Is)).node_type, detail::MakeTuple<f(Is), Ts...>() }), ...);
    }

public:
    DispatchTable()
    {
        /* InitMap(std::make_index_sequence<NodeTypes::Count>{}); // exclude constant, variable, dynamic */
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
    inline auto Get(NodeType const h) -> Callable<T>&
    {
        return const_cast<Callable<T>&>(const_cast<DispatchTable<Ts...> const*>(*this)->Get(h)); // NOLINT
    }

    template<typename T>
    [[nodiscard]] inline auto Get(NodeType const h) const -> Callable<T> const&
    {
        constexpr int64_t idx = detail::tuple_index<Callable<T>, Tuple>::value;
        static_assert(idx >= 0, "Tuple does not contain type T");
        if (auto it = map_.find(h); it != map_.end()) {
            return std::get<static_cast<size_t>(idx)>(it->second);
        }
        throw std::runtime_error(fmt::format("Hash value {} is not in the map\n", h));
    }

    template<typename F>
    void RegisterCallable(NodeType hash, F&& f) {
        map_[hash] = detail::MakeTuple<F, Ts...>(std::forward<F&&>(f));
    }

    template<typename T>
    [[nodiscard]] inline auto TryGet(NodeType const h) const noexcept -> std::optional<Callable<T>>
    {
        constexpr int64_t idx = detail::tuple_index<Callable<T>, Tuple>::value;
        static_assert(idx >= 0, "Tuple does not contain type T");
        if (auto it = map_.find(h); it != map_.end()) {
            return { std::get<static_cast<size_t>(idx)>(it->second) };
        }
        return {};
    }

    [[nodiscard]] auto Contains(NodeType hash) const noexcept -> bool { return map_.contains(hash); }
};


}
#endif
