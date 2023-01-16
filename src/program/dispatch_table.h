/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3

Dispatch class design heavily inspired by Operon, (c) Heal Research
https://github.com/heal-research/operon/
*/

#ifndef DISPATCH_TABLE_H
#define DISPATCH_TABLE_H

#include "../init.h"
#include "../data/data.h"
#include "nodetype.h"
#include "node.h"
#include <optional>
#include <cstddef>
#include <tuple>


// forward declarations
template<typename T> class tree_node_;
using TreeNode = class tree_node_<Node>; 

namespace Brush{

// forward declarations
template<typename R, NodeType NT, typename S, bool Fit, typename W> 
R DispatchOp( const Data::Dataset& d, TreeNode& tn, const W** weights); 
template<typename R, NodeType NT, typename S, bool Fit> 
R DispatchOp( const Data::Dataset& d, TreeNode& tn); 

////////////////////////////////////////////////////////////////////////////////
// Dispatch Table
template<bool Fit>
struct DispatchTable
{
    template<typename T>
    using Callable = typename std::conditional_t<Fit, 
        std::function<T(const Data::Dataset&, TreeNode&)>,
        std::function<T(const Data::Dataset&, TreeNode&, const typename WeightType<T>::type**)>
    >;

    using CallVariant = std::variant< 
        Callable<ArrayXb>,
        Callable<ArrayXi>, 
        Callable<ArrayXf>, 
        Callable<ArrayXXb>,
        Callable<ArrayXXi>, 
        Callable<ArrayXXf>, 
        Callable<TimeSeriesb>,
        Callable<TimeSeriesi>,
        Callable<TimeSeriesf>,
        // jet overloads
        Callable<ArrayXbJet>,
        Callable<ArrayXiJet>, 
        Callable<ArrayXfJet>, 
        Callable<ArrayXXbJet>,
        Callable<ArrayXXiJet>, 
        Callable<ArrayXXfJet>, 
        Callable<Data::TimeSeriesbJet>,
        Callable<Data::TimeSeriesiJet>,
        Callable<Data::TimeSeriesfJet>
        >;
    /// @brief maps Signature hashes -> Dispatch Operator
    using SigMap = std::unordered_map<std::size_t,CallVariant>;
    /// @brief maps NodeTypes -> Signature hash -> Dispatch Operator 
    using DTMap = std::unordered_map<NodeType, SigMap>;

private:
    DTMap map_;

    template<std::size_t... Is>
    void InitMap(std::index_sequence<Is...> /*unused*/)
    {
        
        //TODO: nt(Is) should be a hash, if want to register other functions
        auto nt = [](auto i) { return static_cast<NodeType>(1UL << i); };
        (map_.insert({ nt(Is), MakeOperators<nt(Is)>() }), ...);
    }

    template<NodeType NT>
    SigMap MakeOperators()  
    {
        using signatures = typename Signatures<NT>::type;
        return AddOperator<NT, signatures>( 
                     std::make_index_sequence<std::tuple_size_v<signatures>>()
                     );
    } 

    template<NodeType NT, typename Sigs, std::size_t... Is>
    static constexpr auto AddOperator(std::index_sequence<Is...>)
    {
        SigMap sm;
        (sm.insert({std::tuple_element_t<Is, Sigs>::hash(), 
            MakeCallable<NT, std::tuple_element_t<Is, Sigs>>()}), ...);
        // Add dual signatures that take Jet types
        if constexpr (is_in_v<NT, NodeType::ArgMax, NodeType::Count>){
            (sm.insert({std::tuple_element_t<Is, Sigs>::DualArgs::hash(), 
                MakeCallable<NT, typename std::tuple_element_t<Is, Sigs>::DualArgs>()}), ...);
        }
        else {
            (sm.insert({std::tuple_element_t<Is, Sigs>::Dual::hash(), 
                MakeCallable<NT, typename std::tuple_element_t<Is, Sigs>::Dual>()}), ...);
        }
        return sm;
    }


    template<NodeType N, typename S>
    static constexpr auto MakeCallable()  
    {
        using R = typename S::RetType;
        using W = typename S::WeightType;
        if constexpr (Fit)
            return Callable<R>(DispatchOp<R,N,S,Fit>);
        else
            return Callable<R>(DispatchOp<R,N,S,Fit,W>);
    }

public:
    DispatchTable()
    {
        InitMap(std::make_index_sequence<NodeTypes::Count>{}); 
    }

    void print()
    {
        fmt::print("================== \n");
        fmt::print("dispatch table map_: \n");
        for (const auto& [nt, sigmap]: map_){
                for (const auto& [sig, call]: sigmap){
                    if (Fit)
                        fmt::print("{} : {} : DispatchFit\n",nt, sig);
                    else
                        fmt::print("{} : {} : DispatchPredict\n",nt, sig);
                }
            }
        fmt::print("================== \n");
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
    inline auto Get(NodeType n, std::size_t sig_hash) const -> Callable<T> const&
    {
        // fmt::print("get<Callable<{}>> for {} with hash {}\n",
        //     DataTypeEnum<T>::value, n, sig_hash
        // );
        if (map_.at(n).find(sig_hash) == map_.at(n).end())
        {
            string err;
            err += fmt::format("sig_hash={} not in map_.at({})\n",sig_hash,n);
            err += fmt::format("options:\n");
            for (auto [k, v]: map_.at(n))
                err+= fmt::format("{}\n", k);
            HANDLE_ERROR_THROW(err); 
        }
        // CallVariant callable = map_.at(n).at(sig_hash);
        // try {
        if (std::holds_alternative<Callable<T>>(map_.at(n).at(sig_hash)))
            return std::get<Callable<T>>(map_.at(n).at(sig_hash));
        // }
        // catch(const std::bad_variant_access& e) {

            // auto msg = fmt::format("{}\nTried to ",e.what()); HANDLE_ERROR_THROW(msg);
        // }
        else{
            // if (map_.at(n).size() > 1){
            //     for (const auto & kv : map_.at(n))
            //     {
            //         if (std::holds_alternative<Callable<T>>(kv.second))
            //             return std::get<Callable<T>>(kv.second);
            //     }
            // }
            auto msg = fmt::format("Tried get<Callable<{}>> for {} with hash {}; failed"
            " because map holds index {}\n",
                DataTypeEnum<T>::value, n, sig_hash, map_.at(n).at(sig_hash).index() 
            );
            HANDLE_ERROR_THROW(msg);
        }
        return std::get<Callable<T>>(map_.at(n).at(sig_hash));
    }

};

extern DispatchTable<true> dtable_fit;
extern DispatchTable<false> dtable_predict;
// // format overload 
// template <> struct fmt::formatter<Brush::SearchSpace>: formatter<string_view> {
//   template <typename FormatContext>
//   auto format(const Brush::SearchSpace& SS, FormatContext& ctx) const {
//     string output = "Search Space\n===\n";
//     output += fmt::format("terminal_map: {}\n", SS.terminal_map);
//     output += fmt::format("terminal_weights: {}\n", SS.terminal_weights);
//     for (const auto& [ret_type, v] : SS.node_map) {
//         for (const auto& [args_type, v2] : v) {
//             for (const auto& [node_type, node] : v2) {
//                 output += fmt::format("node_map[{}][{}][{}] = {}, weight = {}\n", 
//                         ret_type,
//                         ArgsName[args_type],
//                         node_type,
//                         node,
//                         SS.weight_map.at(ret_type).at(args_type).at(node_type)
//                         );
//             }
//         }
//     }
//     output += "===";
//     return formatter<string_view>::format(output, ctx);
//   }
// };
} // namespace Brush
#endif
