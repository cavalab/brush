/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3

Dispatch class design heavily inspired by Operon, (c) Heal Research
https://github.com/heal-research/operon/
*/

#ifndef INTERPRETER_H
#define INTERPRETER_H

#include "data/data.h"
#include "nodemap.h"
#include "node.h"
#include <optional>
#include <cstddef>
#include <tuple>


// forward declarations
template<typename T> class tree_node_;
using TreeNode = class tree_node_<Node>; 

namespace Brush{

// forward declarations
template<typename R, NodeType NT, typename S, bool Fit> R DispatchOp(
    const Data::Dataset& d, 
    TreeNode& tn
); 

////////////////////////////////////////////////////////////////////////////////
// Dispatch Table
template<bool Fit>
struct DispatchTable
{
    
    template<typename T>
    using Callable = typename std::function<T(const Data::Dataset&, TreeNode&)>;

    using CallVariant = std::variant< 
        Callable<ArrayXb>,
        Callable<ArrayXi>, 
        Callable<ArrayXf>, 
        Callable<ArrayXXb>,
        Callable<ArrayXXi>, 
        Callable<ArrayXXf>, 
        Callable<TimeSeriesb>,
        Callable<TimeSeriesi>,
        Callable<TimeSeriesf>
        >;
    // map (fit/predict)->(node type)->(signature hash)->(dispatch function)
    using SigMap = std::unordered_map<std::size_t,CallVariant>;
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
        /* constexpr auto signatures = Signatures<NT>::value; */
        using signatures = typename Signatures<NT>::type;
        
        /* return AddOperator<NT, decltype(signatures)>( */ 
        return AddOperator<NT, signatures>( 
                     std::make_index_sequence<std::tuple_size_v<signatures>>()
                     );
    } 

    // template<NodeType NT, typename Sigs, std::size_t... Is>
    // static constexpr auto AddOperator(std::index_sequence<Is...>)
    // {
    //     SigMap sm;
    //     (sm.insert({std::tuple_element_t<Is, Sigs>::hash(), 
    //                 MakeCallable<NT, std::tuple_element_t<Is, Sigs>>()}), ...);
    //     return sm;
    // }
    //TEST WITHOUT CONSTEXPR FOR CLANG:
    template<NodeType NT, typename Sigs, std::size_t... Is>
    static auto AddOperator(std::index_sequence<Is...>)
    {
        SigMap sm;
        (sm.insert({std::tuple_element_t<Is, Sigs>::hash(), 
                    MakeCallable<NT, std::tuple_element_t<Is, Sigs>>()}), ...);
        return sm;
    }

    template<NodeType N, typename S>
    static constexpr auto MakeCallable()  
    {
        using R = typename S::RetType;
        return Callable<R>(DispatchOp<R,N,S,Fit>);
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
        if (map_.at(n).find(sig_hash) == map_.at(n).end())
        {
            string err;
            err += fmt::format("{} not in map_.at({})\n",sig_hash,n);
            err += fmt::format("options:\n");
            for (auto [k, v]: map_.at(n))
                err+= fmt::format("{}\n", k);
            HANDLE_ERROR_THROW(err); 
        }
        try {
            return std::get<Callable<T>>(map_.at(n).at(sig_hash));
        }
        catch(const std::bad_variant_access& e) {
            auto msg = fmt::format("{}",e.what());
            HANDLE_ERROR_THROW(msg);
        }
        return std::get<Callable<T>>(map_.at(n).at(sig_hash));
    }

};

extern DispatchTable<true> dtable_fit;
extern DispatchTable<false> dtable_predict;
} // namespace Brush
#endif
