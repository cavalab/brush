/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/

#ifndef INTERPRETER_H
#define INTERPRETER_H

#include "data/data.h"
#include "nodemap.h"
#include "node.h"
/* #include "operator.h" */
/* #include "dispatch_callables.h" */
/* #include "tree_node.h" */
#include <optional>
#include <cstddef>
#include <tuple>


namespace Brush{

// forward declarations
template<typename T> class tree_node_;
using TreeNode = class tree_node_<Node>; 
template<typename R, NodeType NT, typename S, bool Fit> R DispatchOp(const Data& d, TreeNode& tn); 

////////////////////////////////////////////////////////////////////////////////
// Dispatch Table
template<bool Fit>
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
    using Callable = typename std::function<T(const Data&,TreeNode&)>;

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

    template<NodeType NT, typename Sigs, std::size_t... Is>
    static constexpr auto AddOperator(std::index_sequence<Is...>)
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
                        fmt::print("{}: {}:DispatchFit\n",nt, sig);
                    else
                        fmt::print("{}: {}:DispatchPredict\n",nt, sig);
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
        return std::get<Callable<T>>(map_.at(n).at(sig_hash));
    }

};

extern DispatchTable<true> dtable_fit;
extern DispatchTable<false> dtable_predict;
} // namespace Brush
#endif
