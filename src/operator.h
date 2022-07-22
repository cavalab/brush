#ifndef OPERATOR_H
#define OPERATOR_H

#include "tree_node.h"
/* using TreeNode = class tree_node_<Node>; */ 

namespace Brush{
///////////////////////////////////////////////////////////////////////////////////////
// Operator class
template<NodeType NT, typename S, bool Fit> 
struct Operator 
{
    // TODO
    using Args = typename S::ArgTypes;
    using RetType = typename S::RetType;
    static constexpr size_t ArgCount = S::ArgCount;
    // get arg types from tuple by index
    template <std::size_t N>
    using NthType = typename S::NthType<N>; 
    
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

    // get a std::array of kids
    template<typename T=Args>
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


    // tuple get kids
    template<int I>
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

    template<typename T, size_t ...Is>
    requires (!is_array_v<T>)
    auto get_kids_seq(const Data& d, TreeNode& tn, std::index_sequence<Is...>) const 
    { 
        return std::make_tuple(get_kid<Is>(d,tn)...);
    };

    // get a std::tuple of kids
    template<typename T=Args>
    requires (!is_array_v<T>)
    auto get_kids(const Data& d, TreeNode& tn) const
    {
        return get_kids_seq<T>(d, tn, std::make_index_sequence<ArgCount>{});
    };

    ///////////////////////////////////////////////////////////////////////////
    // fit and predict
    template<typename T=Args>
    requires (is_array_v<T>)
    auto eval(const Data& d, TreeNode& tn) const
    {
        auto inputs = get_kids(d, tn);
        if (tn.n.is_weighted)
            this->apply_weights(inputs, tn.n);
        return std::apply(F, inputs);
    };

    template<typename T=Args>
    requires (!is_array_v<T>)
    auto eval(const Data& d, TreeNode& tn) const
    {
        auto inputs = get_kids(d, tn);
        return std::apply(F, inputs);
    };

    auto fit(const Data& d, TreeNode& tn) const { return eval<true>(d,tn); };

    auto predict(const Data& d, TreeNode& tn) const { return eval<false>(d,tn); };
};
//////////////////////////////////////////////////////////////////////////////////
/// Terminal Overload
template<typename S, bool Fit>
struct Operator<NodeType::Terminal, S, Fit>
{
    using RetType = typename S::RetType;
    auto eval(const Data& d, TreeNode& tn) const { return std::get<RetType>(d[tn.n.feature]); };
};
//////////////////////////////////////////////////////////////////////////////////
// Constant Overloads
template<typename S, bool Fit> 
struct Operator<NodeType::Constant, S, Fit>
{
    using RetType = typename S::RetType;

    template<typename T=RetType> requires same_as<T, ArrayXf>
    auto eval(const Data& d, TreeNode& tn) const { 
        return RetType::Constant(d.n_samples, tn.n.W.at(0)); 
    };

    template<typename T=RetType> requires same_as<T, ArrayXi>
    auto eval(const Data& d, TreeNode& tn) const { 
        return RetType::Constant(d.n_samples, int(tn.n.W.at(0))); 
    };

    template<typename T=RetType> requires same_as<T, ArrayXb>
    auto eval(const Data& d, TreeNode& tn) const { 
        return RetType(d.n_samples); 
    };

    template<typename T=RetType> requires same_as<T, ArrayXXf>
    auto eval(const Data& d, TreeNode& tn) const { 
        return RetType::Constant(d.n_samples, d.n_features, tn.n.W.at(0)); 
    };

    template<typename T=RetType> requires same_as<T, ArrayXXb>
    auto eval(const Data& d, TreeNode& tn) const { 
        return RetType(d.n_samples, d.n_features);
    };
    
    auto fit(const Data& d, TreeNode& tn) const { return eval(d,tn); };
    auto predict(const Data& d, TreeNode& tn) const { return eval(d,tn); };
};

//////////////////////////////////////////////////////////////////////////////////
// Split Node Overloads

////////////////////////////////////////////////////////////////////////////
// fit and predict Dispatch functions
template<typename R, NodeType NT, typename S, bool Fit> //, typename ...Args>
R DispatchOp(const Data& d, TreeNode& tn) 
{
    const auto op = Operator<NT,S,Fit>{};
    return op.eval(d, tn);
};

} // Brush

#endif
