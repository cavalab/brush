#ifndef OPERATOR_H
#define OPERATOR_H

#include "../init.h"
#include "tree_node.h"
#include "../util/utils.h"
using Util::is_tuple;
/* #include "data/timeseries.h" */
/* using TreeNode = class tree_node_<Node>; */ 

namespace Brush{
///////////////////////////////////////////////////////////////////////////////////////
// Operator class

/// @brief Core computation of a node's function to data. 
/// @tparam S the signature of the node 
/// @tparam E unused 
/// @tparam NT node type
/// @tparam Fit true: fit, false: predict
template<NodeType NT, typename S, bool Fit, typename E=void> 
struct Operator 
{
    using ArgTypes = typename S::ArgTypes;
    using RetType = typename S::RetType;
    static constexpr size_t ArgCount = S::ArgCount;
    // get arg types from tuple by index
    template <std::size_t N>
    using NthType = typename S::NthType<N>; 
    // set weight type
    using W = typename S::WeightType; 
    
    static constexpr auto F = [](const auto& ...args) { 
        Function<NT> f; 
        return f(args...); 
    }; 
    /* static constexpr auto F = [](const auto ...args){ Function<NT> f{}; return f(args...); }; */ 

    Operator() = default;
    ////////////////////////////////////////////////////////////////////////////////
    /// Utilities to grab child outputs.

    // get a std::array of kids
    template<typename T=ArgTypes> requires(!is_tuple<T>::value) 
    T get_kids(const Dataset& d, TreeNode& tn, const W** weights=nullptr) const
    {
        T child_outputs;
        using arg_type = typename T::value_type;

        TreeNode* sib = tn.first_child;
        for (int i = 0; i < ArgCount; ++i)
        {
            if (sib == nullptr)
                HANDLE_ERROR_THROW("bad sibling ptr in get kids");
            if constexpr (Fit)
                child_outputs.at(i) = sib->fit<arg_type>(d);
            else
                child_outputs.at(i) = sib->predict<arg_type>(d, weights);
            sib = sib->next_sibling;
        }
        return child_outputs;
    };


    // gets one kid for a tuple of kids
    template<int I>
    NthType<I> get_kid(const Dataset& d,TreeNode& tn, const W** weights ) const
    {
        TreeNode* sib = tn.first_child; 
        for (int i = 0 ; i < I; ++i)
            sib = sib->next_sibling;
        if constexpr(Fit)
            return sib->fit<NthType<I>>(d);
        else
            return sib->predict<NthType<I>>(d,weights);
    };

    /**
     * @brief Makes and returns a tuple of child outputs
     * 
     * @tparam T: a tuple  
     * @tparam Is: integer sequence 
     * @param d : dataset
     * @param tn : a tree node
     * @return a tuple with elements corresponding to each child node
     */
    template<typename T, size_t ...Is> requires(is_tuple<T>::value) 
    T get_kids_seq(const Dataset& d, TreeNode& tn, const W** weights, std::index_sequence<Is...>) const 
    { 
        return std::make_tuple(get_kid<Is>(d,tn,weights)...);
    };

    // get a std::tuple of kids
    template<typename T=ArgTypes> requires(is_tuple<T>::value) 
    T get_kids(const Dataset& d, TreeNode& tn, const W** weights=nullptr) const
    {
        return get_kids_seq<T>(d, tn, weights, std::make_index_sequence<ArgCount>{});
    };

    ////////////////////////////////////////////////////////////////////////////////
    // apply weights
    template<typename T=ArgTypes>
    requires(!is_tuple<T>::value && is_one_of_v<typename T::value_type::Scalar,float,fJet>) 
    void apply_weights(T& inputs, const Node& n, const W** weights=nullptr) const
    {
        /**
         * @brief applies weights from n.W to inputs. 
         * 
         * @tparam T: some floating point thing
         * @param inputs: arguments to the operator
         * @param n: the node with weights
         */
        if (weights != nullptr)
        {
            for (int i = 0; i < inputs.size(); ++i)
            {
                if (weights == nullptr || *weights == nullptr)
                    HANDLE_ERROR_THROW("weights = nullptr\n");

                inputs[i] = inputs[i] * (**weights);
                // increment weight pointer
                *weights = *weights+1;
            }
        }
        else 
        {
            std::transform(
                inputs.begin(),
                inputs.end(),
                n.W.begin(),
                inputs.begin(),
                std::multiplies<>());
        }
    };
    ///////////////////////////////////////////////////////////////////////////
    /// evaluate operator on array of arguments
    template<typename T=ArgTypes>
    requires (!is_tuple<T>::value && is_one_of_v<typename T::value_type::Scalar,float,fJet>) 
    RetType eval(const Dataset& d, TreeNode& tn, const W** weights=nullptr) const
    {
        auto inputs = get_kids(d, tn, weights);
        if (tn.data.is_weighted)
            this->apply_weights(inputs, tn.data, weights);
        return std::apply(F, inputs);
    };

    /// evaluate operator on tuple of arguments
    template<typename T=ArgTypes>
    requires( is_tuple<T>::value || !is_one_of_v<typename T::value_type::Scalar,float,fJet>) 
    RetType eval(const Dataset& d, TreeNode& tn, const W** weights=nullptr) const
    {
        auto inputs = get_kids(d, tn);
        return std::apply(F, inputs);
    };
};
//////////////////////////////////////////////////////////////////////////////////
/// Terminal Overload
template<typename S, bool Fit>
struct Operator<NodeType::Terminal, S, Fit>
{
    using RetType = typename S::RetType;
    using W = typename S::WeightType; 

    template<typename T=RetType> 
        requires (is_one_of_v<typename T::Scalar,bool,int,float>)
    RetType eval(const Dataset& d, const TreeNode& tn, const W** weights=nullptr) const 
    { 
        if (std::holds_alternative<RetType>(d[tn.data.feature]))
            return std::get<RetType>(d[tn.data.feature]);

        HANDLE_ERROR_THROW(fmt::format("Failed to return type {} for '{}'\n",
            DataTypeEnum<RetType>::value,
            tn.data.feature
        ));

        return RetType(); 
    };
    template <typename T = RetType>
        requires( is_one_of_v<typename T::Scalar, bJet, iJet, fJet>)
    RetType eval(const Dataset &d, const TreeNode &tn, const W **weights = nullptr) const
    {
        return std::visit(
            [](const auto &&arg) -> T
            {
                using ArgType = std::decay_t<decltype(arg)>;
                using Scalar = typename T::Scalar;
                if constexpr (std::is_same_v<Jetify_t<ArgType>,T>)
                {
                    return arg.template cast<Scalar>();
                }
                else
                {
                    auto msg = fmt::format("Failed trying to cast {} to {}\n",
                        DataTypeEnum<ArgType>::value,
                        DataTypeEnum<T>::value
                    );
                    HANDLE_ERROR_THROW(msg);
                }
                return T();
            },
            d[tn.data.feature]
        );
    };
};
//////////////////////////////////////////////////////////////////////////////////
// Constant Overloads
template<typename S, bool Fit> 
struct Operator<NodeType::Constant, S, Fit>
{
    using RetType = typename S::RetType;
    using W = typename S::WeightType;

    template<typename T=RetType, typename Scalar=T::Scalar, int N=T::NumDimensions> 
    RetType eval(const Dataset& d, TreeNode& tn, const W** weights=nullptr) const 
    { 
        Scalar w;
        if (weights == nullptr)
        {
            w = Scalar(tn.data.W.at(0));
        }
        else
        {
            if constexpr (is_same_v<Scalar, W>) 
                w = **weights;
            else if constexpr (is_same_v<Scalar, iJet> && is_same_v<W, fJet>)  {
                using WScalar = typename Scalar::Scalar;
                WScalar tmp = WScalar((**weights).a);    
                w = Scalar(tmp);
            }
            else            
                w = Scalar(**weights);
            *weights++;
        }
        if constexpr (N == 1)
            return RetType::Constant(d.get_n_samples(), w); 
        else
            return RetType::Constant(d.get_n_samples(), d.get_n_features(), w); 
    };
};

////////////////////////////////////////////////////////////////////////////
// Operator overloads 
//  Split
#include "split.h"
////////////////////////////////////////////////////////////////////////////
// Dispatch functions
template<typename R, NodeType NT, typename S, bool Fit, typename W> 
R DispatchOp(const Dataset& d, TreeNode& tn, const W** weights) 
{
    const auto op = Operator<NT,S,Fit>{};
    return op.eval(d, tn, weights);
};

template<typename R, NodeType NT, typename S, bool Fit> 
R DispatchFitOp(const Dataset& d, TreeNode& tn) 
{
    const auto op = Operator<NT,S,Fit>{};
    return op.eval(d, tn);
};

} // Brush

#endif
