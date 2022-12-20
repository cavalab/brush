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
/* template<NodeType NT, typename S, bool Fit, typename T=void> Operator; */

/* template <typename T> */
/* template <typename> struct is_tuple: std::false_type {}; */
/* template <typename ...T> struct is_tuple<std::tuple<T...>>: std::true_type {}; */
/* using is_tuple_v = typename is_tuple<T>::value; */


/// @brief 
/// @tparam S 
/// @tparam E 
/// @tparam NT 
/// @tparam Fit 
template<NodeType NT, typename S, bool Fit, typename E=void> 
struct Operator 
{
    using ArgTypes = typename S::ArgTypes;
    using RetType = typename S::RetType;
    static constexpr size_t ArgCount = S::ArgCount;
    // get arg types from tuple by index
    template <std::size_t N>
    using NthType = typename S::NthType<N>; 
    
    static constexpr auto F = [](const auto& ...args) -> RetType { 
        Function<NT> f; 
        return f(args...); 
    }; 
    /* static constexpr auto F = [](const auto ...args){ Function<NT> f{}; return f(args...); }; */ 

    Operator() = default;
    ////////////////////////////////////////////////////////////////////////////////
    /// Utilities to grab child outputs.

    // get a std::array of kids
    template<typename T=ArgTypes>
    enable_if_t<!is_tuple<T>::value, T> 
    get_kids(const Dataset& d, TreeNode& tn) const
    {
        T child_outputs;
        using arg_type = typename T::value_type;

        TreeNode* sib = tn.first_child;
        for (int i = 0; i < ArgCount; ++i)
        {
            if (sib == nullptr)
                HANDLE_ERROR_THROW("bad sibling ptr in get kids");
            if constexpr (Fit)
                child_outputs.at(i) = sib->fit<arg_type>(d) ;
            else
                child_outputs.at(i) = sib->predict<arg_type>(d);
            sib = sib->next_sibling;
        }
        return child_outputs;
    };


    // gets one kid for a tuple of kids
    template<int I>
    NthType<I> get_kid(const Dataset& d,TreeNode& tn ) const
    {
        TreeNode* sib = tn.first_child; 
        for (int i = 0 ; i < I; ++i)
            sib = sib->next_sibling;
        if constexpr(Fit)
            return sib->fit<NthType<I>>(d);
        else
            return sib->predict<NthType<I>>(d);
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
    template<typename T, size_t ...Is>
    enable_if_t<is_tuple<T>::value, T> 
    get_kids_seq(const Dataset& d, TreeNode& tn, std::index_sequence<Is...>) const 
    { 
        return std::make_tuple(get_kid<Is>(d,tn)...);
    };

    // get a std::tuple of kids
    template<typename T=ArgTypes>
    enable_if_t<is_tuple<T>::value, T> 
    get_kids(const Dataset& d, TreeNode& tn) const
    {
        return get_kids_seq<T>(d, tn, std::make_index_sequence<ArgCount>{});
    };

    ////////////////////////////////////////////////////////////////////////////////
    template<typename T=ArgTypes>
    enable_if_t<!is_tuple<T>::value && is_same_v<typename T::value_type::Scalar,float>,void> 
    apply_weights(T& inputs, const Node& n) const
    {
        /**
         * @brief applies weights from n.W to inputs. 
         * 
         * @tparam T: some floating point thing
         * @param inputs: arguments to the operator
         * @param n: the node with weights
         */
        std::transform(
                    inputs.begin(), 
                    inputs.end(),
                    n.W.begin(),
                    inputs.begin(), 
                    std::multiplies<>()
                    );
    };
    ///////////////////////////////////////////////////////////////////////////
    /// evaluate operator on array of arguments
    template<typename T=ArgTypes>
    enable_if_t<!is_tuple<T>::value && is_same_v<typename T::value_type::Scalar,float>, RetType> 
    eval(const Dataset& d, TreeNode& tn) const
    {
        ArgTypes inputs = get_kids(d, tn);
        if (tn.data.is_weighted)
            this->apply_weights(inputs, tn.data);
        RetType out = std::apply(F, inputs);
        return out;
    };

    /// evaluate operator on tuple of arguments
    template<typename T=ArgTypes>
    enable_if_t<is_tuple<T>::value || !is_same_v<typename T::value_type::Scalar,float>, RetType> 
    eval(const Dataset& d, TreeNode& tn) const
    {
        ArgTypes inputs = get_kids(d, tn);
        RetType out = std::apply(F, inputs);
        return out;
    };
};
//////////////////////////////////////////////////////////////////////////////////
/// Terminal Overload
template<typename S, bool Fit>
struct Operator<NodeType::Terminal, S, Fit>
{
    using RetType = typename S::RetType;
    RetType eval(const Dataset& d, const TreeNode& tn) const { 
        // fmt::print("run std::get<{}>(d[{}])\n", DataTypeEnum<RetType>::value, tn.data.feature); 
        RetType out;
        try {
            out = std::get<RetType>(d[tn.data.feature]);
        }
        catch(const std::bad_variant_access& e) {
            HANDLE_ERROR_THROW(fmt::format("{}",e.what()));
        }

        return out; 
    };
};
//////////////////////////////////////////////////////////////////////////////////
// Constant Overloads
template<typename S, bool Fit> 
struct Operator<NodeType::Constant, S, Fit>
{
    using RetType = typename S::RetType;

    template<typename T=RetType> requires same_as<T, ArrayXf>
    RetType eval(const Dataset& d, TreeNode& tn) const { 
        return RetType::Constant(d.get_n_samples(), tn.data.W.at(0)); 
    };

    template<typename T=RetType> requires same_as<T, ArrayXi>
    RetType eval(const Dataset& d, TreeNode& tn) const { 
        return RetType::Constant(d.get_n_samples(), int(tn.data.W.at(0))); 
    };

    template<typename T=RetType> requires same_as<T, ArrayXb>
    RetType eval(const Dataset& d, TreeNode& tn) const { 
        return RetType(d.get_n_samples()); 
    };

    template<typename T=RetType> requires same_as<T, ArrayXXf>
    RetType eval(const Dataset& d, TreeNode& tn) const { 
        return RetType::Constant(d.get_n_samples(), d.get_n_features(), tn.data.W.at(0)); 
    };

    template<typename T=RetType> requires same_as<T, ArrayXXb>
    RetType eval(const Dataset& d, TreeNode& tn) const { 
        return RetType(d.get_n_samples(), d.get_n_features());
    };
};

////////////////////////////////////////////////////////////////////////////
// Operator overloads 
//  Split
#include "split.h"
////////////////////////////////////////////////////////////////////////////
// fit and predict Dispatch functions
template<typename R, NodeType NT, typename S, bool Fit> 
R DispatchOp(const Dataset& d, TreeNode& tn) 
{
    // fmt::print("DispatchOp: Dispatching {} with signature {} of {}\n",
    //     NT, 
    //     S::get_args_type(),
    //     S::get_arg_types()
    // );

    const auto op = Operator<NT,S,Fit>{};
    R out = op.eval(d, tn);
    // if(out.size()==0)
    //     cout << "out empty\n";

    // if constexpr (is_same_v<R,ArrayXf>)
    //     fmt::print("{} returning {}\n",NT, out);
    /* cout << NT << " output: " << out << endl; */
    return out;
    /* return op.eval(d,tn); */
};

} // Brush

#endif
