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


    // tuple get kids
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
     * @brief Get the kids seq object
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
    /* requires (!is_array_v<decay_t<T>>) */
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
        cout << "applying weights to " << n.name << " operator\n";
        std::transform(
                    inputs.begin(), 
                    inputs.end(),
                    n.W.begin(),
                    inputs.begin(), 
                    std::multiplies<>()
                    );
    };
    ///////////////////////////////////////////////////////////////////////////
    // fit and predict
    template<typename T=ArgTypes>
    /* requires (is_array_v<decay_t<T>>) */
    /* enable_if_t<!is_tuple<T>::value, RetType> */ 
    enable_if_t<!is_tuple<T>::value && is_same_v<typename T::value_type::Scalar,float>, RetType> 
    eval(const Dataset& d, TreeNode& tn) const
    {
        // fmt::print("eval::getting kids\n");
        ArgTypes inputs = get_kids(d, tn);
        // fmt::print("eval::applying weights\n");
        if (tn.data.is_weighted)
            this->apply_weights(inputs, tn.data);
        // fmt::print("eval::std::apply F\n");
        RetType out = std::apply(F, inputs);
        // if constexpr (is_same_v<RetType,ArrayXf>)
        //     fmt::print("eval::std::apply result: {}\n",out);
        return out;
    };

    template<typename T=ArgTypes>
    /* requires (!is_array_v<decay_t<T>>) */
    /* enable_if_t<is_tuple<T>::value, RetType> */ 
    enable_if_t<is_tuple<T>::value || !is_same_v<typename T::value_type::Scalar,float>, RetType> 
    eval(const Dataset& d, TreeNode& tn) const
    {
        // fmt::print("eval (tuple)::getting kids\n");
        ArgTypes inputs = get_kids(d, tn);
        // fmt::print("eval (tuple)::apply F, inputs\n");
        RetType out = std::apply(F, inputs);
        /* return std::apply(F, inputs); */
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

#include "split.h"
using namespace Split;

/* template<typename S, bool Fit> */ 
/* requires (is_same_v<NT,NodeType::SplitBest> || is_same_v<NT, NodeType::SplitOn>) */
/* template<> */
/* struct Operator<NodeType::SplitBest,S,Fit> */
/* template<NodeType NT> */
/* concept Splitter = requires(conjunction_v<is_same_v<NT,NodeType::SplitOn>,is_same_v<NT,NodeType::SplitBest>>>) */

template<NodeType NT, typename S, bool Fit> 
struct Operator<NT, S, Fit, enable_if_t<is_one_of_v<NT, NodeType::SplitOn, NodeType::SplitBest>>> 
/* struct Operator< */
{
    using ArgTypes = typename S::ArgTypes;
    using FirstArg = typename S::base::FirstArg;
    using RetType = typename S::RetType;
    static constexpr size_t ArgCount = S::ArgCount;
    // get arg types from tuple by index
    template <std::size_t N>
    using NthType = typename S::NthType<N>; 
    
    /* static constexpr auto F = [](const auto& ...args){ Function<NT> f{}; return f(args...); }; */ 
    static constexpr Function<NT> F{};

    array<RetType,2> get_kids(const array<Dataset, 2>& d, TreeNode& tn) const
    {
        using arg_type = NthType<1>;
        array<arg_type,2> child_outputs;

        TreeNode* sib = tn.first_child;
        if constexpr (NT==NodeType::SplitOn)
            sib = sib->next_sibling;

        cout << "-----> first_child ptr: " << sib << endl;
        for (int i = 0; i < 2; ++i)
        {
            if (d.at(i).get_n_samples() > 0)
            {
                if constexpr (Fit)
                    child_outputs.at(i) = sib->fit<arg_type>(d.at(i));
                else
                    child_outputs.at(i) = sib->predict<arg_type>(d.at(i));
            }
            sib = sib->next_sibling;
            cout << "-----> next sib ptr: " << sib << endl;
        }
        return child_outputs;
    };

    RetType fit(const Dataset& d, TreeNode& tn) const {
        auto& threshold = tn.data.W.at(0);
        auto& feature = tn.data.feature;

        // set feature and threshold
        if constexpr (NT == NodeType::SplitOn)
        {
            // split on first child
            FirstArg split_feature = tn.first_child->fit<FirstArg>(d);
            // get the best splitting threshold
            tie(threshold, ignore) = Split::best_threshold(split_feature, d.y, d.classification);
        }
        else
            tie(feature, threshold) = Split::get_best_variable_and_threshold(d, tn);

        return predict(d, tn);
    }

    RetType predict(const Dataset& d, TreeNode& tn) const 
    {
        const auto& threshold = tn.data.W.at(0);
        const auto& feature = tn.data.feature;

        // split the data
        ArrayXb mask;
        if constexpr (NT==NodeType::SplitBest)
            // mask = Split::threshold_mask(std::get<FirstArg>(d.at(feature)), threshold);
            mask = Split::threshold_mask(d[feature], threshold);
        else {
            auto split_feature = tn.first_child->predict<FirstArg>(d);
            mask = Split::threshold_mask(split_feature, threshold);
        }

        array<Dataset, 2> data_splits = d.split(mask);
        // fmt::print("data_splits sizes: {}, {}\n",
        //         data_splits[0].get_n_samples(), 
        //         data_splits[1].get_n_samples());
        // // if there aren't samples on either side of the split, just return 
        // // one child or the other
        // if (data_splits[0].get_n_samples() == 0)
        // else if (data_splits[1].get_n_samples() == 0)
            
        auto child_outputs = get_kids(data_splits, tn);

        // stitch together outputs
        // fmt::print("stitching outputs\n");
        auto out = Split::stitch(child_outputs, mask);
        /* auto out = mask.select(child_outputs.at(0), child_outputs.at(1)); */
        /* cout << "returning " << std::get<RetType>(out) << endl; */

        return out;
    }
    RetType eval(const Dataset& d, TreeNode& tn) const {
        if constexpr (Fit)
            return fit(d,tn); 
        else
            return predict(d,tn);
    }
};

////////////////////////////////////////////////////////////////////////////
// fit and predict Dispatch functions
template<typename R, NodeType NT, typename S, bool Fit> 
R DispatchOp(const Dataset& d, TreeNode& tn) 
{
    fmt::print("DispatchOp: Dispatching {} with signature {} of {}\n",
        NT, 
        S::get_args_type(),
        S::get_arg_types()
    );

    const auto op = Operator<NT,S,Fit>{};
    R out = op.eval(d, tn);
    if(out.size()==0)
        cout << "out empty\n";

    if constexpr (is_same_v<R,ArrayXf>)
        fmt::print("{} returning {}\n",NT, out);
    /* cout << NT << " output: " << out << endl; */
    return out;
    /* return op.eval(d,tn); */
};

} // Brush

#endif
