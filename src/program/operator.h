#ifndef OPERATOR_H
#define OPERATOR_H

#include "../init.h"
#include "tree_node.h"
#include "../util/utils.h"

namespace Brush{
///////////////////////////////////////////////////////////////////////////////////////
namespace util{
    ////////////////////////////////////////////////////////////////////////////////
    /// @brief get weight 
    /// @tparam T return type
    /// @tparam Scalar scalar type of return type
    /// @param tn tree node
    /// @param weights option pointer to a weight array, used in place of node weight
    /// @return 
    template<typename T, typename Scalar, typename W> 
        requires (!is_one_of_v<Scalar,bool,bJet>)
    Scalar get_weight(const TreeNode& tn, const W** weights=nullptr)
    { 
        Scalar w;
        // Prediction case: weight is stored in the node data.
        if (weights == nullptr)
        {
            w = Scalar(tn.data.W);
        }
        else
        {
            // NLS case 1: floating point weight is stored in weights
            if constexpr (is_same_v<Scalar, W>) 
                w = **weights;
            // NLS case 2: a Jet/Dual weight is stored in weights, but this constant is a 
            // integer type. We need to do some casting
            else if constexpr (is_same_v<Scalar, iJet> && is_same_v<W, fJet>) {
                using WScalar = typename Scalar::Scalar;
                WScalar tmp = WScalar((**weights).a);    
                w = Scalar(tmp);
            }
            // NLS case 3: a Jet/Dual weight is stored in weights, matching Scalar type
            else            
                w = Scalar(**weights);
            *weights = *weights+1;
            
        }
        return w;
    };
    template<typename T, typename Scalar, typename W> 
        requires (is_one_of_v<Scalar,bool,bJet>)
    Scalar get_weight(const TreeNode& tn, const W** weights=nullptr)
    {
        // we cannot weight a boolean feature. Nevertheless, we need to provide
        // an implementation for get_weight behavior, so the metaprogramming
        // doesn't fail to get a matching signature. 

        if (tn.data.get_is_weighted())
            // Node's init() function avoids the creation of weighted nodes, 
            // and the setter for `is_weighted` prevent enabling weight on 
            // boolean values. 
            HANDLE_ERROR_THROW(fmt::format("boolean terminal is weighted, but "
            "it should not\n"));

        return Scalar(true);
    };
}
////////////////////////////////////////////////////////////////////////////////
// Operator class

/// @brief Core computation of a node's function to data. 
/// @tparam S the signature of the node 
/// @tparam NT node type
/// @tparam Fit true: fit, false: predict
/// @tparam E used for node type specialization
template<NodeType NT, typename S, bool Fit, typename E=void> 
struct Operator 
{
    /**
    *   @brief set argument types to those of the signature unless:
    * 
    *   a) the operator is unary and there are more than one arguments
    *   b) the operator is binary and associative  
    * 
    *   In the case of a) or b), arguments to the operator are stacked into an 
    *   array and the operator is applied to that array
    */
    using ArgTypes = conditional_t<
        ((UnaryOp<NT> || NaryOp<NT>) && S::ArgCount > 1),
        Array<typename S::FirstArg::Scalar, -1, S::ArgCount>,
        typename S::ArgTypes>;

    /// @brief return type of the operator
    using RetType = typename S::RetType;

    /// @brief stores the argument count of the operator
    static constexpr size_t ArgCount = S::ArgCount;

    /// utility for returning the type of the Nth argument
    template <std::size_t N>
    using NthType = typename S::NthType<N>; 

    /// set weight type
    using W = typename S::WeightType; 
    
    /// @brief wrapper function for the node function
    static constexpr auto F = [](const auto& ...args) { 
        Function<NT> f; 
        return f(args...); 
    }; 

    Operator() = default;
    ////////////////////////////////////////////////////////////////////////////////
    // Utilities to grab child outputs.

    /// get a std::array or eigen array of kids
    template<typename T=ArgTypes> requires(is_std_array_v<T> || is_eigen_array_v<T>) 
    T get_kids(const Dataset& d, TreeNode& tn, const W** weights=nullptr) const
    {
        T child_outputs;
        using arg_type = std::conditional_t<is_std_array_v<T>,
            typename T::value_type, Array<typename S::FirstArg::Scalar, -1, 1>>;
        if constexpr (is_eigen_array_v<T>)
            child_outputs.resize(d.get_n_samples(), Eigen::NoChange);

        TreeNode* sib = tn.first_child;
        for (int i = 0; i < ArgCount; ++i)
        {
            if (sib == nullptr)
                HANDLE_ERROR_THROW("bad sibling ptr in get kids");
            if constexpr (Fit){
                if constexpr(is_std_array_v<T>)
                    child_outputs.at(i) = sib->fit<arg_type>(d);
                else
                    child_outputs.col(i) = sib->fit<arg_type>(d);
            }
            else{
                if constexpr(is_std_array_v<T>)
                    child_outputs.at(i) = sib->predict<arg_type>(d, weights);
                else
                    child_outputs.col(i) = sib->predict<arg_type>(d, weights);
            }
            sib = sib->next_sibling;
        }
        return child_outputs;
    };

    /// gets one kid for a tuple of kids
    template<int I>
    NthType<I> get_kid(const Dataset& d,TreeNode& tn, const W** weights ) const
    {
        auto sib = tree<TreeNode>::sibling_iterator(tn.first_child) ;
        sib += I;
        if constexpr(Fit)
            return sib->fit<NthType<I>>(d);
        else
            return sib->predict<NthType<I>>(d,weights);
    };

    /**
     * @brief Makes and returns a tuple of child outputs
     * 
     * @tparam T a tuple  
     * @tparam Is integer sequence 
     * @param d dataset
     * @param tn a tree node
     * @return a tuple with elements corresponding to each child node
     */
    template<typename T, size_t ...Is> requires(is_tuple_v<T>) 
    T get_kids_seq(const Dataset& d, TreeNode& tn, const W** weights, std::index_sequence<Is...>) const 
    { 
        return std::make_tuple(get_kid<Is>(d,tn,weights)...);
    };

    /// @brief get a std::tuple of kids. Used when child arguments are different types.
    /// @tparam T argument types 
    /// @param d the dataset
    /// @param tn the tree node
    /// @param weights option pointer to a weight array, used in place of node weight
    /// @return a tuple of the child arguments
    template<typename T=ArgTypes> requires(is_tuple_v<T>) 
    T get_kids(const Dataset& d, TreeNode& tn, const W** weights=nullptr) const
    {
        return get_kids_seq<T>(d, tn, weights, std::make_index_sequence<ArgCount>{});
    };

    ///////////////////////////////////////////////////////////////////////////

    /// @brief Apply node function in a functional style
    /// @tparam T argument types
    /// @param inputs the child node outputs
    /// @return return values applying F to the inputs
    template<typename T=ArgTypes> requires ( is_std_array_v<T> || is_tuple_v<T>)
    RetType apply(const T& inputs) const
    {
        return std::apply(F, inputs);
    }

    /// @brief Apply the node function like a function
    /// @tparam T argument types
    /// @param inputs the child node outputs
    /// @return return values applying F to the inputs
    template<typename T=ArgTypes> requires ( is_eigen_array_v<T> && !is_std_array_v<T>)
    RetType apply(const T& inputs) const
    {
        return F(inputs);
    }

    /// @brief evaluate the operator on the data. main entry point. 
    /// @tparam T argument types
    /// @tparam Scalar the underlying scalar type of the return type
    /// @param d dataset
    /// @param tn tree node
    /// @param weights option pointer to a weight array, used in place of node weight
    /// @return output values from applying operator function 
    template<typename T=ArgTypes, typename Scalar=RetType::Scalar>
    RetType eval(const Dataset& d, TreeNode& tn, const W** weights=nullptr) const
    {
        auto inputs = get_kids(d, tn, weights);
        if constexpr (is_one_of_v<Scalar,float,fJet>)
        {
            if (tn.data.get_is_weighted())
            {
                auto w = util::get_weight<RetType,Scalar,W>(tn, weights);
                return this->apply(inputs)*w;
            }
        }
        return this->apply(inputs);
    };

    // overloaded version for offset sum
    template<typename T=ArgTypes, typename Scalar=RetType::Scalar>
    requires is_in_v<NT, NodeType::OffsetSum>
    RetType eval(const Dataset& d, TreeNode& tn, const W** weights=nullptr) const
    {
        auto inputs = get_kids(d, tn, weights);
        if constexpr (is_one_of_v<Scalar,float,fJet>)
        {
            if (tn.data.get_is_weighted())
            {
                auto w = util::get_weight<RetType,Scalar,W>(tn, weights);
                return this->apply(inputs) + w;
            }
        }
        return this->apply(inputs);
    };
};

//////////////////////////////////////////////////////////////////////////////////
/// Terminal Overload
template<typename S, bool Fit>
struct Operator<NodeType::Terminal, S, Fit>
{
    using RetType = typename S::RetType;
    using W = typename S::WeightType; 

    // Standard C++ types
    template<typename T=RetType, typename Scalar=typename T::Scalar> 
        requires (is_one_of_v<Scalar,bool,int,float>)
    RetType eval(const Dataset& d, const TreeNode& tn, const W** weights=nullptr) const 
    { 
        if constexpr (is_one_of_v<Scalar,float,fJet>)
        {
            if (tn.data.get_is_weighted())
            {
                auto w = util::get_weight<RetType,Scalar,W>(tn, weights);
                return this->get<RetType>(d, tn.data.get_feature())*w;
            }
        }
        return this->get<RetType>(d,tn.data.get_feature());
    };

    // Jet types
    template <typename T = RetType, typename Scalar=typename T::Scalar>
        requires( is_one_of_v<Scalar, bJet, iJet, fJet>)
    RetType eval(const Dataset &d, const TreeNode &tn, const W **weights = nullptr) const
    {
        using nonJetType = UnJetify_t<RetType>; 
        if constexpr (is_one_of_v<Scalar,float,fJet>)
        {
            if (tn.data.get_is_weighted())
            {
                auto w = util::get_weight<RetType,Scalar,W>(tn, weights);
                return this->get<nonJetType>(d, tn.data.get_feature()).template cast<Scalar>()*w;
            }
        }
        return this->get<nonJetType>(d, tn.data.get_feature()).template cast<Scalar>();
    };

    // Accessing dataset directly
    template<typename T>
    auto get(const Dataset& d, const string& feature) const
    {
        if (std::holds_alternative<T>(d[feature]))
            return std::get<T>(d[feature]); 

        HANDLE_ERROR_THROW(fmt::format("Failed to return type {} for '{}'\n",
            DataTypeEnum<RetType>::value,
            feature
        ));

        return T(); 
    }
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
        // Scalar w = get_weight(tn, weights);
        Scalar w = util::get_weight<RetType,Scalar,W>(tn, weights);
        if constexpr (N == 1)
            return RetType::Constant(d.get_n_samples(), w); 
        else
            return RetType::Constant(d.get_n_samples(), d.get_n_features(), w); 
    };
    
};

////////////////////////////////////////////////////////////////////////////
// MeanLabel overload
template<typename S, bool Fit> 
struct Operator<NodeType::MeanLabel, S, Fit>
{
    using RetType = typename S::RetType;
    using W = typename S::WeightType; 

    RetType fit(const Dataset& d, TreeNode& tn) const {
        // we take the mode of the labels if it is a classification problem
        if (d.classification)
        {
            std::unordered_map<float, int> counters;
            for (float val : d.y) {
                if (counters.find(val) != counters.end()) {
                    counters[val] += 1;
                }
                else
                {
                    counters[val] = 1;
                }
            }

            auto mode = std::max_element(
                counters.begin(), counters.end(),
                [](const auto& a, const auto& b) { return a.second < b.second; }
            );

            tn.data.W = mode->first;
        }
        else
        {
            tn.data.W = d.y.mean();
        }
            
        return predict(d, tn);
    };

    template<typename T=RetType, typename Scalar=T::Scalar, int N=T::NumDimensions> 
    RetType predict(const Dataset& d, TreeNode& tn, const W** weights=nullptr) const 
    { 
        Scalar w = util::get_weight<RetType,Scalar,W>(tn, weights);
        if constexpr (N == 1)
            return RetType::Constant(d.get_n_samples(), w); 
        else
            return RetType::Constant(d.get_n_samples(), d.get_n_features(), w); 
    };

    RetType eval(const Dataset& d, TreeNode& tn, const W** weights=nullptr) const {
        if constexpr (Fit)
            return fit(d,tn); 
        else
            return predict(d,tn,weights);
    };
};

////////////////////////////////////////////////////////////////////////////
// Operator overloads 
//  Split
#include "split.h"
////////////////////////////////////////////////////////////////////////////
// Dispatch functions
template<typename R, NodeType NT, typename S, bool Fit, typename W> 
inline R DispatchOp(const Dataset& d, TreeNode& tn, const W** weights) 
{
    const auto op = Operator<NT,S,Fit>{};
    return op.eval(d, tn, weights);
};

template<typename R, NodeType NT, typename S, bool Fit> 
inline R DispatchOp(const Dataset& d, TreeNode& tn) 
{
    const auto op = Operator<NT,S,Fit>{};
    return op.eval(d, tn);
};

} // Brush

#endif
