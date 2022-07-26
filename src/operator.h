#ifndef OPERATOR_H
#define OPERATOR_H

#include "init.h"
#include "tree_node.h"
/* #include "data/timeseries.h" */
/* using TreeNode = class tree_node_<Node>; */ 

namespace Brush{
///////////////////////////////////////////////////////////////////////////////////////
// Operator class
/* template<NodeType NT, typename S, bool Fit, typename T=void> Operator; */

template<NodeType NT, typename S, bool Fit, typename E=void> 
struct Operator 
{
    using Args = typename S::ArgTypes;
    using RetType = typename S::RetType;
    static constexpr size_t ArgCount = S::ArgCount;
    // get arg types from tuple by index
    template <std::size_t N>
    using NthType = typename S::NthType<N>; 
    
    static constexpr auto F = [](const auto& ...args){ Function<NT> f{}; return f(args...); }; 
    /* static constexpr auto F = [](const auto ...args){ Function<NT> f{}; return f(args...); }; */ 

    Operator() = default;
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
        if (tn.data.is_weighted)
            this->apply_weights(inputs, tn.data);
        return std::apply(F, inputs);
    };

    template<typename T=Args>
    requires (!is_array_v<T>)
    auto eval(const Data& d, TreeNode& tn) const
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
    auto eval(const Data& d, TreeNode& tn) const { 
        fmt::print("run std::get<{}>(d[{}])\n", DataTypeEnum<RetType>::value, tn.data.feature); 
        auto out = std::get<RetType>(d[tn.data.feature]);
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
    auto eval(const Data& d, TreeNode& tn) const { 
        return RetType::Constant(d.n_samples, tn.data.W.at(0)); 
    };

    template<typename T=RetType> requires same_as<T, ArrayXi>
    auto eval(const Data& d, TreeNode& tn) const { 
        return RetType::Constant(d.n_samples, int(tn.data.W.at(0))); 
    };

    template<typename T=RetType> requires same_as<T, ArrayXb>
    auto eval(const Data& d, TreeNode& tn) const { 
        return RetType(d.n_samples); 
    };

    template<typename T=RetType> requires same_as<T, ArrayXXf>
    auto eval(const Data& d, TreeNode& tn) const { 
        return RetType::Constant(d.n_samples, d.n_features, tn.data.W.at(0)); 
    };

    template<typename T=RetType> requires same_as<T, ArrayXXb>
    auto eval(const Data& d, TreeNode& tn) const { 
        return RetType(d.n_samples, d.n_features);
    };
};

//////////////////////////////////////////////////////////////////////////////////
// Split Node Overloads
namespace Split {
    template<typename T>
    ArrayXb threshold_mask(const T& x, const float& threshold);
    float gini_impurity_index(const ArrayXf& classes, const vector<float>& uc);
    float gain(const ArrayXf& lsplit, const ArrayXf& rsplit, bool classification, 
            vector<float> unique_classes);

    template<typename T> vector<float> get_thresholds(const T& x); 
    tuple<string,float> get_best_variable_and_threshold(const Data& d, TreeNode& tn);

    template<typename T>
    tuple<float,float> best_threshold(const T& x, const ArrayXf& y, bool classification)
    {
        /* for each unique value in x, calculate the reduction in the 
        * heuristic brought about by
        * splitting between that value and the next. 
        * set threshold according to the biggest reduction. 
        * 
        * returns: the threshold and the score.
        */
        // get all possible split masks based on variant type
        
        vector<float> all_thresholds = get_thresholds(x); 
        fmt::print("x: {}\n", x);
        fmt::print("y: {}\n", y);
        fmt::print("classification: {}\n", classification);
        fmt::print("all thresholds: {}\n", all_thresholds);

        //////////////////// shared //////////////////////
        float best_thresh, best_score = MAX_FLT;
        int i = 0 ;
        vector<float> unique_classes;
        if (classification)
            unique_classes = unique(y);

        for (const auto thresh: all_thresholds)
        {

            ArrayXb mask = threshold_mask(x, thresh);
            vector<size_t> L_idx, R_idx;
            tie (L_idx, R_idx) = Util::mask_to_indices(mask);

            // split data
            const ArrayXf& lhs = y(L_idx); 
            const ArrayXf& rhs = y(R_idx); 

            if (lhs.size() == 0 || rhs.size() == 0)
                continue;

            //TODO: templatize gain for classification/regression
            float score = gain(lhs, rhs, classification, unique_classes);
            fmt::print("threshold={}; lhs={};rhs={}; score = {}\n",thresh,lhs,rhs,score);
            if (score < best_score || i == 0)
            {
                best_score = score;
                best_thresh = thresh;
            }
            ++i;
        }

        best_thresh = std::isinf(best_thresh)? 
            0 : std::isnan(best_thresh)? 
            0 : best_thresh;

        return make_tuple(best_thresh, best_score);

    }

    template<typename T>
    void get_best_threshold_by_type(const Data& d, auto& results)
    {
        DataType DT = DataTypeEnum<T>::value;
        fmt::print("get_best_threshold_by_type [T = {}]\n",DT);

        vector<string> keys;
        float best_score = MAX_FLT;
        string feature="";
        float threshold=0.0;
        int i = 0;

        if (d.features_of_type.find(DT) != d.features_of_type.end())
            keys = d.features_of_type.at(DT);
        else
        {
            fmt::print("didn't find features of type {} in data\n",DT);
            return; // std::make_tuple(feature, threshold, best_score);
        }

        for (const auto& key : keys) 
        {
            float tmp_thresh, score;

            tie(tmp_thresh, score) = best_threshold(std::get<T>(d[key]), d.y, d.classification);
            fmt::print("best threshold for {} = {:.3f}, score = {:.3f}\n",key,tmp_thresh,score);
            if (score < best_score | i == 0)
            {
                best_score = score;
                feature = key;
                threshold = tmp_thresh;
            }
            ++i;
        }
        auto tmp = std::make_tuple(feature, threshold, best_score);
        fmt::print("returning {}\n",tmp);
        results.push_back(std::make_tuple(feature, threshold, best_score));
    }

    template<typename Ts,  std::size_t... Is> 
    auto get_best_thresholds(const Data&d, std::index_sequence<Is...>)
    {
        fmt::print("get_best_thresholds\n");
        using entry = tuple<string, float, float>;
        auto compare = [](const entry& a, const entry& b){ 
            return (std::get<2>(a) < std::get<2>(b)); 
        };

        vector<entry> results;
        fmt::print("get_best_thresholds::results size:{}\n",results.size());
        (..., (get_best_threshold_by_type<std::tuple_element_t<Is,Ts>>(d, results)));
        fmt::print("getting best\n");
        auto best = std::ranges::min_element(results, compare);
        fmt::print("best: {}\n",(*best));
        return (*best);
    }

    /// Stitches together outputs from left or right child based on threshold
    template<typename T>
    T stitch(array<T,2>& child_outputs, const ArrayXb& mask)
    {
        T result(mask.size());

        vector<size_t> L_idx, R_idx;
        tie (L_idx, R_idx) = Util::mask_to_indices(mask);
        result(L_idx) = child_outputs.at(0);
        result(R_idx) = child_outputs.at(1);
        return result;

    }
        
    
} // namespace Split


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
    using Args = typename S::ArgTypes;
    using FirstArg = typename S::base::FirstArg;
    using RetType = typename S::RetType;
    static constexpr size_t ArgCount = S::ArgCount;
    // get arg types from tuple by index
    template <std::size_t N>
    using NthType = typename S::NthType<N>; 
    
    /* static constexpr auto F = [](const auto& ...args){ Function<NT> f{}; return f(args...); }; */ 

    auto get_kids(const array<Data, 2>& d, TreeNode& tn) const
    {
        using arg_type = NthType<1>;
        array<arg_type,2> child_outputs;

        TreeNode* sib = tn.first_child;

        if constexpr (NT==NodeType::SplitOn)
            sib = sib->next_sibling;

        for (int i = 0; i < 2; ++i)
        {
            if constexpr (Fit)
                child_outputs.at(i) = sib->fit<arg_type>(d.at(i));
            else
                child_outputs.at(i) = sib->predict<arg_type>(d.at(i));
            sib = sib->next_sibling;
        }
        return child_outputs;
    };

    auto fit(const Data& d, TreeNode& tn) const {
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

    auto predict(const Data& d, TreeNode& tn) const 
    {
        const auto& threshold = tn.data.W.at(0);
        const auto& feature = tn.data.feature;

        // split the data
        ArrayXb mask;
        if constexpr (NT==NodeType::SplitBest)
            mask = Split::threshold_mask(std::get<FirstArg>(d[feature]), threshold);
        else {
            auto split_feature = tn.first_child->predict<FirstArg>(d);
            mask = Split::threshold_mask(split_feature, threshold);
        }

        array<Data, 2> data_splits = d.split(mask);

        auto child_outputs = get_kids(data_splits, tn);

        // stitch together outputs
        auto out = Split::stitch(child_outputs, mask);
        /* auto out = mask.select(child_outputs.at(0), child_outputs.at(1)); */
        /* cout << "returning " << std::get<RetType>(out) << endl; */

        return out;
    }
    auto eval(const Data& d, TreeNode& tn) const {
        if constexpr (Fit)
            return fit(d,tn); 
        else
            return predict(d,tn);
    }
};

////////////////////////////////////////////////////////////////////////////
// fit and predict Dispatch functions
template<typename R, NodeType NT, typename S, bool Fit> 
R DispatchOp(const Data& d, TreeNode& tn) 
{
    fmt::print("DispatchOp: Dispatching {}\n",NT);
    const auto op = Operator<NT,S,Fit>{};
    R out = op.eval(d, tn);
    // TODO: figure out why fmt::print isn't working with Eigen::Matrix
    /* fmt::print("{} returning {}\n",NT, out); */
    cout << NT << " output: " << out << endl;
    /* return out; */
    return op.eval(d,tn);
};

} // Brush

#endif
