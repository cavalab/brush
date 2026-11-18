/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/
#ifndef SPLIT_H
#define SPLIT_H

//////////////////////////////////////////////////////////////////////////////////
// Split Node Overloads
namespace Split{
    template<typename T>
    ArrayXb threshold_mask(const T& x, const float& threshold);
    /// Applies a learned threshold to a feature, returning a mask.
    template<typename T> requires same_as<typename T::Scalar, bool>
    ArrayXb threshold_mask(const T& x, const float& threshold) { 
        return x; 
    }
    template<typename T> requires same_as<typename T::Scalar, bJet>
    ArrayXb threshold_mask(const T& x, const float& threshold) { 
        ArrayXb ret(x.size()); 
        for (int i = 0; i< x.size(); ++i)
            ret(i) = x(i).a;
        return ret; 
    }
    template<typename T> requires same_as<typename T::Scalar, float>
    ArrayXb threshold_mask(const T& x, const float& threshold) { 
        return (x >= threshold); 
    }
    template<typename T> requires same_as<typename T::Scalar, fJet>
    ArrayXb threshold_mask(const T& x, const float& threshold) { 
        ArrayXb ret(x.size()); 
        std::transform(
            x.begin(), x.end(), ret.begin(), 
            [&](const auto& e){return e >= threshold;}
        );
        return ret; 
    }
    template<typename T> requires same_as<typename T::Scalar, int>
    ArrayXb threshold_mask(const T& x, const float& threshold) { 
        return (x == threshold); 
    }

    template<typename T> requires same_as<typename T::Scalar, iJet>
    ArrayXb threshold_mask(const T& x, const float& threshold) { 
        // return (x == threshold); 
        ArrayXb ret(x.size()); 
        std::transform(
            x.begin(), x.end(), ret.begin(), 
            [&](const auto& e){return e == threshold;}
        );
        return ret;
    }
    float gini_impurity_index(const ArrayXf& classes, const vector<float>& uc);
    float gain(const ArrayXf& lsplit, const ArrayXf& rsplit, bool classification,
            vector<float> unique_classes, const vector<float>& class_weights = {});

    template<typename T> vector<float> get_thresholds(const T& x); 
    struct Candidate {
        string feature;
        float threshold = 0.0f;
        float score = MAX_FLT;
        vector<int> categories;
    };
    Candidate best_category_split(const ArrayXi& x, const ArrayXf& y, bool classification,
                                  const vector<float>& class_weights = {});
    Candidate get_best_split(const Dataset& d, const vector<float>& class_weights = {});
    ArrayXb category_mask(const ArrayXi& x, const vector<int>& categories);

    template<typename T>
    tuple<float,float> best_threshold(const T& x, const ArrayXf& y, bool classification,
                                     const vector<float>& class_weights = {})
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

        //////////////////// shared //////////////////////
        float best_thresh = 0.0f, best_score = MAX_FLT;

        vector<float> unique_classes;
        if (classification)
            unique_classes = unique(y);

        // all_thresholds contains the unique values to be used as thresholds
        // with a >= operator  
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
            float score = gain(lhs, rhs, classification, unique_classes, class_weights);
            /* fmt::print("threshold={}; lhs={};rhs={}; score = {}\n",thresh,lhs,rhs,score); */
            if (std::isfinite(score) && score < best_score)
            {
                best_score = score;
                best_thresh = thresh;
            }
        }

        best_thresh = std::isinf(best_thresh)? 
            0 : std::isnan(best_thresh)? 
            0 : best_thresh;

        return make_tuple(best_thresh, best_score);

    }

    template<typename T>
    void get_best_threshold_by_type(const Dataset& d, auto& results,
                                    const vector<float>& class_weights)
    {
        DataType DT = DataTypeEnum<T>::value;
        /* fmt::print("get_best_threshold_by_type [T = {}]\n",DT); */

        if (d.features_of_type.find(DT) == d.features_of_type.end())
            return;

        for (const auto& key : d.features_of_type.at(DT))
        {
            Candidate candidate;
            if constexpr (std::is_same_v<T, ArrayXi>)
                candidate = best_category_split(std::get<T>(d[key]), d.y, d.classification, class_weights);
            else
                tie(candidate.threshold, candidate.score) =
                    best_threshold(std::get<T>(d[key]), d.y, d.classification, class_weights);
            candidate.feature = key;
            if (std::isfinite(candidate.score) && candidate.score < MAX_FLT)
                results.push_back(std::move(candidate));
        }
    }

    template<typename Ts,  std::size_t... Is> 
    auto get_best_thresholds(const Dataset&d, const vector<float>& class_weights,
                             std::index_sequence<Is...>)
    {
        /* fmt::print("get_best_thresholds\n"); */
        vector<Candidate> results;
        /* fmt::print("get_best_thresholds::results size:{}\n",results.size()); */
        (..., (get_best_threshold_by_type<std::tuple_element_t<Is,Ts>>(d, results, class_weights)));
        /* fmt::print("getting best\n"); */
        if (results.empty())
            return Candidate{};
        Candidate best = results.front();
        int ties = 1;
        for (size_t i = 1; i < results.size(); ++i) {
            const float tolerance = 1e-7f * std::max(1.0f, std::abs(best.score));
            if (results[i].score < best.score - tolerance) {
                best = results[i];
                ties = 1;
            } else if (std::abs(results[i].score - best.score) <= tolerance) {
                ++ties;
                if (Util::r.rnd_int(1, ties) == 1)
                    best = results[i];
            }
        }
        return best;
    }

    /// Stitches together outputs from left or right child based on threshold
    template<typename T>
    T stitch(std::array<T,2>& child_outputs, const ArrayXb& mask)
    {
        T result(mask.size());

        vector<size_t> L_idx, R_idx;
        tie (L_idx, R_idx) = Util::mask_to_indices(mask);
        result(L_idx) = child_outputs.at(0);
        result(R_idx) = child_outputs.at(1);
        return result;

    }
} // namespace Split

////////////////////////////////////////////////////////////////////////////////
// Split operator overload
template<NodeType NT, typename S, bool Fit> 
struct Operator<NT, S, Fit, enable_if_t<is_in_v<NT, NodeType::SplitOn, NodeType::SplitBest>>> 
{
    using ArgTypes = typename S::ArgTypes;
    using FirstArg = typename S::FirstArg;
    using RetType = typename S::RetType;
    using W = typename S::WeightType;
    static constexpr size_t ArgCount = S::ArgCount;
    // get arg types from tuple by index
    template <std::size_t N>
    using NthType = typename S::template NthType<N>;
    
    /* static constexpr auto F = [](const auto& ...args){ Function<NT> f{}; return f(args...); }; */ 
    static constexpr Function<NT> F{};

    // Optimizer parameters are laid out in postorder, including subtrees
    // receiving no samples. Advance past those parameters without evaluating
    // a branch on an empty dataset.
    static size_t count_weights(const TreeNode* node)
    {
        size_t count = 0;
        for (auto child = node->first_child; child != nullptr; child = child->next_sibling)
            count += count_weights(child);
        const auto& n = node->data;
        if (HasOptimizableWeight(n))
            ++count;
        return count;
    }

    std::array<RetType,2> get_kids(const std::array<Dataset, 2>& d, TreeNode& tn, const W** weights, const vector<float>& class_weights) const
    {
        using arg_type = NthType<1>;
        std::array<arg_type,2> child_outputs;

        TreeNode* sib = tn.first_child;
        if constexpr (NT==NodeType::SplitOn)
            sib = sib->next_sibling;

        for (int i = 0; i < 2; ++i)
        {
            if (d.at(i).get_n_samples() > 0)
            {
                if constexpr (Fit)
                    child_outputs.at(i) = sib->fit<arg_type>(d.at(i), class_weights);
                else
                    child_outputs.at(i) = sib->predict<arg_type>(d.at(i), weights);
            }
            else if constexpr (!Fit)
            {
                if (weights != nullptr)
                    *weights += count_weights(sib);
            }
            sib = sib->next_sibling;
        }
        return child_outputs;
    };

    RetType fit(const Dataset& d, TreeNode& tn, const vector<float>& class_weights) const {
        auto& threshold = tn.data.W;

        // set feature and threshold
        if constexpr (NT == NodeType::SplitOn)
        {
            // split on first child
            FirstArg split_feature = tn.first_child->fit<FirstArg>(d, class_weights);

            // get the best splitting threshold
            if (!tn.data.weight_is_fixed) // avoid changing fixed weights
                tie(threshold, ignore) = Split::best_threshold(split_feature, d.y, d.classification, class_weights);
        }
        else // splitbest
        {
            // avoid updating the split feature
            if (tn.data.get_keep_split_feature() && tn.data.get_feature()!="")
            {
                // TODO: I think the if-else clausules could be simplified

                auto values = d[tn.data.get_feature()];

                if (!tn.data.weight_is_fixed){
                    // Threshold will be optimized regardless.
                    if (std::holds_alternative<ArrayXf>(values))
                        tie(threshold, ignore) = Split::best_threshold(std::get<ArrayXf>(values), d.y, d.classification, class_weights);
                    else if (std::holds_alternative<ArrayXi>(values)) {
                        auto best = Split::best_category_split(std::get<ArrayXi>(values), d.y, d.classification, class_weights);
                        threshold = best.threshold;
                        tn.data.split_categories = std::move(best.categories);
                    }
                    else if (std::holds_alternative<ArrayXb>(values))
                        tie(threshold, ignore) = Split::best_threshold(std::get<ArrayXb>(values), d.y, d.classification, class_weights);
                }
                
            }
            else // keep_split_feature == false
            {
                if (!tn.data.weight_is_fixed){
                    auto best = Split::get_best_split(d, class_weights);
                    threshold = best.threshold;
                    tn.data.set_feature(best.feature);
                    tn.data.split_categories = std::move(best.categories);
                    if (!best.feature.empty())
                        tn.data.set_feature_type(d.get_feature_type(best.feature));
                }
            }
        }

        return predict(d, tn, nullptr, class_weights);
    }

    RetType predict(const Dataset& d, TreeNode& tn, const W** weights=nullptr, const vector<float>& class_weights = {}) const
    {
        const auto& threshold = tn.data.W;
        const auto& feature = tn.data.get_feature();

        // split the data
        ArrayXb mask;
        if constexpr (NT==NodeType::SplitBest) {
            if (feature.empty()) {
                mask.resize(d.get_n_samples());
                mask.fill(true);
            } else {
                const auto& value = d[feature];
                if (!tn.data.split_categories.empty() && std::holds_alternative<ArrayXi>(value))
                    mask = Split::category_mask(std::get<ArrayXi>(value), tn.data.split_categories);
                else
                    mask = Split::threshold_mask(value, threshold);
            }
        } else {
            auto split_feature = tn.first_child->predict<FirstArg>(d, weights);
            mask = Split::threshold_mask(split_feature, threshold);
        }

        std::array<Dataset, 2> data_splits = d.split(mask);
            
        auto child_outputs = get_kids(data_splits, tn, weights, class_weights);

        // stitch together outputs
        // fmt::print("stitching outputs\n");
        auto out = Split::stitch(child_outputs, mask);
        /* auto out = mask.select(child_outputs.at(0), child_outputs.at(1)); */
        /* cout << "returning " << std::get<RetType>(out) << endl; */

        return out;
    }
    RetType eval(const Dataset& d, TreeNode& tn, const W** weights=nullptr, const vector<float>& class_weights = {}) const {
        if constexpr (Fit)
            return fit(d,tn,class_weights);
        else
            return predict(d,tn,weights);
    }
};


#endif
