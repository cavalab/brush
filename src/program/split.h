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
        return (x > threshold); 
    }
    template<typename T> requires same_as<typename T::Scalar, fJet>
    ArrayXb threshold_mask(const T& x, const float& threshold) { 
        ArrayXb ret(x.size()); 
        std::transform(
            x.begin(), x.end(), ret.begin(), 
            [&](const auto& e){return e > threshold;}
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
            vector<float> unique_classes);

    template<typename T> vector<float> get_thresholds(const T& x); 
    tuple<string,float> get_best_variable_and_threshold(const Dataset& d, TreeNode& tn);

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
            /* fmt::print("threshold={}; lhs={};rhs={}; score = {}\n",thresh,lhs,rhs,score); */
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
    void get_best_threshold_by_type(const Dataset& d, auto& results)
    {
        DataType DT = DataTypeEnum<T>::value;
        /* fmt::print("get_best_threshold_by_type [T = {}]\n",DT); */

        vector<string> keys;
        float best_score = MAX_FLT;
        string feature="";
        float threshold=0.0;
        int i = 0;

        if (d.features_of_type.find(DT) != d.features_of_type.end())
            keys = d.features_of_type.at(DT);
        else
        {
            /* fmt::print("didn't find features of type {} in data\n",DT); */
            return; // std::make_tuple(feature, threshold, best_score);
        }

        for (const auto& key : keys) 
        {
            float tmp_thresh, score;

            tie(tmp_thresh, score) = best_threshold(std::get<T>(d[key]), d.y, d.classification);
            // fmt::print("best threshold for {} = {:.3f}, score = {:.3f}\n",key,tmp_thresh,score);
            if (score < best_score | i == 0)
            {
                best_score = score;
                feature = key;
                threshold = tmp_thresh;
            }
            ++i;
        }
        auto tmp = std::make_tuple(feature, threshold, best_score);
        results.push_back(std::make_tuple(feature, threshold, best_score));
    }

    template<typename Ts,  std::size_t... Is> 
    auto get_best_thresholds(const Dataset&d, std::index_sequence<Is...>)
    {
        /* fmt::print("get_best_thresholds\n"); */
        using entry = tuple<string, float, float>;
        auto compare = [](const entry& a, const entry& b){ 
            return (std::get<2>(a) < std::get<2>(b)); 
        };

        vector<entry> results;
        /* fmt::print("get_best_thresholds::results size:{}\n",results.size()); */
        (..., (get_best_threshold_by_type<std::tuple_element_t<Is,Ts>>(d, results)));
        /* fmt::print("getting best\n"); */
        auto best = std::ranges::min_element(results, compare);
        /* fmt::print("best: {}\n",(*best)); */
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
    using NthType = typename S::NthType<N>; 
    
    /* static constexpr auto F = [](const auto& ...args){ Function<NT> f{}; return f(args...); }; */ 
    static constexpr Function<NT> F{};

    array<RetType,2> get_kids(const array<Dataset, 2>& d, TreeNode& tn, const W** weights=nullptr) const
    {
        using arg_type = NthType<1>;
        array<arg_type,2> child_outputs;

        TreeNode* sib = tn.first_child;
        if constexpr (NT==NodeType::SplitOn)
            sib = sib->next_sibling;

        for (int i = 0; i < 2; ++i)
        {
            if (d.at(i).get_n_samples() > 0)
            {
                if constexpr (Fit)
                    child_outputs.at(i) = sib->fit<arg_type>(d.at(i));
                else
                    child_outputs.at(i) = sib->predict<arg_type>(d.at(i), weights);
            }
            sib = sib->next_sibling;
        }
        return child_outputs;
    };

    RetType fit(const Dataset& d, TreeNode& tn) const {
        auto& threshold = tn.data.W;

        // set feature and threshold
        if constexpr (NT == NodeType::SplitOn)
        {
            // split on first child
            FirstArg split_feature = tn.first_child->fit<FirstArg>(d);
            // get the best splitting threshold
            tie(threshold, ignore) = Split::best_threshold(split_feature, d.y, d.classification);
        }
        else
        {
            string feature = "";
            tie(feature, threshold) = Split::get_best_variable_and_threshold(d, tn);
            tn.data.set_feature(feature);
        }

        return predict(d, tn);
    }

    RetType predict(const Dataset& d, TreeNode& tn, const W** weights=nullptr) const 
    {
        const auto& threshold = tn.data.W;
        const auto& feature = tn.data.get_feature();

        // split the data
        ArrayXb mask;
        if (feature == "")
        {
            mask.resize(d.get_n_samples());
            mask.fill(true);
        }
        else if constexpr (NT==NodeType::SplitBest)
            mask = Split::threshold_mask(d[feature], threshold);
        else {
            auto split_feature = tn.first_child->predict<FirstArg>(d, weights);
            mask = Split::threshold_mask(split_feature, threshold);
        }

        array<Dataset, 2> data_splits = d.split(mask);
            
        auto child_outputs = get_kids(data_splits, tn, weights);

        // stitch together outputs
        // fmt::print("stitching outputs\n");
        auto out = Split::stitch(child_outputs, mask);
        /* auto out = mask.select(child_outputs.at(0), child_outputs.at(1)); */
        /* cout << "returning " << std::get<RetType>(out) << endl; */

        return out;
    }
    RetType eval(const Dataset& d, TreeNode& tn, const W** weights=nullptr) const {
        if constexpr (Fit)
            return fit(d,tn); 
        else
            return predict(d,tn,weights);
    }
};


#endif