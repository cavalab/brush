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
        /* fmt::print("x: {}\n", x); */
        /* fmt::print("y: {}\n", y); */
        /* fmt::print("classification: {}\n", classification); */
        /* fmt::print("all thresholds: {}\n", all_thresholds); */

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

#endif