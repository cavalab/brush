#include "operator.h"

namespace Brush::Split{

State stitch(auto& child_outputs, const Data& d, const ArrayXb& mask)
{
    R result(mask.size());
    // ArrayXb mask = d[this->feature] < this->threshold;
    // ArrayXb mask = d.get(this->feature) < this->threshold;

    vector<size_t> L_idx, R_idx;
    tie (L_idx, R_idx) = Util::mask_to_indices(mask);
    result(L_idx) = get<R>(child_outputs.at(0));
    result(R_idx) = get<R>(child_outputs.at(1));
    return result;

}
auto get_best_variable_and_threshold(const Data& d)
{
    /* loops thru variables in d and picks the best threshold
     * and feature to split at.
     */
    float best_score = 0;
    int i = 0;
    vector<auto> feature_types{DataType::ArrayF,DataType::ArrayI,DataType::ArrayB};
    string feature;
    float threshold;

    for (auto& ft: feature_types)
    {
        for (const auto& key : d.features_of_type.at(ft)) 
        {
            float tmp_thresh, score;

            tie(tmp_thresh, score) = best_threshold(d[key], d.y, d.classification);
            if (score < best_score || i == 0)
            {
                best_score = score;
                this->feature = key;
                this->threshold = tmp_thresh;
            }
            ++i;
        }
    }
    return std::make_tuple(feature, threshold);
}

template<> auto get_thresholds<ArrayXb>(const ArrayXb& x){ vector<float>{0.0}; }
template<> auto get_thresholds<ArrayXi>(const ArrayXb& x){ 
    vector<float> thresholds;
    for (const auto& val : unique(x))
        thresholds.push_back(val);
    return thresholds;
}

template<> auto get_thresholds<ArrayXf>(const ArrayXb& x){ 
    vector<float> thresholds;
    auto s = unique(x);
    for (unsigned i =0; i<s.size()-1; ++i)
    {
        thresholds.push_back(s.at(i) + s.at(i+1));
    }
    return thresholds;
}

tuple<float,float> best_threshold(const auto& x, const ArrayXf& y, bool classification)
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
    float score, best_thresh, best_score;
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

        score = gain(lhs, rhs, classification, unique_classes);
        /* cout << "score: " << score << "\n"; */
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

/// Applies a learned threshold to a feature, returning a mask.
ArrayXb threshold_mask<ArrayXb>(const T& x, const float& threshold) { return x; }
ArrayXb threshold_mask<ArrayXf>(const T& x, const float& threshold) { return (x > threshold); }
ArrayXb threshold_mask<ArrayXi>(const T& x, const float& threshold) { return (x == threshold); }

float gain(const ArrayXf& lsplit, 
            const ArrayXf& rsplit, 
            bool classification, vector<float> unique_classes)
    {
        float lscore, rscore, score;
        if (classification)
        {
            lscore = gini_impurity_index(lsplit, unique_classes);
            rscore = gini_impurity_index(rsplit, unique_classes);
            cout << "lscore: " << lscore << "\n";
            cout << "rscore: " << rscore << "\n";
            score = (lscore*float(lsplit.size()) + 
                    rscore*float(rsplit.size()))
                        /(float(lsplit.size()) + float(rsplit.size()));
        }
        else
        {
            lscore = variance(lsplit)/float(lsplit.size());
            rscore = variance(rsplit)/float(rsplit.size());
            score = lscore + rscore; 
        }

        return score;
    }

float gini_impurity_index(const ArrayXf& classes, 
                          const vector<float>& uc)
{
    VectorXf class_weights = VectorXf::Zero(uc.size());
    for (auto c : uc){
        class_weights(int(c)) = float(
                (classes.cast<int>() == int(c)).count()
                )/classes.size();
        cout << "class_weights for " << c << ": " 
             << class_weights(int(c)) << "\n"; 
    }
    /* float total_weight=class_weights.sum(); */
    float gini = 1 - class_weights.dot(class_weights);

    return gini;
}

} //Brush::Split
