#include "operator.h"

namespace Brush::Split{

tuple<string,float> get_best_variable_and_threshold(const Data& d, TreeNode& tn)
{
    /* loops thru variables in d and picks the best threshold
     * and feature to split at.
     */
    float best_score = 0;
    int i = 0;
    vector<DataType> feature_types{DataType::ArrayF,DataType::ArrayI,DataType::ArrayB};
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
                tn.n.feature = key;
                tn.n.W.at(0) = tmp_thresh;
            }
            ++i;
        }
    }
    return std::make_tuple(feature, threshold);
}

template<> vector<float> get_thresholds<ArrayXb>(const ArrayXb& x){ return vector<float>{0.0}; }
template<> vector<float> get_thresholds<ArrayXi>(const ArrayXi& x){ 
    vector<float> thresholds;
    for (const auto& val : unique(x))
        thresholds.push_back(val);
    return thresholds;
}

template<> vector<float> get_thresholds<ArrayXf>(const ArrayXf& x){ 
    vector<float> thresholds;
    auto s = unique(x);
    for (unsigned i =0; i<s.size()-1; ++i)
    {
        thresholds.push_back(s.at(i) + s.at(i+1));
    }
    return thresholds;
}


/// Applies a learned threshold to a feature, returning a mask.
template<>
ArrayXb threshold_mask<ArrayXb>(const ArrayXb& x, const float& threshold) { return x; }
template<>
ArrayXb threshold_mask<ArrayXf>(const ArrayXf& x, const float& threshold) { return (x > threshold); }
template<>
ArrayXb threshold_mask<ArrayXi>(const ArrayXi& x, const float& threshold) { return (x == threshold); }


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
