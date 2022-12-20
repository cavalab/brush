#include "operator.h"
#include <utility>

namespace Brush::Split{

tuple<string,float> get_best_variable_and_threshold(const Dataset& d, TreeNode& tn)
{
    /* loops thru variables in d and picks the best threshold
     * and feature to split at.
     */
    using FeatTypes = tuple<ArrayXf,ArrayXi,ArrayXb>;
    constexpr auto size = std::tuple_size<FeatTypes>::value;
    auto [feature, threshold, best_score] = get_best_thresholds<FeatTypes>(d, std::make_index_sequence<size>{});
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
        thresholds.push_back((s.at(i) + s.at(i+1))/2.0);
    }
    return thresholds;
}


/// Applies a learned threshold to a feature, returning a mask.
template<>
ArrayXb threshold_mask<ArrayXb>(const ArrayXb& x, const float& threshold) { 
    return x; 
}
template<>
ArrayXb threshold_mask<ArrayXf>(const ArrayXf& x, const float& threshold) { 
    return (x > threshold); 
}
template<>
ArrayXb threshold_mask<ArrayXi>(const ArrayXi& x, const float& threshold) { 
    return (x == threshold); 
}
template<>
ArrayXb threshold_mask<State>(const State& x, const float& threshold) { 
    // return std::visit(
    //     x
    // ); 
    return std::visit(
        [&](const auto& arg) -> ArrayXb { 
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, ArrayXb> 
                          || std::is_same_v<T, ArrayXi> 
                          || std::is_same_v<T, ArrayXf> 
                         )
                return threshold_mask(arg, threshold); 
            else
                return ArrayXb::Constant(arg.size(), true);
        },
        x
    );
}


float gain(const ArrayXf& lsplit, 
            const ArrayXf& rsplit, 
            bool classification, vector<float> unique_classes)
    {
        float lscore, rscore, score;
        if (classification)
        {
            lscore = gini_impurity_index(lsplit, unique_classes);
            rscore = gini_impurity_index(rsplit, unique_classes);
            /* cout << "lscore: " << lscore << "\n"; */
            /* cout << "rscore: " << rscore << "\n"; */
            score = (lscore*float(lsplit.size()) + 
                    rscore*float(rsplit.size()))
                        /(float(lsplit.size()) + float(rsplit.size()));
        }
        else
        {
            lscore = variance(lsplit)/float(lsplit.size());
            rscore = variance(rsplit)/float(rsplit.size());
            /* cout << "lscore: " << lscore << "\n"; */
            /* cout << "rscore: " << rscore << "\n"; */
            score = lscore + rscore; 
        }

        return score;
    }

float gini_impurity_index(const ArrayXf& classes, 
                          const vector<float>& unique_classes)
{
    vector<float> class_weights;
    for (auto c : unique_classes){
        class_weights.push_back(
            float( (classes.cast<int>() == int(c)).count())/classes.size()
        );
        cout << "class_weights for " << c << ": " 
             << class_weights.back() << "\n"; 
    }
    /* float total_weight=class_weights.sum(); */
    auto cw = VectorXf::Map(class_weights.data(), class_weights.size());
    float gini = 1 - cw.dot(cw);

    return gini;
}

} //Brush::Split