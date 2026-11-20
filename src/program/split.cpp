#include "operator.h"
#include <unordered_set>
#include <utility>
using Brush::Data::State;

namespace Brush::Split{

Candidate get_best_split(const Dataset& d, const vector<float>& class_weights)
{
    // Compare candidate splits from float, integer, and boolean features.
    // get_best_thresholds (in split.h) selects among the lowest scores,
    // breaking near-ties randomly.
    using FeatTypes = tuple<ArrayXf,ArrayXi,ArrayXb>;
    constexpr auto size = std::tuple_size<FeatTypes>::value;
    return get_best_thresholds<FeatTypes>(d, class_weights, std::make_index_sequence<size>{});
}

ArrayXb category_mask(const ArrayXi& x, const vector<int>& categories)
{
    // Membership selects the first branch; other values select the second.
    ArrayXb mask(x.size());
    const std::unordered_set<int> selected(categories.begin(), categories.end());
    for (int i = 0; i < x.size(); ++i)
        mask(i) = selected.find(x(i)) != selected.end();
    return mask;
}

Candidate best_category_split(const ArrayXi& x, const ArrayXf& y, bool classification,
                              const vector<float>& class_weights)
{
    // Integer features represent categories: search for a group of values
    // (for example, {0, 2}) versus all remaining values.
    Candidate best;
    const auto categories = unique(x);
    if (categories.size() < 2)
        return best;
    const auto classes = classification ? unique(y) : vector<float>{};

    // Aggregate each category once, so candidate groups can be scored without
    // rescanning every row. Mass is the sum of sample weights (or row count);
    // class masses are used for Gini, while sums are used for regression.
    struct Stats {
        double mass = 0.0, sum = 0.0, sum_squared = 0.0;
        vector<double> classes;
        explicit Stats(size_t n_classes) : classes(n_classes, 0.0) {}
        void add(const Stats& other) {
            mass += other.mass;
            sum += other.sum;
            sum_squared += other.sum_squared;
            for (size_t i = 0; i < classes.size(); ++i)
                classes[i] += other.classes[i];
        }
    };

    vector<Stats> stats(categories.size(), Stats(classes.size()));
    std::unordered_map<int, size_t> category_index;

    for (size_t i = 0; i < categories.size(); ++i)
        category_index[categories[i]] = i;

    std::map<float, size_t> class_index;
    for (size_t i = 0; i < classes.size(); ++i)
        class_index[classes[i]] = i;

    Stats total(classes.size());
    for (int i = 0; i < x.size(); ++i) {
        // Classification labels are encoded as indices into class_weights.
        // Regression and unweighted classification give each row weight 1.
        const double weight = classification && !class_weights.empty()
            ? class_weights.at(static_cast<size_t>(y(i))) : 1.0;
        auto& category = stats[category_index.at(x(i))];
        category.mass += weight;
        category.sum += weight * y(i);
        category.sum_squared += weight * double(y(i)) * y(i);
        if (classification)
            category.classes[class_index.at(y(i))] += weight;
    }
    for (const auto& category : stats)
        total.add(category);

    // Consider a group and its complement; a lower post-split score is better.
    double best_score = std::numeric_limits<double>::infinity();
    auto consider = [&](const Stats& left, vector<int> subset) {
        const double right_mass = total.mass - left.mass;
        // A branch with no effective weight cannot define an impurity.
        if (left.mass <= 0.0 || right_mass <= 0.0)
            return;
        double score;
        if (classification) {
            // Weighted average of the two branches' Gini impurities:
            // 1 - sum(left_class^2 / left_mass + right_class^2 / right_mass) / total_mass.
            double explained = 0.0;
            for (size_t i = 0; i < classes.size(); ++i) {
                const double left_class = left.classes[i];
                const double right_class = total.classes[i] - left_class;
                explained += left_class * left_class / left.mass
                           + right_class * right_class / right_mass;
            }
            score = 1.0 - explained / total.mass;
        } else {
            // Within each branch, sum(y^2) - sum(y)^2 / count is the
            // squared error around its mean. Normalize by the total count.
            const double right_sum = total.sum - left.sum;
            const double left_error = left.sum_squared - left.sum * left.sum / left.mass;
            const double right_error = total.sum_squared - left.sum_squared
                                     - right_sum * right_sum / right_mass;
            score = std::max(0.0, (left_error + right_error) / total.mass);
        }
        if (!std::isfinite(score))
            return;
        constexpr double tolerance = 1e-12;
        if (score > best_score + tolerance)
            return;

        // A partition and its complement have the same score. Always store the
        // side containing the smallest category; break equal-score ties by the
        // sorted category list, independently of class-label ordering.
        std::sort(subset.begin(), subset.end());
        if (!std::binary_search(subset.begin(), subset.end(), categories.front())) {
            vector<int> complement;
            std::set_difference(categories.begin(), categories.end(), subset.begin(), subset.end(),
                                std::back_inserter(complement));
            subset = std::move(complement);
        }
        if (score < best_score - tolerance
            || (std::abs(score - best_score) <= tolerance && subset < best.categories)) {
            best_score = score;
            best.score = static_cast<float>(score);
            // Prediction uses the full category group, not this single value.
            best.threshold = static_cast<float>(subset.front());
            best.categories = std::move(subset);
        }
    };

    // Search all unordered partitions for small categorical features. The first
    // category is always included to avoid testing the same split twice; the
    // all-categories group is skipped because its other branch would be empty.
    if (categories.size() <= 10) {
        const size_t combinations = size_t{1} << (categories.size() - 1);
        for (size_t bits = 0; bits + 1 < combinations; ++bits) {
            vector<int> subset{categories.front()};
            Stats left = stats.front();
            for (size_t i = 1; i < categories.size(); ++i)
                if (bits & (size_t{1} << (i - 1))) {
                    subset.push_back(categories[i]);
                    left.add(stats[i]);
                }
            consider(left, std::move(subset));
        }
    } else {
        // Exhaustive search grows exponentially. For larger features, test
        // single categories and prefixes ordered by target mean (regression)
        // or by each class's weighted proportion (classification). These are
        // candidate splits, so the best possible grouping is not guaranteed.
        // Separate class orderings also avoid treating class codes as numbers.
        for (size_t i = 0; i < categories.size(); ++i)
            consider(stats[i], {categories[i]});
        const size_t orderings = classification ? classes.size() : 1;
        for (size_t c = 0; c < orderings; ++c) {
            vector<std::pair<double, size_t>> ordered;
            for (size_t i = 0; i < categories.size(); ++i) {
                const double value = stats[i].mass > 0.0
                    ? (classification ? stats[i].classes[c] : stats[i].sum) / stats[i].mass
                    : 0.0;
                ordered.emplace_back(value, i);
            }
            std::sort(ordered.begin(), ordered.end());
            vector<int> subset;
            Stats left(classes.size());
            for (size_t i = 0; i + 1 < ordered.size(); ++i) {
                const auto category = ordered[i].second;
                subset.push_back(categories[category]);
                left.add(stats[category]);
                consider(left, subset);
            }
        }
    }
    return best;
}

template<> vector<float> get_thresholds<ArrayXb>(const ArrayXb& x){ return vector<float>{1.0}; }
template<> vector<float> get_thresholds<ArrayXbJet>(const ArrayXbJet& x){ return vector<float>{1.0}; }
template<> vector<float> get_thresholds<ArrayXi>(const ArrayXi& x){ 
    vector<float> thresholds;
    for (const auto& val : unique(x))
        thresholds.push_back(val);
    return thresholds;
}
template<> vector<float> get_thresholds<ArrayXiJet>(const ArrayXiJet& x){ 
    vector<float> thresholds;
    for (const auto& val : unique(x))
        thresholds.push_back(val.a);
    return thresholds;
}

template<> vector<float> get_thresholds<ArrayXf>(const ArrayXf& x){ 
    vector<float> thresholds;
    auto s = unique(x);
    // With x >= threshold, the smallest value puts every row on one side.
    // Starting at 1 also handles an empty feature without unsigned underflow.
    for (size_t i = 1; i < s.size(); ++i)
    {
        thresholds.push_back(s.at(i));
    }
    return thresholds;
}
template<> vector<float> get_thresholds<ArrayXfJet>(const ArrayXfJet& x){ 
    vector<float> thresholds;
    auto s = unique(x);
    for (size_t i = 1; i < s.size(); ++i)
    {
        thresholds.push_back(s.at(i).a);
    }
    return thresholds;
}


template<>
ArrayXb threshold_mask<State>(const State& x, const float& threshold) { 
    return std::visit(
        [&](const auto& arg) -> ArrayXb { 
            using T = std::decay_t<decltype(arg)>;
            if constexpr (T::NumDimensions == 1)
                return threshold_mask(arg, threshold); 
            else
                return ArrayXb::Constant(arg.size(), true);
        },
        x
    );
}
float gain(const ArrayXf& lsplit, const ArrayXf& rsplit, bool classification,
           vector<float> unique_classes, const vector<float>& class_weights)
{
    if (classification) {
        // Accumulate weighted counts for every class in each branch. With no
        // class weights, each row counts once, as in the original Gini score.
        auto weighted_counts = [&](const ArrayXf& labels) {
            vector<double> counts(unique_classes.size(), 0.0);
            double mass = 0.0;
            for (float label : labels) {
                const double weight = class_weights.empty()
                    ? 1.0 : class_weights.at(static_cast<size_t>(label));
                const auto it = std::find(unique_classes.begin(), unique_classes.end(), label);
                counts.at(std::distance(unique_classes.begin(), it)) += weight;
                mass += weight;
            }
            return std::make_pair(counts, mass);
        };

        const auto [left, left_mass] = weighted_counts(lsplit);
        const auto [right, right_mass] = weighted_counts(rsplit);

        // Reject a branch containing no effective sample weight.
        if (left_mass <= 0.0 || right_mass <= 0.0)
            return MAX_FLT;

        // Weighted mean Gini of the two branches; lower scores are better.
        double explained = 0.0;
        for (size_t i = 0; i < unique_classes.size(); ++i)
            explained += left[i] * left[i] / left_mass + right[i] * right[i] / right_mass;

        return static_cast<float>(1.0 - explained / (left_mass + right_mass));
    }

    return (variance(lsplit) * float(lsplit.size()) + variance(rsplit) * float(rsplit.size()))
         / float(lsplit.size() + rsplit.size());
}

float gini_impurity_index(const ArrayXf& classes, 
                          const vector<float>& unique_classes)
{
    vector<float> class_weights;
    for (auto c : unique_classes){
        class_weights.push_back(
            float( (classes.cast<int>() == int(c)).count())/classes.size()
        );
    }
    /* float total_weight=class_weights.sum(); */
    auto cw = VectorXf::Map(class_weights.data(), class_weights.size());
    float gini = 1 - cw.dot(cw);

    return gini;
}

} //Brush::Split
