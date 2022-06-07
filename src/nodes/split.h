/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/

#ifndef SPLIT_H
#define SPLIT_H
#include "base.h"
using std::get;

namespace Brush {
namespace nodes {

/* Node for split functions 
 * 
 * */
template<typename F> class SplitNode; 

template<typename R, typename... Args>
class SplitNode<R(Args...)> : public TypedNodeBase<R, Args...>
{
    public:
        using base = TypedNodeBase<R, Args...>;
        using Function = std::function<std::array<Data,2>(Args...)>;
        using TupleArgs = typename base::TupleArgs;

        /// whether the variable choice is fixed
        bool fixed_variable;
        /// the learned feature choice
        // unsigned int loc;
        string feature;
        /// the learned threshold
        float threshold;
        /// sample feature space
        //TODO: implement!
        float feature_sample = 1.0;


        SplitNode(string name, string feature = "") : base(name) 
        {
            /* TODO: this constructor should determine whether we are searching
             * for the best split feature or taking a feature for splitting as
             * input. Basically, if there are three arguments to SplitNode,
             * the first should be used to pick a threshold, and the other two
             * should be returned conditionally.
            */
            // if (loc != -1)
            if (base::ArgCount == 3)
            {
                this->feature = feature;
                this->fixed_variable = true;
            }
        };

        State fit(const Data& d, TreeNode*& first_child, TreeNode*& last_child) override 
	    {

            /* 1) choose best feature
             * 2) choose best threshold of feature
             * 3) split data on feature at threshold
             * 4) evaluate child nodes on split data
             * 5) stitch child outputs together and return
             */

            cout << "fitting " << this->name << endl;

            // set feature and threshold
            if (this->fixed_variable)
            {
                tie(this->threshold, ignore) = best_threshold(
                                                             d[this->feature],
                                                             d.y, 
                                                             d.classification
                                                             );
            }
            else
                set_variable_and_threshold(d);

            // split the data
            // ArrayXb mask = std::get<R>(d[this->feature] < this->threshold;
            // ArrayXb mask = d[this->feature] < this->threshold;
            ArrayXb mask = this->threshold_mask(d);
            array<Data, 2> data_splits = d.split(mask);

            array<State, base::ArgCount> child_outputs;
            child_outputs = this->get_children_fit(data_splits, first_child, last_child);
            // cout << "gathering inputs..." << endl;
            // TreeNode* sib = first_child;
            // for (int i = 0; i < base::ArgCount; ++i)
            // {
            //     cout << i << endl;
            //     child_outputs.at(i) = sib->fit(data_splits.at(i));
            //     sib = sib->next_sibling;
            // }

            // stitch together outputs
            State out = this->stitch(child_outputs, d, mask);

            cout << "returning " << std::get<R>(out) << endl;

 			return out;
        };

        State predict(const Data& d, TreeNode*& first_child, 
                TreeNode*& last_child) override
	    {
            cout << "predicting " << this->name << endl;
            cout << "first_child: " << first_child << endl;
            cout << "last_child: " << last_child << endl;

            // split the data
            // ArrayXb mask = d.get(this->feature) < this->threshold;
            ArrayXb mask = this->threshold_mask(d);
            array<Data, 2> data_splits = d.split(mask);

            // array<State, base::ArgCount> child_outputs;
            auto child_outputs = this->get_children_predict(data_splits,    
                                                            first_child, 
                                                            last_child
                                                           );

            // stitch together outputs
            State out = this->stitch(child_outputs, d, mask);

            cout << "returning " << std::get<R>(out) << endl;
            return out;
        };

        /// split the gradient and send it to the children
        void grad_descent(const ArrayXf& gradient, const Data& d, 
                          TreeNode*& first_child, TreeNode*& last_child) override
        {
            // ArrayXb mask = d[this->feature] < this->threshold;
            // ArrayXb mask = d.get(this->feature) < this->threshold;
            ArrayXb mask = this->threshold_mask(d);
            array<Data, 2> data_splits = d.split(mask);

            array<ArrayXf, 2> grad_splits = Brush::Util::split(gradient, mask);

            first_child->grad_descent(grad_splits.at(0), data_splits.at(0)); 
            last_child->grad_descent(grad_splits.at(1), data_splits.at(1)); 

            base::set_prob_change(gradient.matrix().norm());
        };

    private:
        /// Utility to grab child outputs. 
        array<State, base::ArgCount> get_children(
                                const array<Data, 2>& data_splits,
                                TreeNode*& first_child, 
                                TreeNode*& last_child, 
                                State (TreeNode::*fn)(const Data&)
                                )
        {
            array<State, base::ArgCount> child_outputs;

            TreeNode* sib = first_child;
            for (int i = 0; i < base::ArgCount; ++i)
            {
                cout << i << endl;
                child_outputs.at(i) = (sib->*fn)(data_splits.at(i));
                sib = sib->next_sibling;
            }
            return child_outputs;
            
        };

        array<State, base::ArgCount> get_children_fit(
            const array<Data, 2>& data_splits, 
            TreeNode*& first_child, 
            TreeNode*& last_child)
        {
            return get_children(data_splits, first_child, last_child, &TreeNode::fit);
        }
        array<State, base::ArgCount> get_children_predict(
            const array<Data, 2>& data_splits,
            TreeNode*& first_child, 
            TreeNode*& last_child)
        {
            return get_children(data_splits, first_child, last_child, &TreeNode::predict);
        }
        /// Stitches together outputs from left or right child based on threshold
        State stitch(array<State, base::ArgCount>& child_outputs, const Data& d,
                     const ArrayXb& mask)
        {
            // TODO: this wont work as written; the index of the left and right
            // child need to be taken into account
            R result(mask.size());
            // ArrayXb mask = d[this->feature] < this->threshold;
            // ArrayXb mask = d.get(this->feature) < this->threshold;

            vector<size_t> L_idx, R_idx;
            tie (L_idx, R_idx) = Util::mask_to_indices(mask);
            result(L_idx) = get<R>(child_outputs.at(0));
            result(R_idx) = get<R>(child_outputs.at(1));
            // int lhs=0, rhs=0;
            // for (int i = 0; i < mask.size(); ++i)
            // {
            //     result(i) = mask(i) ? get<R>(child_outputs.at(0))(i) 
            //                         : get<R>(child_outputs.at(1))(i);
            // }
            return result;

        }
        /// Applies a learned threshold to a feature, returning a mask.
        ArrayXb threshold_mask(const State& x, const float& threshold)
        {
            return std::visit(overloaded {
                [&](const ArrayXb& arg) -> ArrayXb { return arg; },
                [&](const ArrayXi& arg) -> ArrayXb { return (arg == this->threshold); },
                [&](const ArrayXf& arg) -> ArrayXb { return (arg < this->threshold); },
                [&](const auto& arg)    { 
                    HANDLE_ERROR_THROW("Split threshold not defined for this State type!"); return ArrayXb(); },
                // [&](const ArrayXXb& arg) { return arg; },
                // [&](const ArrayXXi& arg) { return arg; },
                // [&](const ArrayXXf& arg) { return arg; },
                // [&](const TimeSeries& arg) { return arg; }
                }, x);

        }
        ArrayXb threshold_mask(const Data& d)
        {
            return this->threshold_mask(d[this->feature], this->threshold);

        }
        
        State get_unique(const State& value)
        {
            return std::visit([&](auto&& arg) 
            {
                using T = std::decay_t<decltype(arg)>;
                if constexpr (std::is_same_v<T, ArrayXb> 
                              || std::is_same_v<T, ArrayXi> 
                              || std::is_same_v<T, ArrayXf> 
                             )
                    return State(Util::unique(arg));
                else if constexpr (std::is_same_v<T, ArrayXXb> 
                                   || std::is_same_v<T, ArrayXXi> 
                                   || std::is_same_v<T, ArrayXXf> 
                                   || std::is_same_v<T, data::TimeSeries> 
                                  )
                {
                    HANDLE_ERROR_THROW("Can't get unique vals for type!");
                    return value;
                }
            },
            value
            );
        }
         
        void set_variable_and_threshold(const Data& d)
        {
            /* loops thru variables in d and picks the best threshold
             * and feature to split at.
             */
            float best_score = 0;
            int i = 0;
            vector<std::type_index> feature_types{typeid(ArrayXi),
                                                  typeid(ArrayXb),
                                                  typeid(ArrayXf)
                                                 };
            for (auto& ft: feature_types)
            {
                for (const auto& key : d.features_of_type.at(ft)) 
                {
                    float tmp_thresh, score;

                    tie(tmp_thresh, score) = best_threshold(d[key], 
                                                            d.y, 
                                                            d.classification
                                                           );
                    if (score < best_score || i == 0)
                    {
                        best_score = score;
                        this->feature = key;
                        this->threshold = tmp_thresh;
                    }
                    ++i;
                }
            }
        }

        tuple<float,float> best_threshold(const State& x, const ArrayXf& y, 
                                          bool classification)
        {
            /* for each unique value in x, calculate the reduction in the 
            * heuristic brought about by
            * splitting between that value and the next. 
            * set threshold according to the biggest reduction. 
            * 
            * returns: the threshold and the score.
            */
            // get all possible split masks based on variant type
            
            vector<float> all_thresholds = std::visit(overloaded 
            {
                [&](const ArrayXb& x) -> vector<float>
                {
                    return vector<float>{0.0};
                },
                [&](const ArrayXi& x) -> vector<float>
                {
                    vector<float> thresholds;
                    for (const auto& val : unique(x))
                        thresholds.push_back(val);
                    return thresholds;
                },
                [&](const ArrayXf& x) -> vector<float>
                {
                    vector<float> thresholds;
                    auto s = unique(x);
                    for (unsigned i =0; i<s.size()-1; ++i)
                    {
                        thresholds.push_back(s.at(i) + s.at(i+1));
                    }
                    return thresholds;
                },
                [&](const auto& x) -> vector<float>
                {
                    HANDLE_ERROR_THROW("Threshold not implemented for this type!");
                    return vector<float>();
                }
            }, x);
            // map<float, ArrayXb> all_masks = \
            // std::visit(overloaded {
            //     [&](const ArrayXb& x) -> map<float, ArrayXb>
            //     {
            //         return {0.0, x};
            //     },
            //     [&](const ArrayXi& x) -> map<float, ArrayXb>
            //     {
            //         map<float, ArrayXb> masks;
            //         for (const auto& val : unique(x))
            //             masks[val] = (x == val);
            //         return masks;
            //     },
            //     [&](const ArrayXb& x) -> map<float, ArrayXb>
            //     {
            //         map<float, ArrayXb> masks;
            //         auto s = unique(x);
            //         for (unsigned i =0; i<s.size()-1; ++i)
            //         {
            //             float val = (s.at(i) + s.at(i+1)) / 2;
            //             masks[val] = (x < val);
            //         }
            //         return masks;
            //     },
            //     [&](const auto& x) -> map<float, ArrayXb>
            //     {
            //         HANDLE_ERROR_THROW("Threshold not implemented for this type!");
            //         return {};
            //     }
            // }, x);

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

                score = gain(lhs, rhs, classification, 
                            unique_classes);
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

        // tuple<float,float> set_threshold(const Data& d, string var)
        // {
        //     /* for each unique value in x, calculate the reduction in the 
        //      * heuristic brought about by
        //      * splitting between that value and the next. 
        //      * set threshold according to the biggest reduction. 
        //      * 
        //      * returns: the threshold and the score.
        //      */
        //     //TODO: we need way to subset the data by features of a specific
        //     // type. In this case, array features.
        //     const auto& x = d[var]; 
        //     const ArrayXf& y = d.y;

        //     auto s = this->get_unique(x);

        //     // we'll treat x as a float if it has more than 10 unique values
        //     // using T = std::decay_t<decltype(x)>;
        //     // bool x_is_float = std::is_same_v<T, ArrayXf>;
        //     bool x_is_float = std::holds_alternative<ArrayXf>(x);

        //     vector<float> unique_classes = unique(y);
        //     float thresh, score, best_score;

        //     for (unsigned i =0; i<s.size()-1; ++i)
        //     {

        //         float val;
                
        //         if (x_is_float)
        //         {
        //             val = (s.at(i) + s.at(i+1)) / 2;
        //             mask = (x < val);
        //         }
        //         else
        //         {
        //             val = s.at(i);
        //             mask = (x == val);
        //         }
        //         vector<size_t> L_idx, R_idx;
        //         tie (L_idx, R_idx) = Util::mask_to_indices(this->threshold_mask(x, val));

        //         // split data
        //         const ArrayXf& lhs = y(L_idx); //target_splits[0];
        //         const ArrayXf& rhs = y(R_idx); //target_splits[1];

        //         if (lhs.size() == 0 || rhs.size() == 0)
        //             continue;

        //         score = gain(lhs, rhs, d.classification, 
        //                      unique_classes);
        //         /* cout << "score: " << score << "\n"; */
        //         if (score < best_score || i == 0)
        //         {
        //             best_score = score;
        //             thresh = val;
        //         }
        //     }

        //     thresh = std::isinf(thresh)? 
        //         0 : std::isnan(thresh)? 
        //         0 : thresh;

        //     return make_tuple(thresh, score);
        // }
       
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
};

} // nodes
} // Brush

#endif
