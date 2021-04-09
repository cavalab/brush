/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/

#ifndef SPLIT_H
#define SPLIT_H
#include "base.h"

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
        unsigned int loc;
        /// the learned threshold
        float threshold;
        /// sample feature space
        //TODO: implement!
        float feature_sample = 1.0;


        SplitNode(string name, int loc = -1) : base(name) 
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
                this->loc = loc;
                this->fixed_variable = true;
            }
        };

        State fit(const Data& d, TreeNode*& child1, TreeNode*& child2) override 
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
                // TODO: replace loc with child arg 1 output
                tie(this->threshold, ignore) = set_threshold(d, this->loc);
            }
            else
                set_variable_and_threshold(d);

            // split the data
            ArrayXb mask = d.X.row(this->loc).array() < this->threshold;
            array<Data, 2> data_splits = d.split(mask);

            array<State, base::ArgCount> child_outputs;
            child_outputs = this->get_children_fit(data_splits, child1, child2);
            // cout << "gathering inputs..." << endl;
            // TreeNode* sib = child1;
            // for (int i = 0; i < base::ArgCount; ++i)
            // {
            //     cout << i << endl;
            //     child_outputs.at(i) = sib->fit(data_splits.at(i));
            //     sib = sib->next_sibling;
            // }

            // stitch together outputs
            State out = this->stitch(child_outputs, d);

            cout << "returning " << std::get<R>(out) << endl;

 			return out;
        };

        State predict(const Data& d, TreeNode*& child1, 
                TreeNode*& child2) override
	    {
            cout << "predicting " << this->name << endl;
            cout << "child1: " << child1 << endl;
            cout << "child2: " << child2 << endl;

            // split the data
            ArrayXb mask = d.X.row(this->loc).array() < this->threshold;
            array<Data, 2> data_splits = d.split(mask);

            // array<State, base::ArgCount> child_outputs;
            auto child_outputs = this->get_children_predict(data_splits, child1, 
                                                       child2);
            // cout << "gathering inputs..." << endl;
            // TreeNode* sib = child1;
            // for (int i = 0; i < base::ArgCount; ++i)
            // {
            //     cout << i << endl;
            //     child_outputs.at(i) = sib->predict(data_splits.at(i));
            //     sib = sib->next_sibling;
            // }

            // stitch together outputs
            State out = this->stitch(child_outputs, d);

            cout << "returning " << std::get<R>(out) << endl;
            return out;
        };

        /// split the gradient and send it to the children
        void grad_descent(const ArrayXf& gradient, const Data& d, 
                          TreeNode*& child1, TreeNode*& child2) override
        {
            ArrayXb mask = d.X.row(this->loc).array() < this->threshold;
            array<Data, 2> data_splits = d.split(mask);

            array<ArrayXf, 2> grad_splits = Brush::Util::split(gradient, mask);

            child1->grad_descent(grad_splits.at(0), data_splits.at(0)); 
            child2->grad_descent(grad_splits.at(1), data_splits.at(1)); 

            base::set_weight(gradient.matrix().norm());
        };

    private:
        /// Utility to grab child outputs. 
        array<State, base::ArgCount> get_children(
                                const array<Data, 2>& data_splits,
                                TreeNode*& child1, 
                                TreeNode*& child2, 
                                State (TreeNode::*fn)(const Data&)
                                )
        {
            array<State, base::ArgCount> child_outputs;

            TreeNode* sib = child1;
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
            TreeNode*& child1, 
            TreeNode*& child2)
        {
            return get_children(data_splits, child1, child2, &TreeNode::fit);
        }
        array<State, base::ArgCount> get_children_predict(
            const array<Data, 2>& data_splits,
            TreeNode*& child1, 
            TreeNode*& child2)
        {
            return get_children(data_splits, child1, child2, &TreeNode::predict);
        }
        /// Stitches together outputs from left or right child based on threshold
        State stitch(array<State, base::ArgCount>& child_outputs, const Data& d)
        {
            R result;
            ArrayXb mask = d.X.row(this->loc).array() < this->threshold;
            for (int i = 0; i < mask.size(); ++i)
            {
                result(i) = mask(i) ? get<R>(child_outputs.at(0))(i) 
                                    : get<R>(child_outputs.at(1))(i);
            }
            return result;

        }

        void set_variable_and_threshold(const Data& d)
        {
            /* loops thru variables in d.X and picks the best threshold
             * and feature to split at.
             */
            float best_score = 0;
            for (int i = 0; i < d.X.rows(); ++i)
            {
                float tmp_thresh, score;
                tie(tmp_thresh, score) = set_threshold(d, i);

                if (score < best_score || i == 0)
                {
                    best_score = score;
                    this->loc = i;
                    this->threshold = tmp_thresh;
                }

            }
        }

        tuple<float,float> set_threshold(const Data& d, int var_idx)
        {
            /* for each unique value in x, calculate the reduction in the 
             * heuristic brought about by
             * splitting between that value and the next. 
             * set threshold according to the biggest reduction. 
             */
            const ArrayXf& x = d.X.row(var_idx); 
            const VectorXf& y = d.y;

            vector<float> s = unique(x);

            // we'll treat x as a float if it has more than 10 unique values
            bool x_is_float = d.X_dtypes.at(var_idx) == "float";

            vector<float> unique_classes = unique(y);
            vector<int> idx(x.size());
            std::iota(idx.begin(),idx.end(), 0);
            Map<ArrayXi> midx(idx.data(),idx.size());
            float thresh, score, best_score;

            for (unsigned i =0; i<s.size()-1; ++i)
            {

                float val;
                ArrayXi split_idx;
                
                if (x_is_float)
                {
                    val = (s.at(i) + s.at(i+1)) / 2;
                    split_idx = (x < val).select(midx,-midx-1);
                }
                else
                {
                    val = s.at(i);
                    split_idx = (x == val).select(midx,-midx-1);
                }

                /* cout << "split val: " << val << "\n"; */

                // split data
                vector<float> d1, d2; 
                for (unsigned j=0; j< split_idx.size(); ++j)
                {
                    if (split_idx(j) <0)
                        d2.push_back(y(-1-split_idx(j)));
                    else
                        d1.push_back(y(split_idx(j)));
                }
                if (d1.empty() || d2.empty())
                    continue;

                Map<VectorXf> map_d1(d1.data(), d1.size());  
                Map<VectorXf> map_d2(d2.data(), d2.size());  
                /* cout << "d1: " << map_d1.transpose() << "\n"; */
                /* cout << "d2: " << map_d2.transpose() << "\n"; */
                score = gain(map_d1, map_d2, d.classification, 
                        unique_classes);
                /* cout << "score: " << score << "\n"; */
                if (score < best_score || i == 0)
                {
                    best_score = score;
                    thresh = val;
                }
                /* cout << val << "," << score << "\n"; */
            }

            thresh = std::isinf(thresh)? 
                0 : std::isnan(thresh)? 
                0 : thresh;

            return make_tuple(thresh, score);
        }
       
        float gain(const VectorXf& lsplit, 
                const VectorXf& rsplit, 
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
                lscore = variance(lsplit.array())/float(lsplit.size());
                rscore = variance(rsplit.array())/float(rsplit.size());
                score = lscore + rscore; 
            }

            return score;
        }

        float gini_impurity_index(const VectorXf& classes, 
                vector<float> uc)
        {
            VectorXf class_weights(uc.size());
            for (auto c : uc){
                class_weights(c) = 0;
                class_weights(c) = float(
                        (classes.cast<int>().array() == int(c)).count()
                        )/classes.size();
                cout << "class_weights for " << c << ": " 
                     << class_weights(c) << "\n"; 
            }
            /* float total_weight=class_weights.sum(); */
            float gini = 1 - class_weights.dot(class_weights);

            return gini;
        }
};

} // nodes
} // Brush

#endif