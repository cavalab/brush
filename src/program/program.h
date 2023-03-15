/* Brush

copyright 2020 William La Cava
license: GNU/GPL v3
*/
#ifndef PROGRAM_H
#define PROGRAM_H
//external includes

//

#include <string>
#include "assert.h"

// internal includes

// #include "data/data.h"
#include "../init.h"
#include "tree_node.h"
#include "node.h"
#include "../search_space.h"
#include "../params.h"
#include "../util/utils.h"
#include "functions.h"
// #include "weight_optimizer.h"


using std::cout;
using std::string;
using Brush::Data::State;
using Brush::Data::Dataset;
using Brush::SearchSpace;

namespace Brush {


typedef tree<Node>::pre_order_iterator Iter; 
typedef tree<Node>::post_order_iterator PostIter; 

struct Fitness {
    vector<float> values;
    bool valid;
};

// for unsupervised learning, classification and regression. 
template<typename T> struct Program //: public tree<Node>
{
    /// whether fit has been called
    bool is_fitted_;
    /// fitness 
    Fitness fitness;
    
    // vector<float> fitness_values;
    // bool fitness_valid;

    /// the type of output from the tree object
    using TreeType = conditional_t<std::is_same_v<T,ArrayXXf>, ArrayXXf, ArrayXf>;
    /// the underlying tree
    tree<Node> Tree; 
    /// reference to search space
    std::optional<std::reference_wrapper<SearchSpace>> SSref;

    Program() = default;
    Program(const std::reference_wrapper<SearchSpace> s, const tree<Node> t)
        : Tree(t) 
    {
        SSref = std::optional<std::reference_wrapper<SearchSpace>>{s};
    }

    inline void set_search_space(const std::reference_wrapper<SearchSpace> s)
    {
        SSref = std::optional<std::reference_wrapper<SearchSpace>>{s};
    }

    int size(){
        return Tree.size();
    }

    Program<T>& fit(const Dataset& d)
    {
        TreeType out =  Tree.begin().node->fit<TreeType>(d);
        this->is_fitted_ = true;
        update_weights(d);
        // this->valid = true;
        return *this;
    };

    template <typename R, typename W>
    R predict_with_weights(const Dataset &d, const W** weights)
    {
        if (!is_fitted_)
            HANDLE_ERROR_THROW("Program is not fitted. Call 'fit' first.\n");

        return Tree.begin().node->predict<R>(d, weights);
    };

    auto predict_with_weights(const Dataset &d, const ArrayXf& weights)
    {
        float const * wptr = weights.data(); 
        return this->predict_with_weights<T>(d, &wptr);
    };

    template <typename R = T>
    enable_if_t<is_same_v<R, TreeType>, R>
    predict(const Dataset &d)
    {
        if (!is_fitted_)
            HANDLE_ERROR_THROW("Program is not fitted. Call 'fit' first.\n");

        auto out = Tree.begin().node->predict<TreeType>(d);
        return out;
    };

    /// @brief Specialized predict function for binary classification. 
    /// @tparam R: return type, typically left blank 
    /// @param d : data
    /// @return out: binary labels
    template <typename R = T>
    enable_if_t<is_same_v<R, ArrayXb>, R>
    predict(const Dataset &d)
    {
        if (!is_fitted_)
            HANDLE_ERROR_THROW("Program is not fitted. Call 'fit' first.\n");

        R out = (Tree.begin().node->predict<TreeType>(d) > 0.5);
        return out;
    };

    /// @brief Specialized predict function for multiclass classification. 
    /// @tparam R: return type, typically left blank 
    /// @param d : data
    /// @return out: integer labels
    template <typename R = T>
    enable_if_t<is_same_v<R, ArrayXi>, R>
    predict(const Dataset &d)
    {
        if (!is_fitted_)
            HANDLE_ERROR_THROW("Program is not fitted. Call 'fit' first.\n");

        TreeType out = Tree.begin().node->predict<TreeType>(d);
        auto argmax = Function<NodeType::ArgMax>{};
        R label = argmax(out);
        return label;
    };

    template <typename R = T>
    enable_if_t<is_same_v<R, ArrayXb>, ArrayXf>
    predict_proba(const Dataset &d) { return predict<ArrayXf>(d); };

    /// @brief Convenience function to call fit directly from X,y data.
    /// @param X : Input features
    /// @param y : Labels
    /// @return : reference to program 
    Program<T>& fit(const Ref<const ArrayXXf>& X, const Ref<const ArrayXf>& y)
    {
        Dataset d(X,y);
        return fit(d);
    };

    /// @brief Convenience function to call predict directly from X data.
    /// @param X : Input features
    /// @return : predictions 
    T predict(const Ref<const ArrayXXf>& X)
    {
        Dataset d(X);
        return predict(d);
    };

    template <typename R = T>
    enable_if_t<is_same_v<R, ArrayXb>, ArrayXf>
    predict_proba(const Ref<const ArrayXXf>& X)
    {
        Dataset d(X);
        return predict_proba(d);
    };

    void grad_descent(const ArrayXf& gradient, const Dataset& d)
    {
        //TODO
    };

    void update_weights(const Dataset& d);

    int get_n_weights() const
    {
        int count=0;
        // check tree nodes for weights
        for (PostIter i = Tree.begin_post(); i != Tree.end_post(); ++i)
        {
            const auto& node = i.node->data; 
            if (node.is_weighted)
                count += node.W.size();
        }
        return count;
    }

    /// return the weights of the tree as an array 
    ArrayXf get_weights()
    {
        ArrayXf weights(get_n_weights()); 
        int i = 0;
        for (PostIter t = Tree.begin_post(); t != Tree.end_post(); ++t)
        {
            const auto& node = t.node->data; 
            if (node.is_weighted)
            {
                for (const auto& w: node.W)
                {
                    weights(i) = w;
                    ++i;
                }
            }
        }
        return weights;
    }

    void set_weights(const ArrayXf& weights)
    {
        // take the weights set them in the tree. 
        // return the weights of the tree as an array 
        if (weights.size() != get_n_weights())
            HANDLE_ERROR_THROW("Tried to set_weights of incorrect size");
        for (PostIter i = Tree.begin_post(); i != Tree.end_post(); ++i)
        {
            auto& node = i.node->data; 
            int j = 0;
            if (node.is_weighted)
            {
                for (auto& w: node.W)
                {
                    w = weights(j);
                    ++j;
                }
            }
        }
    }
    string get_model(string fmt="compact", bool pretty=false)
    {
        auto head = Tree.begin(); 
        if (fmt=="tree")
            return head.node->get_tree_model(pretty);
        else if (fmt=="dot")
            return get_dot_model(); ;
        return head.node->get_model(pretty);
    }

    string get_dot_model(){
        string out = "digraph G {\norientation=landscape;\n";

        for (Iter iter = Tree.begin(); iter!=Tree.end(); iter++)
        {
            const auto& parent = iter.node->data;
            auto kid = iter.node->first_child;

            for (int i = 0; i < iter.number_of_children(); ++i)
            {
                string label="";
                if (parent.is_weighted)
                    label = fmt::format("{:.3f}",parent.W.at(i));
                else if (Is<NodeType::SplitOn>(parent.node_type) && i == 0)
                    label = fmt::format("{:.3f}",parent.W.at(i)); 

                out += fmt::format("{} [comment=\"{}\"] -> {} [label=\"{}\"];\n", 
                        parent.get_name(),
                        parent.complete_hash,
                        kid->data.get_name(),
                        label
                        );
                kid = kid->next_sibling;
            }

        }
        out += "}\n";
        return out;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Mutation & Crossover


    // tree<Node> point_mutation(tree<Node>& Tree, Iter spot);
    // tree<Node> insert_mutation(tree<Node>& Tree, Iter spot);
    // tree<Node> delete_mutation(tree<Node>& Tree, Iter spot);

    /// point mutation: replace node with same typed node
    void point_mutation(Iter spot, const SearchSpace& SS)
    {
        // cout << "point mutation\n";
        auto newNode = SS.get_node_like(spot.node->data); 
        this->Tree.replace(spot, newNode);
    };
    /// insert a node with spot as a child
    void insert_mutation(Iter spot, const SearchSpace& SS)
    {
        // cout << "insert mutation\n";
        auto spot_type = spot.node->data.ret_type;
        auto n = SS.get_op_with_arg(spot_type, spot_type); 
        // make node n wrap the subtree at the chosen spot
        auto parent_node = this->Tree.wrap(spot, n);

        // now fill the arguments of n appropriately
        bool spot_filled = false;
        for (auto a: n.arg_types)
        {
            
            if (spot_filled)
            {
                // if spot is in its child position, append children
                this->Tree.append_child(parent_node, SS.get_terminal(a));
            }
            // if types match, treat this spot as filled by the spot node 
            else if (a == spot_type)
                spot_filled = true;
            // otherwise, add siblings before spot node
            else
                this->Tree.insert(spot, SS.get_terminal(a));

        } 
    };
    /// delete subtree and replace it with a terminal of the same return type
    void delete_mutation(Iter spot, const SearchSpace& SS)
    {
        // cout << "delete mutation\n";
        auto terminal = SS.get_terminal(spot.node->data.ret_type); 
        this->Tree.erase_children(spot); 
        this->Tree.replace(spot, terminal);
    };

    // inline Program<T> mutate() const { assert(this->SSref); return mutate(SSref.value()); }

    Program<T> mutate(const SearchSpace& SS) const
    {
        /* Types of mutation:
        * point mutation
        * insertion mutation
        * deletion mutation
        */
        // assert(this->SSref);
        // auto SS = this->SSref.value();
        Program<T> child(*this);

        // choose location by weighted sampling of program
        vector<float> weights(child.Tree.size());
        std::transform(child.Tree.begin(), child.Tree.end(), 
                       weights.begin(),
                       [](const auto& n){ return n.get_prob_change(); }
                      );

        auto spot = r.select_randomly(child.Tree.begin(), child.Tree.end(), 
                                      weights.begin(), weights.end());

        // choose one of these options
        string choice = r.random_choice(PARAMS["mutation_options"]);

        if (choice == "insert")
            child.insert_mutation(spot, SS);
        else if (choice == "delete")
            child.delete_mutation(spot, SS);
        else 
            child.point_mutation(spot, SS);

        return child;
    };
    /// swaps subtrees between this and other (note the pass by copy)
    Program<T> cross(Program<T> other) const
    {
        /* subtree crossover between this and other, producing new Program */
        // choose location by weighted sampling of program
        // TODO: why doesn't this copy the search space reference to child?
        Program<T> child(*this);

        // pick a subtree to replace
        vector<float> child_weights(child.Tree.size());
        std::transform(child.Tree.begin(), child.Tree.end(), 
                       child_weights.begin(),
                       [](const auto& n){ return n.get_prob_change(); }
                      );
        // fmt::print("child weights: {}\n", child_weights);
        bool matching_spots_found = false;
        for (int tries = 0; tries < 3; ++tries)
        {
            auto child_spot = r.select_randomly(child.Tree.begin(), 
                                                child.Tree.end(), 
                                                child_weights.begin(), 
                                                child_weights.end()
                                            );
            auto child_ret_type = child_spot.node->data.ret_type;
            // fmt::print("child_spot : {}\n",child_spot.node->data);
            // fmt::print("child_ret_type: {}\n",child_ret_type);
            // pick a subtree to insert
            // need to pick a node that has a matching output type to the child_spot
            vector<float> other_weights(other.Tree.size());
            std::transform(other.Tree.begin(), other.Tree.end(), 
                other_weights.begin(),
                [child_ret_type](const auto& n){ 
                    if (n.ret_type == child_ret_type)
                        return n.get_prob_change(); 
                    else
                        return float(0.0);
                    }
                );
            for (const auto& w: other_weights)
            {
                matching_spots_found = w > 0.0;
                if (matching_spots_found) 
                    break;
            }
            if (matching_spots_found) 
            {
                auto other_spot = r.select_randomly(
                    other.Tree.begin(), 
                    other.Tree.end(), 
                    other_weights.begin(), 
                    other_weights.end()
                );
                                
                // fmt::print("other_spot : {}\n",other_spot.node->data);
                // swap subtrees at child_spot and other_spot
                child.Tree.move_ontop(child_spot, other_spot);
                return child;
            }
            // fmt::print("try {} failed\n",tries);
        }

        return child;
    };

    /// @brief turns program tree into a linear program. 
    /// @return a vector of nodes encoding the program in reverse polish notation
    vector<Node> linearize() const {
        vector<Node> linear_program;
        for (PostIter i = Tree.begin_post(); i != Tree.end_post(); ++i)
            linear_program.push_back(i.node->data);
        return linear_program;
    }
}; // Program
} // Brush

#include "optimizer/weight_optimizer.h"

namespace Brush {
    template<typename T> 
    void Program<T>::update_weights(const Dataset& d)
    {
        // Updates the weights within a tree. 
        // make an optimizer
        auto WO = WeightOptimizer(); 
        // get new weights from optimization.
        WO.update((*this), d);
    };

    // serialization
    // serialization for program
    template<typename T>
    void to_json(json &j, const Program<T> &p)
    {
        j = json{{"Tree",p.Tree}, {"is_fitted_", p.is_fitted_}}; 
    }

    template<typename T>
    void from_json(const json &j, Program<T>& p)
    {
        j.at("Tree").get_to(p.Tree);
        j.at("is_fitted_").get_to(p.is_fitted_);
    }
} // Brush

#endif
