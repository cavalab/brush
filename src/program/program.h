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


using std::cout;
using std::string;
using Brush::Data::State;
using Brush::Data::Dataset;
using Brush::SearchSpace;

namespace Brush {


typedef tree<Node>::pre_order_iterator Iter; 

// for unsupervised learning, classification and regression. 
template<typename T> struct Program //: public tree<Node>
{
    /// @brief whether fit has been called
    bool is_fitted_;
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

    Program<T>& fit(const Dataset& d)
    {
        TreeType out =  Tree.begin().node->fit<TreeType>(d);
        is_fitted_ = true;
        return *this;
    };

    template <typename R = T>
    enable_if_t<is_same_v<R, TreeType>, R>
    predict(const Dataset &d)
    {
        if (!is_fitted_)
            HANDLE_ERROR_THROW("Program is not fitted. Call 'fit' first.\n");

        R out = Tree.begin().node->predict<TreeType>(d);
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
        cout << "point mutation\n";
        auto newNode = SS.get_node_like(spot.node->data); 
        this->Tree.replace(spot, newNode);
    };
    /// insert a node with spot as a child
    void insert_mutation(Iter spot, const SearchSpace& SS)
    {
        cout << "insert mutation\n";
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
        cout << "delete mutation\n";
        auto terminal = SS.get_terminal(spot.node->data.ret_type); 
        this->Tree.erase_children(spot); 
        this->Tree.replace(spot, terminal);
    };

    inline Program<T> mutate() const { assert(this->SSref); return mutate(SSref.value()); }

    Program<T> mutate(const SearchSpace& SS) const
    {
        /* Types of mutation:
        * point mutation
        * insertion mutation
        * deletion mutation
        */

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
        string choice = r.random_choice(params.mutation_options);

        if (choice == "insert")
            child.insert_mutation(spot, SS);
        else if (choice == "delete")
            child.delete_mutation(spot, SS);
        else 
            child.point_mutation(spot, SS);

        return child;
    };
    /// swaps subtrees between this and other (note the pass by copy)
    Program<T> cross(const Program<T>& other) const
    {
        /* subtree crossover between this and other, producing new Program */
        // choose location by weighted sampling of program
        Program<T> child(*this);

        // pick a subtree to replace
        vector<float> child_weights(child.Tree.size());
        std::transform(child.Tree.begin(), child.Tree.end(), 
                       child_weights.begin(),
                       [](const auto& n){ return n.get_prob_change(); }
                      );

        auto child_spot = r.select_randomly(child.Tree.begin(), 
                                            child.Tree.end(), 
                                            child_weights.begin(), 
                                            child_weights.end()
                                           );
        // pick a subtree to insert
        vector<float> other_weights(other.Tree.size());
        std::transform(other.Tree.begin(), other.Tree.end(), 
                       other_weights.begin(),
                       [](const auto& n){ return n.get_prob_change(); }
                      );

        auto other_spot = r.select_randomly(other.Tree.begin(), 
                                            other.Tree.end(), 
                                            other_weights.begin(), 
                                            other_weights.end()
                                           );
                        
        // swap subtrees at child_spot and other_spot
        child.Tree.move_ontop(child_spot, other_spot);

        return child;
    };
}; // Program

} // Brush

#endif
