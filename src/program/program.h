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

        return Tree.begin().node->predict<TreeType>(d);
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

        return (Tree.begin().node->predict<TreeType>(d) > 0.5);
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
        return argmax(out);
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
                // count += node.W.size();
                ++count;
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
                weights(i) = node.W;
                ++i;
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
        int j = 0;
        for (PostIter i = Tree.begin_post(); i != Tree.end_post(); ++i)
        {
            auto& node = i.node->data; 
            if (node.is_weighted)
            {
                node.W = weights(j);
                ++j;
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

    string get_dot_model()
    {
        // TODO: make the node names their hash or index, and the node label the nodetype name. 
        // ref: https://stackoverflow.com/questions/10579041/graphviz-create-new-node-with-this-same-label#10579155
        string out = "digraph G {\n";
        // bool first = true;
        std::map<string, unsigned int> node_count;
        int i = 0;
        for (Iter iter = Tree.begin(); iter!=Tree.end(); iter++)
        {
            const auto& parent_data = iter.node->data;


            string parent_id = fmt::format("{}",fmt::ptr(iter.node));
            parent_id = parent_id.substr(2);


            // if the first node is weighted, make a dummy output node so that the 
            // first node's weight can be shown
            if (i==0 && parent_data.is_weighted)
            {
                out += "y [shape=box];\n";
                out += fmt::format("y -> \"{}\" [label=\"{:.2f}\"];\n", 
                        // parent_data.get_name(false),
                        parent_id,
                        parent_data.W
                        );

            }

            // add the node
            bool is_constant = Is<NodeType::Constant>(parent_data.node_type);
            string node_label = parent_data.get_name(is_constant);

            if (Is<NodeType::SplitBest>(parent_data.node_type)){
                node_label = fmt::format("{}>{:.2f}?", parent_data.get_feature(), parent_data.W); 
            }
            out += fmt::format("\"{}\" [label=\"{}\"];\n", parent_id, node_label); 

            // add edges to the node's children
            auto kid = iter.node->first_child;
            for (int j = 0; j < iter.number_of_children(); ++j)
            {
                string edge_label="";
                string head_label="";
                string tail_label="";
                bool use_head_tail_labels = false;
                
                string kid_id = fmt::format("{}",fmt::ptr(kid));
                kid_id = kid_id.substr(2);

                if (kid->data.is_weighted && Isnt<NodeType::Constant>(kid->data.node_type)){
                    edge_label = fmt::format("{:.2f}",kid->data.W);
                }

                if (Is<NodeType::SplitOn>(parent_data.node_type)){
                    use_head_tail_labels=true;
                    if (j == 0)
                        tail_label = fmt::format(">{:.2f}",parent_data.W); 
                    else if (j==1)
                        tail_label = "Y"; 
                    else
                        tail_label = "N";

                    head_label=edge_label;
                }
                else if (Is<NodeType::SplitBest>(parent_data.node_type)){
                    use_head_tail_labels=true;
                    if (j == 0){
                        tail_label = "Y"; 
                    }
                    else
                        tail_label = "N";

                    head_label = edge_label;
                }

                if (use_head_tail_labels){
                    out += fmt::format("\"{}\" -> \"{}\" [headlabel=\"{}\",taillabel=\"{}\"];\n", 
                            parent_id,
                            kid_id,
                            head_label,
                            tail_label
                            );

                }
                else{
                    out += fmt::format("\"{}\" -> \"{}\" [label=\"{}\"];\n", 
                            parent_id,
                            kid_id,
                            edge_label
                            );
                }
                kid = kid->next_sibling;
            }
            ++i;
        }
        out += "}\n";
        return out;
    }

    ////////////////////////////////////////////////////////////////////////////
    // Mutation & Crossover

    Program<T> mutate() const;
    /// swaps subtrees between this and other (note the pass by copy)
    Program<T> cross(Program<T> other) const;

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

////////////////////////////////////////////////////////////////////////////////
// weight optimization
#include "optimizer/weight_optimizer.h"
namespace Brush{

template<typename T> 
void Program<T>::update_weights(const Dataset& d)
{
    // Updates the weights within a tree. 
    // make an optimizer
    auto WO = WeightOptimizer(); 
    // get new weights from optimization.
    WO.update((*this), d);
};

////////////////////////////////////////////////////////////////////////////////
// mutation and crossover
#include "../variation.h"
template<typename T>
Program<T> Program<T>::mutate() const
{
    return variation::mutate(*this, this->SSref.value().get());
};

/// swaps subtrees between this and other (note the pass by copy)
template<typename T>
Program<T> Program<T>::cross(Program<T> other) const
{
    return variation::cross(*this, other);
};


////////////////////////////////////////////////////////////////////////////////
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

}//namespace Brush

#endif
