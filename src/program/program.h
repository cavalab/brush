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
#include "../vary/search_space.h"
#include "../params.h"
#include "../util/utils.h"
#include "functions.h"
// #include "../variation.h"
// #include "weight_optimizer.h"


using std::cout;
using std::string;
using Brush::Data::Dataset;
using Brush::SearchSpace;

namespace Brush {


typedef tree<Node>::pre_order_iterator Iter; 
typedef tree<Node>::post_order_iterator PostIter; 

using PT = ProgramType;

// for unsupervised learning, classification and regression. 

/**
 * @brief An individual program, a.k.a. model. 
 * 
 * @tparam PType one of the ProgramType enum values. 
 */
template<PT PType> struct Program 
{
    /// @brief an enum storing the program type. 
    static constexpr PT program_type = PType;

    /// @brief the return type of the tree when calling :func:`predict`. 
    using RetType = typename std::conditional_t<PType == PT::Regressor, ArrayXf,
        std::conditional_t<PType == PT::BinaryClassifier, ArrayXb,
        std::conditional_t<PType == PT::MulticlassClassifier, ArrayXi,
        std::conditional_t<PType == PT::Representer, ArrayXXf, ArrayXf
        >>>>;

    /// the type of output from the tree object
    using TreeType = std::conditional_t<PType == PT::BinaryClassifier, ArrayXf,
        std::conditional_t<PType == PT::MulticlassClassifier, ArrayXXf, 
        RetType>>;

    /// whether fit has been called
    bool is_fitted_;

    /// fitness 
    // Fitness fitness;
    
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

    Program<PType> copy() { return Program<PType>(*this); }

    inline void set_search_space(const std::reference_wrapper<SearchSpace> s)
    {
        SSref = std::optional<std::reference_wrapper<SearchSpace>>{s};
    }

    /// @brief count the (recursive) complexity of the program.
    /// @return int complexity.
    int complexity() const{
        auto head = Tree.begin(); 
        
        return head.node->get_complexity();
    }

    /// @brief count the linear complexity of the program.
    /// @return int complexity.
    int linear_complexity() const{
        auto head = Tree.begin(); 
        
        return head.node->get_linear_complexity();
    }

    /// @brief count the tree size of the program, including the weights in weighted nodes.
    /// @param include_weight whether to include the node's weight in the count.
    /// @return int number of nodes.
    int size(bool include_weight=true) const{
        auto head = Tree.begin(); 
        
        return head.node->get_size(include_weight);
    }

    /// @brief count the size of a given subtree, optionally including the
    /// weights in weighted nodes. This function is not exposed to the python wrapper.
    /// @param top root node of the subtree.
    /// @param include_weight whether to include the node's weight in the count.
    /// @return int number of nodes.
    int size_at(Iter& top, bool include_weight=true) const{

        return top.node->get_size(include_weight);
    }

    /// @brief count the tree depth of the program. The depth is not influenced by weighted nodes.
    /// @return int tree depth.
    int depth() const{
        //tree.hh count the number of edges. We need to ensure that a single-node
        //tree has depth>0
        return 1+Tree.max_depth();
    }

    /// @brief count the depth of a given subtree. The depth is not influenced by
    /// weighted nodes. This function is not exposed to the python wrapper.
    /// @param top root node of the subtree.
    /// @return int tree depth.
    int depth_at(Iter& top) const{
        return 1+Tree.max_depth(top);
    }

    /// @brief count the depth until reaching the given subtree. The depth is
    /// not influenced by weighted nodes. This function is not exposed to the python wrapper.
    /// @param top root node of the subtree.
    /// @return int tree depth.
    int depth_to_reach(Iter& top) const{
        return 1+Tree.depth(top);
    }

    Program<PType>& fit(const Dataset& d)
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
        return this->predict_with_weights<RetType>(d, &wptr);
    };

    /**
     * @brief the standard predict function.
     * Returns the output of the Tree directly. 
     * 
     * @tparam R return type, default 
     * @param d dataset
     * @return 
     */
    template <typename R = RetType>
    TreeType predict(const Dataset &d) requires(is_same_v<R, TreeType>)
    {
        if (!is_fitted_)
            HANDLE_ERROR_THROW("Program is not fitted. Call 'fit' first.\n");

        return Tree.begin().node->predict<TreeType>(d);
    };

    /// @brief Specialized predict function for binary classification. 
    /// @tparam R: return type, typically left blank 
    /// @param d : data
    /// @return out: binary labels
    template <typename R = RetType>
    ArrayXb predict(const Dataset &d)   requires(is_same_v<R, ArrayXb>)
    {
        if (!is_fitted_)
            HANDLE_ERROR_THROW("Program is not fitted. Call 'fit' first.\n");
            
        return (Tree.begin().node->predict<TreeType>(d) > 0.5);
    };

    /// @brief Specialized predict function for multiclass classification. 
    /// @tparam R: return type, typically left blank 
    /// @param d : data
    /// @return out: integer labels
    template <typename R = RetType>
    ArrayXi predict(const Dataset &d)   requires(is_same_v<R, ArrayXi>)
    {
        if (!is_fitted_)
            HANDLE_ERROR_THROW("Program is not fitted. Call 'fit' first.\n");

        TreeType out = Tree.begin().node->predict<TreeType>(d);
        auto argmax = Function<NodeType::ArgMax>{};
        return argmax(out);
    };

    // template <typename R = RetType>
    template <PT P = PType>
        requires((P == PT::BinaryClassifier) || (P == PT::MulticlassClassifier))
    TreeType predict_proba(const Dataset &d) 
    { 
        return predict<TreeType>(d); 
    };

    /// @brief Convenience function to call fit directly from X,y data.
    /// @param X : Input features
    /// @param y : Labels
    /// @return : reference to program 
    Program<PType>& fit(const Ref<const ArrayXXf>& X, const Ref<const ArrayXf>& y)
    {
        Dataset d(X,y);
        return fit(d);
    };

    /// @brief Convenience function to call predict directly from X data.
    /// @param X : Input features
    /// @return : predictions 
    RetType predict(const Ref<const ArrayXXf>& X)
    {
        Dataset d(X);
        return predict(d);
    };

    /**
     * @brief Predict probabilities from X.
     * 
     * Requires a BinaryClassifier or MulticlassClassifier.
     * 
     * @tparam P parameter for type checking, typically left blank. 
     */
    template <PT P = PType>
        requires((P == PT::BinaryClassifier) || (P == PT::MulticlassClassifier))
    TreeType predict_proba(const Ref<const ArrayXXf>& X) 
    {
        Dataset d(X);
        return predict_proba(d);
    };

    /**
     * @brief Updates the program's weights using non-linear least squares.
     * 
     * @param d the dataset
     */
    void update_weights(const Dataset& d);

    /// @brief returns the number of weights in the program.
    int get_n_weights() const
    {
        int count=0;
        // check tree nodes for weights
        for (PostIter i = Tree.begin_post(); i != Tree.end_post(); ++i)
        {
            const auto& node = i.node->data; 
            // some nodes cannot have their weights optimized, others must have
            if ( Is<NodeType::OffsetSum>(node.node_type)
            ||   (node.get_is_weighted() && IsWeighable(node.node_type)) )
                ++count;
        }
        return count;
    }

    /**
     * @brief Get the weights of the tree as an array
     * 
     * @return ArrayXf of weights in the program, encoded in post-fix order. 
     */
    ArrayXf get_weights()
    {
        ArrayXf weights(get_n_weights()); 
        int i = 0;
        for (PostIter t = Tree.begin_post(); t != Tree.end_post(); ++t)
        {
            const auto& node = t.node->data; 
            if ( Is<NodeType::OffsetSum>(node.node_type)
            ||   (node.get_is_weighted() && IsWeighable(node.node_type)) )
            {
                weights(i) = node.W;
                ++i;
            }
        }
        return weights;
    }

    /**
     * @brief Set the weights in the tree from an array of weights. 
     * 
     * @param weights an array of weights. The number of the weights in the tree 
     *  must match the length of `weights`.
     */
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
            if ( Is<NodeType::OffsetSum>(node.node_type)
            ||   (node.get_is_weighted() && IsWeighable(node.node_type)) )
            {
                node.W = weights(j);
                ++j;
            }
        }
    }

    /**
     * @brief Iterates over the program, locking the nodes until it reaches
     * a certain depth.
     * 
     * @param end_depth the depth to stop locking nodes. Default 0.
     * @param skip_leaves whether to skip leaves and leave them unlocked. 
     * Default true.  
     */
    void lock_nodes(int end_depth=0, bool skip_leaves=true)
    {
        // iterate over the nodes, locking them if their depth does not exceed end_depth.
        if (end_depth<=0)
            return;

        // we need the iterator to calculate the depth, but 
        // the lambda below iterate using nodes. So we are creating an iterator
        // and using it to access depth.
        auto tree_iter = Tree.begin();

        std::for_each(Tree.begin(), Tree.end(),
            [&](auto& n){ 
                auto d = Tree.depth(tree_iter);
                std::advance(tree_iter, 1);

                if (skip_leaves && IsLeaf(n.node_type))
                    return;

                if (d<=end_depth)
                    n.fixed = true; 
                    // n.set_prob_change(0.0f); 
            }
        );
    }

    /**
     * @brief Iterates over the program, unlocking the nodes until it reaches
     * a certain depth. It does not protect the root nodes of logistic regression
     * models.
     * 
     * @param start_depth the depth to start unlocking nodes. Default 0.
     */
    void unlock_nodes(int start_depth=0)
    {
        auto tree_iter = Tree.begin();

        std::for_each(Tree.begin(), Tree.end(),
            [&](auto& n){ 
                auto d = Tree.depth(tree_iter);
                std::advance(tree_iter, 1);

                if (d>=start_depth)                
                    n.fixed = false; 
                    // n.set_prob_change(1.0f); 
            }
        );
    }

    /**
     * @brief Get the model as a string
     * 
     * @param fmt one of "compact", "tree", or "dot". Default "compact".  
     * 
     *  - *compact* : the program as an equation. 
     *  - *tree* : the program as a (small batch, artisinal) tree. 
     *  - *dot* : the program in the dot language for downstream visualization.
     * 
     * @param pretty currently unused. 
     * @return string the model in string form.  
     */
    string get_model(string fmt="compact", bool pretty=false) const
    {
        auto head = Tree.begin(); 
        if (fmt=="tree")
            return head.node->get_tree_model(pretty);
        else if (fmt=="dot")
            return get_dot_model(); ;
        return head.node->get_model(pretty);
    }

    /**
     * @brief Get the model as a dot object
     * 
     * @param extras extra code passed to the beginning of the dot code. 
     * @return string the model in dot language. 
     */
    string get_dot_model(string extras="") const
    {
        // TODO: make the node names their hash or index, and the node label the nodetype name. 
        // ref: https://stackoverflow.com/questions/10579041/graphviz-create-new-node-with-this-same-label#10579155
        string out = "digraph G {\n";
        if (! extras.empty())
            out += fmt::format("{}\n", extras);

        auto get_id = [](const auto& n){
            if (Is<NodeType::Terminal>(n->data.node_type)) 
                return n->data.get_name(false);

            return fmt::format("{}",fmt::ptr(n)).substr(2);
        };
        // bool first = true;
        std::map<string, unsigned int> node_count;
        int i = 0;
        for (Iter iter = Tree.begin(); iter!=Tree.end(); iter++)
        {
            const auto& parent = iter.node;
            // const auto& parent_data = iter.node->data;

            string parent_id = get_id(parent);
            // if (Is<NodeType::Terminal>(parent_data.node_type)) 
            //     parent_id = parent_data.get_name(false);
            // else{
            //     parent_id = fmt::format("{}",fmt::ptr(iter.node)).substr(2);
            // }
            // // parent_id = parent_id.substr(2);

            // if the first node is weighted, make a dummy output node so that the 
            // first node's weight can be shown
            if (i==0 && parent->data.get_is_weighted())
            {
                out += "y [shape=box];\n";
                out += fmt::format("y -> \"{}\" [label=\"{:.2f}\"];\n", 
                        // parent_data.get_name(false),
                        parent_id,
                        parent->data.W
                        );
            }

            // add the node
            bool is_constant = Is<NodeType::Constant, NodeType::MeanLabel>(parent->data.node_type);
            string node_label = parent->data.get_name(is_constant);

            if (Is<NodeType::SplitBest>(parent->data.node_type)){
                node_label = fmt::format("{}>{:.2f}?", parent->data.get_feature(), parent->data.W); 
            }
            if (Is<NodeType::OffsetSum>(parent->data.node_type)){
                node_label = fmt::format("Add"); 
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
                
                string kid_id = get_id(kid);
                // string kid_id = fmt::format("{}",fmt::ptr(kid));
                // kid_id = kid_id.substr(2);

                if (kid->data.get_is_weighted()
                && Isnt<NodeType::Constant, NodeType::MeanLabel, 
                        NodeType::OffsetSum, NodeType::SplitBest>(kid->data.node_type))
                {
                    edge_label = fmt::format("{:.2f}",kid->data.W);
                }

                if (Is<NodeType::SplitOn>(parent->data.node_type)){
                    use_head_tail_labels=true;
                    if (j == 0)
                        tail_label = fmt::format(">{:.2f}",parent->data.W); 
                    else if (j==1)
                        tail_label = "Y"; 
                    else
                        tail_label = "N";

                    head_label=edge_label;
                }
                else if (Is<NodeType::SplitBest>(parent->data.node_type)){
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
        
            // adding the offset as the last child
            if (Is<NodeType::OffsetSum>(parent->data.node_type)){
                // drawing the edge
                out += fmt::format("\"{}\" -> \"{}\" [label=\"\"];\n", 
                        parent_id,
                        parent_id+"Offset"
                        );
                        
                // drawing the node
                out += fmt::format("\"{}\" [label=\"{:.2f}\"];\n",
                        parent_id+"Offset",
                        parent->data.W
                        ); 
            }
                        
            ++i;
        }
        out += "}\n";
        return out;
    }

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
// #include "../variation.h"
namespace Brush{

template<ProgramType PType> 
void Program<PType>::update_weights(const Dataset& d)
{
    // Updates the weights within a tree. 
    // make an optimizer
    auto WO = WeightOptimizer(); 
    // get new weights from optimization.
    WO.update((*this), d);
};


////////////////////////////////////////////////////////////////////////////////
// serialization
// serialization for program
template<ProgramType PType>
void to_json(json &j, const Program<PType> &p)
{
    j = json{{"Tree",p.Tree}, {"is_fitted_", p.is_fitted_}}; 
}

template<ProgramType PType>
void from_json(const json &j, Program<PType>& p)
{
    j.at("Tree").get_to(p.Tree);
    j.at("is_fitted_").get_to(p.is_fitted_);
}

}//namespace Brush



#endif
