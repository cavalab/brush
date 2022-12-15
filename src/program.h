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
#include "init.h"
#include "tree_node.h"
#include "state.h"
#include "node.h"
#include "search_space.h"
#include "params.h"
#include "util/utils.h"


using std::cout;
using std::string;
using Brush::Data::State;
using Brush::Data::Dataset;
using Brush::SearchSpace;

namespace Brush {


typedef tree<Node>::pre_order_iterator Iter; 

/* template<typename T> class Program; */


/* tree<Node> make_program(SearchSpace& SS, DataType root_type, */ 
/*                             int max_d, int max_breadth, int max_size); */

// TODO: instead of templating this, define meaningful derived classes
// for unsupervised learning, classification and regression. 
template<typename T> struct Program //: public tree<Node>
{
    bool is_fitted_;
    /* using RetType = typename DataTypeType<T>::type; */
    const DataType RetDataType = DataTypeEnum<T>::value;

    /// the underlying program
    tree<Node> prg; 
    /// reference to search space
    std::optional<std::reference_wrapper<SearchSpace>> SSref;

    /* Program(SearchSpace& ss, int depth=0, int breadth = 0, int size = 0): SS(ss) */
    /* { */

    /*     this->prg = make_program(this->SS, RetDataType, depth, breadth, size); */
    /* }; */
    Program() = default;
    Program(const std::reference_wrapper<SearchSpace> s, const tree<Node> t)
        : prg(t) 
    {
        SSref = std::optional<std::reference_wrapper<SearchSpace>>{s};
    }

    T fit(const Dataset& d)
    {
        fmt::print("Fitting {}\n", this->get_model());
        T out =  prg.begin().node->fit<T>(d);
        fmt::print("Output {}\n", out);
        is_fitted_ = true;
        return out;
    };

    void update_weights(const Dataset& d)
    {
        // Updates the weights within a tree. 

        // get a copy of the weights from the tree. 
        auto weights = this->get_weights();
        // make an optimizer
        auto WO = WeightOptimizer(); 
        // get new weights from optimization.
        auto new_weights = WO.update(d, this->prg, weights);
        this->set_weights(new_weights);
    }

    auto get_weights()
    {
        // return the weights of the tree as an array (probably eigen array)
    }

    auto set_weights(const T& weights)
    {
        // take the weights updated by WeightOptimizer and set them in the tree. 
    }

    T predict(const Dataset& d)
    {
        if (!is_fitted_)
            HANDLE_ERROR_THROW("Program is not fitted. Call 'fit' first.\n");

        fmt::print("Predicting {}\n", this->get_model());
        T out = prg.begin().node->predict<T>(d);
        return out;
    };

    T fit(const Ref<const ArrayXXf>& X, const Ref<const ArrayXf>& y)
    {
        Dataset d(X,y);
        return fit(d);
    };

    T predict(const Ref<const ArrayXXf>& X, const Ref<const ArrayXf>& y)
    {
        Dataset d(X,y);
        return predict(d);
    };

    void grad_descent(const ArrayXf& gradient, const Dataset& d)
    {
        //TODO
    };

    string get_model(string fmt="compact", bool pretty=false)
    {
        auto head = prg.begin(); 
        if (fmt=="tree")
            return head.node->get_tree_model(pretty);
        else if (fmt=="dot")
            return get_dot_model(); ;
        return head.node->get_model(pretty);
    }

    string get_dot_model(){
        string out = "digraph G {\norientation=landscape;\n";

        for (Iter iter = prg.begin(); iter!=prg.end(); iter++)
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


    // tree<Node> point_mutation(tree<Node>& prg, Iter spot);
    // tree<Node> insert_mutation(tree<Node>& prg, Iter spot);
    // tree<Node> delete_mutation(tree<Node>& prg, Iter spot);

    /// point mutation: replace node with same typed node
    void point_mutation(Iter spot, const SearchSpace& SS)
    {
        cout << "point mutation\n";
        auto newNode = SS.get_node_like(spot.node->data); 
        this->prg.replace(spot, newNode);
    };
    /// insert a node with spot as a child
    void insert_mutation(Iter spot, const SearchSpace& SS)
    {
        cout << "insert mutation\n";
        auto spot_type = spot.node->data.ret_type;
        auto n = SS.get_op_with_arg(spot_type, spot_type); 
        // make node n wrap the subtree at the chosen spot
        auto parent_node = this->prg.wrap(spot, n);

        // now fill the arguments of n appropriately
        bool spot_filled = false;
        for (auto a: n.arg_types)
        {
            
            if (spot_filled)
            {
                // if spot is in its child position, append children
                this->prg.append_child(parent_node, SS.get_terminal(a));
            }
            // if types match, treat this spot as filled by the spot node 
            else if (a == spot_type)
                spot_filled = true;
            // otherwise, add siblings before spot node
            else
                this->prg.insert(spot, SS.get_terminal(a));

        } 
    };
    /// delete subtree and replace it with a terminal of the same return type
    void delete_mutation(Iter spot, const SearchSpace& SS)
    {
        cout << "delete mutation\n";
        auto terminal = SS.get_terminal(spot.node->data.ret_type); 
        this->prg.erase_children(spot); 
        this->prg.replace(spot, terminal);
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
        vector<float> weights(child.prg.size());
        std::transform(child.prg.begin(), child.prg.end(), 
                       weights.begin(),
                       [](const auto& n){ return n.get_prob_change(); }
                      );

        auto spot = r.select_randomly(child.prg.begin(), child.prg.end(), 
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
        vector<float> child_weights(child.prg.size());
        std::transform(child.prg.begin(), child.prg.end(), 
                       child_weights.begin(),
                       [](const auto& n){ return n.get_prob_change(); }
                      );

        auto child_spot = r.select_randomly(child.prg.begin(), 
                                            child.prg.end(), 
                                            child_weights.begin(), 
                                            child_weights.end()
                                           );
        // pick a subtree to insert
        vector<float> other_weights(other.prg.size());
        std::transform(other.prg.begin(), other.prg.end(), 
                       other_weights.begin(),
                       [](const auto& n){ return n.get_prob_change(); }
                      );

        auto other_spot = r.select_randomly(other.prg.begin(), 
                                            other.prg.end(), 
                                            other_weights.begin(), 
                                            other_weights.end()
                                           );
                        
        // swap subtrees at child_spot and other_spot
        child.prg.move_ontop(child_spot, other_spot);

        return child;
    };
}; // Program

typedef Program<ArrayXXf> MultiProgram;

} // Brush

#endif
