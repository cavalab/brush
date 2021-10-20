/* Brush

copyright 2020 William La Cava
license: GNU/GPL v3
*/
#ifndef PROGRAM_H

#define PROGRAM_H
//external includes

//
#include <iostream>

#include <string>
#include <Eigen/Dense>  // For python factory constructor

// internal includes
#include "tree.h"

// #include "data/data.h"
#include "state.h"
#include "node.h"
#include "search_space.h"
#include "params.h"

using std::cout;
using std::string;
using Brush::data::State;
using Brush::data::Data;
using Brush::SearchSpace;

namespace Brush {


typedef tree<NodeBase*>::pre_order_iterator Iter; 

/* template<typename T> class Program; */

template<typename T>
T RandomDequeue(std::vector<T>& Q)
{
    int loc = r.rnd_int(0, Q.size()-1);
    std::swap(Q[loc], Q[Q.size()-1]);
    T val = Q.back();
    Q.pop_back();
    return val;
}

tree<NodeBase*> make_program(SearchSpace& SS, type_index root_type, 
                            int max_d, int max_breadth, int max_size);

// TODO: instead of templating this, define meaningful derived classes
// for unsupervised learning, classification and regression. 
template<typename T> class Program //: public tree<NodeBase*>
{

    public:

    /// the underlying program
    tree<NodeBase*> prg; 
    /// reference to search space
    SearchSpace& SS;

    // Factory function to construct a Program using the Pybind11 interface
    static Program create_py(MatrixXf X, ArrayXf y, int depth=0, int breadth=0, int size=0)
    {
        Longitudinal Z;
        Data data(X, y, Z);
        SearchSpace SS;
        SS.init(data);

        return Program(SS, depth, breadth, size);
    }
    
    Program(SearchSpace& ss, int depth=0, int breadth = 0, int size = 0): SS(ss)
    {
        // make a random program
        if (depth == 0)
            depth = r.rnd_int(1, params.max_depth);
        if (breadth == 0)
            breadth = r.rnd_int(1, params.max_breadth);
        if (size == 0)
            size = r.rnd_int(1, params.max_size);

        this->prg = make_program(this->SS, typeid(T), depth, breadth, size);
    };

    T fit(const Data& d)
    {
        Iter start = prg.begin(); 
        State out = start.node->fit(d);
        return std::get<T>(out);
    };

    T predict(const Data& d)
    {
        Iter start = prg.begin(); 
        State out = start.node->predict(d);
        cout << "Program::predict returning\n";
        return std::get<T>(out);
    };

    void grad_descent(const ArrayXf& gradient, const Data& d)
    {
        Iter start = prg.begin(); 
        start.node->grad_descent(gradient, d);
    };

    string get_model(bool pretty=false)
    {
        return prg.get_model(pretty);
    }

    string get_tree_model(bool pretty=false)
    {
        return prg.get_tree_model(pretty);
    }

    ////////////////////////////////////////////////////////////////////////////
    // Mutation & Crossover


    // tree<NodeBase*> point_mutation(tree<NodeBase*>& prg, Iter spot);
    // tree<NodeBase*> insert_mutation(tree<NodeBase*>& prg, Iter spot);
    // tree<NodeBase*> delete_mutation(tree<NodeBase*>& prg, Iter spot);

    /// point mutation: replace node with same typed node
    void point_mutation(Iter spot)
    {
        cout << "point mutation\n";
        auto newNode = this->SS.get_node_like(spot.node->data); 
        this->prg.replace(spot, newNode);
    };
    /// insert a node with spot as a child
    void insert_mutation(Iter spot)
    {
        cout << "insert mutation\n";
        auto spot_type = spot.node->data->ret_type();
        auto n = this->SS.get_op_with_arg(spot_type, spot_type); 
        // make node n wrap the subtree at the chosen spot
        auto parent_node = this->prg.wrap(spot, n);

        // now fill the arguments of n appropriately
        bool spot_filled = false;
        for (auto a: n->arg_types())
        {
            
            if (spot_filled)
            {
                // if spot is in its child position, append children
                this->prg.append_child(parent_node, this->SS.get_terminal(a));
            }
            // if types match, treat this spot as filled by the spot node 
            else if (a == spot_type)
                spot_filled = true;
            // otherwise, add siblings before spot node
            else
                this->prg.insert(spot, this->SS.get_terminal(a));

        } 
    };
    /// delete subtree and replace it with a terminal of the same return type
    void delete_mutation(Iter spot)
    {
        cout << "delete mutation\n";
        auto terminal = this->SS.get_terminal(spot.node->data->ret_type()); 
        this->prg.erase_children(spot); 
        this->prg.replace(spot, terminal);
    };

    Program<T> mutate() const
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
                       [](const auto& node){ return node->get_prob_change(); }
                      );

        auto spot = r.select_randomly(child.prg.begin(), child.prg.end(), 
                                      weights.begin(), weights.end());

        // choose one of these options
        string choice = r.random_choice(params.mutation_options);

        if (choice == "insert")
            child.insert_mutation(spot);
        else if (choice == "delete")
            child.delete_mutation(spot);
        else 
            child.point_mutation(spot);

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
                       [](const auto& node){ return node->get_prob_change(); }
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
                       [](const auto& node){ return node->get_prob_keep(); }
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
