/* Brush

copyright 2020 William La Cava
license: GNU/GPL v3
*/
#ifndef VARIATION_H
#define VARIATION_H

// #include "search_space.h"
// #include "program/program.h"
// #include "program/tree_node.h"
// #include "node.h"

// namespace Brush{

// typedef tree<Node>::pre_order_iterator Iter; 

////////////////////////////////////////////////////////////////////////////
// Mutation & Crossover


// tree<Node> point_mutation(tree<Node>& Tree, Iter spot);
// tree<Node> insert_mutation(tree<Node>& Tree, Iter spot);
// tree<Node> delete_mutation(tree<Node>& Tree, Iter spot);
namespace variation {

typedef tree<Node>::pre_order_iterator Iter; 

/// point mutation: replace node with same typed node
inline void point_mutation(tree<Node>& Tree, Iter spot, const SearchSpace& SS)
{
    // cout << "point mutation\n";
    auto newNode = SS.get_node_like(spot.node->data); 
    Tree.replace(spot, newNode);
}

/// insert a node with spot as a child
inline void insert_mutation(tree<Node>& Tree, Iter spot, const SearchSpace& SS)
{
    // cout << "insert mutation\n";
    auto spot_type = spot.node->data.ret_type;
    auto n = SS.get_op_with_arg(spot_type, spot_type); 
    // make node n wrap the subtree at the chosen spot
    auto parent_node = Tree.wrap(spot, n);

    // now fill the arguments of n appropriately
    bool spot_filled = false;
    for (auto a: n.arg_types)
    {
        
        if (spot_filled)
        {
            // if spot is in its child position, append children
            Tree.append_child(parent_node, SS.get_terminal(a));
        }
        // if types match, treat this spot as filled by the spot node 
        else if (a == spot_type)
            spot_filled = true;
        // otherwise, add siblings before spot node
        else
            Tree.insert(spot, SS.get_terminal(a));

    } 
}

/// delete subtree and replace it with a terminal of the same return type
inline void delete_mutation(tree<Node>& Tree, Iter spot, const SearchSpace& SS)
{
    // cout << "delete mutation\n";
    auto terminal = SS.get_terminal(spot.node->data.ret_type); 
    Tree.erase_children(spot); 
    Tree.replace(spot, terminal);
};

inline void toggle_weight_mutation(tree<Node>& Tree, Iter spot, const SearchSpace& SS)
{
    fmt::print("Tree before toggle weight: {}\n",Tree);
    fmt::print("spot: {}\n",spot.node->data.get_name());
    spot.node->data.is_weighted = !spot.node->data.is_weighted;
    fmt::print("toggled spot: {}\n",spot.node->data.get_name());
    fmt::print("Tree after toggle weight:  {}\n",Tree);
}

/**
 * @brief Mutate a program.
 * 
 * Types of mutation:
 * 
 *  - point mutation changes a single node. 
 *  - insertion mutation inserts a node as the parent of an existing node, and fills in the other arguments. 
 *  - deletion mutation deletes a node
 *  - toggle_weight mutation turns a node's weight on or off. 
 * @tparam T program type
 * @param parent the program to be mutated
 * @param SS a search space
 * @return `child`, the mutated program
 */
template<typename T>
Program<T> mutate(const Program<T>& parent, const SearchSpace& SS)
{
    Program<T> child(parent);

    // choose location by weighted sampling of program
    vector<float> weights(child.Tree.size());
    std::transform(child.Tree.begin(), child.Tree.end(), 
                    weights.begin(),
                    [](const auto& n){ return n.get_prob_change(); }
                    );

    auto spot = r.select_randomly(child.Tree.begin(), child.Tree.end(), 
                                    weights.begin(), weights.end());

    // choose one of these options
    auto options = PARAMS["mutation_options"].get<std::map<string,float>>();
    string choice = r.random_choice(options);

    if (choice == "insert")
        insert_mutation(child.Tree, spot, SS);
    else if (choice == "delete")
        delete_mutation(child.Tree, spot, SS);
    else if (choice == "point") 
        point_mutation(child.Tree, spot, SS);
    else if (choice == "toggle_weight") 
        toggle_weight_mutation(child.Tree, spot, SS);
    else{
        string msg = fmt::format("{} not a valid mutation choice", choice);
        HANDLE_ERROR_THROW(msg);
    }


    return child;
};
/// swaps subtrees between this and other (note the pass by copy)
template<typename T>
Program<T> cross(const Program<T>& root, const Program<T>& other) 
{
    /* subtree crossover between this and other, producing new Program */
    // choose location by weighted sampling of program
    // TODO: why doesn't this copy the search space reference to child?
    Program<T> child(root);

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
} //namespace vary
#endif