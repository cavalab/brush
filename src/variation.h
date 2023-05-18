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


/**
 * @brief Namespace for variation functions like crossover and mutation. 
 * 
 */
namespace variation {

typedef tree<Node>::pre_order_iterator Iter; 

/// point mutation: replace node with same typed node
inline void point_mutation(tree<Node>& Tree, Iter spot, const SearchSpace& SS)
{
    // cout << "point mutation\n";

    // get_node_like will sample a similar node based on node_weights or terminal_weights
    auto newNode = SS.get_node_like(spot.node->data); 
    Tree.replace(spot, newNode);
}

/// insert a node with spot as a child
inline void insert_mutation(tree<Node>& Tree, Iter spot, const SearchSpace& SS)
{
    // cout << "insert mutation\n";
    auto spot_type = spot.node->data.ret_type;
    
    // pick a random compatible node to insert (with probabilities given by
    // node_weights). The `-1` represents the node being inserted.
    // Ideally, it should always find at least one match (the same node
    // used as a reference when calling the function). However, we have a 
    // size restriction, which will be relaxed here (just as it is in the PTC2
    // algorithm). This mutation can create a new expression that exceeds the
    // maximum size by the highest arity among the operators.
    auto n = SS.get_op_with_arg(spot_type, spot_type, true,
                                PARAMS["max_size"].get<int>()-Tree.size()-1); 

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

    // get_terminal will sample based on terminal_weights
    auto terminal = SS.get_terminal(spot.node->data.ret_type); 
    Tree.erase_children(spot); 
    Tree.replace(spot, terminal);
};

/// @brief toggle the node's weight on or off. 
/// @param Tree the program tree
/// @param spot an iterator to the node that is being mutated
/// @param SS the search space (unused)
inline void toggle_weight_mutation(tree<Node>& Tree, Iter spot, const SearchSpace& SS)
{
    spot.node->data.is_weighted = !spot.node->data.is_weighted;
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
 * 
 * Every mutation has a probability of occur based on global parameters. The
 * place where the mutation will take place is sampled based on attribute 
 * `get_prob_change` of each node in the tree. Inside each type of mutation, 
 * when a new node is inserted, it is sampled based on `terminal_weights`.
 * 
 * By default, all probability distributions are uniform, but they can be
 * dynamically optimized based on a Multi-Armed Bandit.
 * 
 * @tparam T program type
 * @param parent the program to be mutated
 * @param SS a search space
 * @return `child`, the mutated program
 */
template<ProgramType T>
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

    auto options = PARAMS["mutation_options"].get<std::map<string,float>>();

    // Setting to zero the weight of variations that increase the expression
    // if the expression is already at the maximum size or depth
    if (child.Tree.size()+1 >= PARAMS["max_size"].get<int>()
    ||  child.Tree.depth(spot)+child.Tree.max_depth(spot)+1 >= PARAMS["max_depth"].get<int>())
    {
        // avoid using mutations that increase size/depth 
        options["insert"] = 0.0;
    }

    // choose a valid mutation option
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

/// @brief swaps subtrees between root and other, returning new program 
/// @tparam T the program type
/// @param root the root parent
/// @param other the donating parent
/// @return new program of type `T`
template<ProgramType T>
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
    
    bool matching_spots_found = false;
    for (int tries = 0; tries < 3; ++tries)
    {
        auto child_spot = r.select_randomly(child.Tree.begin(), 
                                            child.Tree.end(), 
                                            child_weights.begin(), 
                                            child_weights.end()
                                        );

        auto child_ret_type = child_spot.node->data.ret_type;

        auto allowed_size  = PARAMS["max_size"].get<int>() -
                             ( child.Tree.size() - child.Tree.size(child_spot) );
        auto allowed_depth = PARAMS["max_depth"].get<int>() - 
                             ( child.Tree.depth(child_spot) );

        // pick a subtree to insert. Selection is based on other_weights
        vector<float> other_weights(other.Tree.size());

        // iterator to get the size of subtrees inside transform
        auto other_iter = other.Tree.begin();

        // lambda function to check feasibility of solution and increment the iterator 
        const auto check_and_incrm = [other, &other_iter, allowed_size, allowed_depth]() -> bool {
            int s = other.Tree.size(other_iter);
            int d = other.Tree.max_depth(other_iter);

            std::advance(other_iter, 1);
            return (s <= allowed_size) && (d <= allowed_depth);
        };

        std::transform(other.Tree.begin(), other.Tree.end(), 
            other_weights.begin(),
            [child_ret_type, check_and_incrm](const auto& n){
                // need to pick a node that has a matching output type to the child_spot.
                // also need to check if swaping this node wouldn't exceed max_size
                if (check_and_incrm() && (n.ret_type == child_ret_type))
                    return n.get_prob_change(); 
                else
                    // setting the weight to zero to indicate a non-feasible crossover point
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