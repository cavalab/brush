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

#include <map>

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
inline bool point_mutation(tree<Node>& Tree, Iter spot, const SearchSpace& SS)
{
    // cout << "point mutation\n";

    // get_node_like will sample a similar node based on node_map_weights or
    // terminal_weights, and maybe will return a Node.
    std::optional<Node> newNode = SS.get_node_like(spot.node->data);

    // if optional contains a Node, we access its contained value
    if (newNode) {
        Tree.replace(spot, *newNode);
        return true;
    }

    // in case mutation fails
    return false;
}

/// insert a node with spot as a child
inline bool insert_mutation(tree<Node>& Tree, Iter spot, const SearchSpace& SS)
{
    // cout << "insert mutation\n";
    auto spot_type = spot.node->data.ret_type;
    
    // pick a random compatible node to insert (with probabilities given by
    // node_map_weights). The `-1` represents the node being inserted.
    // Ideally, it should always find at least one match (the same node
    // used as a reference when calling the function). However, we have a 
    // size restriction, which will be relaxed here (just as it is in the PTC2
    // algorithm). This mutation can create a new expression that exceeds the
    // maximum size by the highest arity among the operators.
    std::optional<Node> n = SS.get_op_with_arg(spot_type, spot_type, true,
                                PARAMS["max_size"].get<int>()-Tree.size()-1); 

    if (!n) // there is no operator with compatible arguments
        return false;

    // make node n wrap the subtree at the chosen spot
    auto parent_node = Tree.wrap(spot, *n);

    // now fill the arguments of n appropriately
    bool spot_filled = false;
    for (auto a: (*n).arg_types)
    {
        if (spot_filled)
        {
            // if spot is in its child position, append children.
            // reminding that get_terminal may fail as well
            Tree.append_child(parent_node, SS.get_terminal(a));
        }
        // if types match, treat this spot as filled by the spot node 
        else if (a == spot_type)
            spot_filled = true;
        // otherwise, add siblings before spot node
        else
            Tree.insert(spot, SS.get_terminal(a));
    } 

    return true;
}

/// delete subtree and replace it with a terminal of the same return type
inline bool delete_mutation(tree<Node>& Tree, Iter spot, const SearchSpace& SS)
{
    // cout << "delete mutation\n";

    // get_terminal will sample based on terminal_weights
    auto terminal = SS.get_terminal(spot.node->data.ret_type); 
    
    Tree.erase_children(spot); 

    // TODO: this may fail. I need to return optional here as well
    Tree.replace(spot, terminal);

    return true;
};

/// @brief toggle the node's weight on or off. 
/// @param Tree the program tree
/// @param spot an iterator to the node that is being mutated
/// @param SS the search space (unused)
inline bool toggle_weight_mutation(tree<Node>& Tree, Iter spot, const SearchSpace& SS)
{
    spot.node->data.is_weighted = !spot.node->data.is_weighted;

    return true;
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
std::optional<Program<T>> mutate(const Program<T>& parent, const SearchSpace& SS)
{
    Program<T> child(parent);

    // TODO: update documentation

    // choose location by weighted sampling of program
    vector<float> weights(child.Tree.size());
    std::transform(child.Tree.begin(), child.Tree.end(), 
                    weights.begin(),
                    [](const auto& n){ return n.get_prob_change(); }
                    );

    auto spot = r.select_randomly(child.Tree.begin(), child.Tree.end(), 
                                    weights.begin(), weights.end());

    auto options = PARAMS["mutation_options"].get<std::map<string,float>>();

    // these restrictions below increase the performance

    // don't increase an expression already at its maximum size!!
    // Setting to zero the weight of variations that increase the expression
    // if the expression is already at the maximum size or depth
    if (child.Tree.size()+1      >= PARAMS["max_size"].get<int>()
    ||  child.Tree.max_depth()+1 >= PARAMS["max_depth"].get<int>())
    {
        // avoid using mutations that increase size/depth. New mutations that
        // has similar behavior should be listed here.
        options["insert"] = 0.0;
    }

    // don't shrink an expression already at its minimum size
    if (child.Tree.size() <= 1 || child.Tree.max_depth() <= 1)
    {
        // avoid using mutations that decrease size/depth. New mutations that
        // has similar behavior should be listed here.
        options["delete"] = 0.0;
    }

    // choose a valid mutation option
    string choice = r.random_choice(options);

    // Every mutation here works inplace, so they return bool instead of
    // std::optional to indicare the result of their manipulation over the
    // program tree. Here we call the mutation function and return the result
    using MutationFunc = std::function<bool(tree<Node>&, Iter, const SearchSpace&)>;

    std::map<std::string, MutationFunc> mutations{
        {"insert",        insert_mutation},
        {"delete",        delete_mutation},
        {"point",         point_mutation},
        {"toggle_weight", toggle_weight_mutation}
    };

    // Try to find the mutation function based on the choice
    auto it = mutations.find(choice);
    if (it == mutations.end()) {
        std::string msg = fmt::format("{} not a valid mutation choice", choice);
        HANDLE_ERROR_THROW(msg);
    }

    bool success = it->second(child.Tree, spot, SS);
    if (success
    && ((child.Tree.size()      <= PARAMS["max_size"].get<int>())
    &&  (child.Tree.max_depth() <= PARAMS["max_depth"].get<int>())) ){
        return child;
    } else {
        return std::nullopt;
    }
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
} //namespace variation
#endif