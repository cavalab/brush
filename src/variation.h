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
    
class MutationBase {
public:
    using Iter = tree<Node>::pre_order_iterator;

    MutationBase(const SearchSpace& SS, size_t max_size, size_t max_depth)
        : SS_(SS)
        , max_size_(max_size)
        , max_depth_(max_depth)
    {
    }
        
    virtual auto find_spots(tree<Node>& Tree) const -> vector<float>
    {
        vector<float> weights(Tree.size());

        // by default, mutation can happen anywhere, based on node weights
        std::transform(Tree.begin(), Tree.end(), weights.begin(),
                       [&](const auto& n){ return n.get_prob_change();});
        
        // Should have same size as prog.Tree.size, even if all weights <= 0.0
        return weights;
    }

    virtual auto operator()(tree<Node>& Tree, Iter spot) const -> bool = 0;

    auto SS() const -> SearchSpace { return SS_; }
    auto max_size() const -> size_t { return max_size_; }
    auto max_depth() const -> size_t{ return max_depth_; }
protected:
    static size_t size_with_weights(tree<Node>& Tree, bool include_weight=true)
    {
        // re-implementation of int Node::size(bool include_weight=true) meant
        // to work with the tree<Node> instead of brush's programs.
        // TODO: find a better way to have this function available to mutations
        // and avoid repeated functions
        size_t acc = 0;

        std::for_each(Tree.begin(), Tree.end(), 
            [include_weight, &acc](auto& node){ 
                ++acc; // the node operator or terminal
                
                // SplitBest has an optimizable decision tree consisting of 3 nodes
                // (terminal, arithmetic comparison, value) that needs to be taken
                // into account
                if (Is<NodeType::SplitBest>(node.node_type))
                    acc += 3;

                if ( (include_weight && node.get_is_weighted()==true)
                &&   Isnt<NodeType::Constant, NodeType::MeanLabel>(node.node_type) )
                    // Taking into account the weight and multiplication, if enabled.
                    // weighted constants still count as 1 (simpler than constant terminals)
                    acc += 2;
             });

        return acc;
    }

private:
    SearchSpace SS_; // where to sample nodes to change the program

    // constrains
    size_t max_size_;
    size_t max_depth_;
};

/// @brief replace node with same typed node
/// @param prog the program
/// @param Tree the program tree
/// @param spot an iterator to the node that is being mutated
/// @param SS the search space to sample a node like `spot`
/// @return boolean indicating the success (true) or fail (false) of the operation
class PointMutation : public MutationBase
{
public:
    explicit PointMutation(const SearchSpace& SS, size_t max_size=0, size_t max_depth=0)
        : MutationBase(SS, max_size, max_depth)
    {
    }

    auto operator()(tree<Node>& Tree, Iter spot) const -> bool override
    {
        // cout << "point mutation\n";

        // get_node_like will sample a similar node based on node_map_weights or
        // terminal_weights, and maybe will return a Node.
        optional<Node> newNode = SS().get_node_like(spot.node->data);

        if (!newNode) // overload to check if newNode == nullopt
            return false;

        // if optional contains a Node, we access its contained value
        Tree.replace(spot, *newNode);

        return true;
    }
};

/// @brief insert a node with spot as a child
/// @param prog the program
/// @param Tree the program tree
/// @param spot an iterator to the node that is being mutated
/// @param SS the search space to sample a node like `spot`
/// @return boolean indicating the success (true) or fail (false) of the operation
class InsertMutation : public MutationBase
{
public:
    explicit InsertMutation(const SearchSpace& SS, size_t max_size=0, size_t max_depth=0)
        : MutationBase(SS, max_size, max_depth)
    {
    }

    auto find_spots(tree<Node>& Tree) const -> vector<float> override
    {
        vector<float> weights;

        if (size_with_weights(Tree) < max_size()) {
            Iter iter = Tree.begin();
            std::transform(Tree.begin(), Tree.end(), std::back_inserter(weights),
                        [&](const auto& n){ 
                            size_t d = 1+Tree.depth(iter);
                            std::advance(iter, 1);

                            // check if SS holds an operator to avoid failing `check` in sample_op_with_arg
                            if ((d >= max_depth())
                            ||  (SS().node_map.find(n.ret_type) == SS().node_map.end())) {
                                return 0.0f;
                            }
                            else {
                                return n.get_prob_change(); 
                            }
                        });
        }
        else {
            // fill the vector with zeros, since we're already at max_size
            weights.resize(Tree.size());
            std::fill(weights.begin(), weights.end(), 0.0f); 
        }
        
        return weights;
    }

    auto operator()(tree<Node>& Tree, Iter spot) const -> bool override
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
        std::optional<Node> n = SS().sample_op_with_arg(spot_type, spot_type, true,
                                    max_size()-Tree.size()-1); 

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
                // TODO: reminding that sample_terminal may fail as well
                auto opt = SS().sample_terminal(a);

                if (!opt)
                    return false;

                Tree.append_child(parent_node, opt.value());
            }
            // if types match, treat this spot as filled by the spot node 
            else if (a == spot_type)
                spot_filled = true;
            // otherwise, add siblings before spot node
            else {
                auto opt = SS().sample_terminal(a);

                if (!opt)
                    return false;

                Tree.insert(spot, opt.value());
            }
        } 

        return true;
    }
};

/// @brief delete subtree and replace it with a terminal of the same return type
/// @param prog the program
/// @param Tree the program tree
/// @param spot an iterator to the node that is being mutated
/// @param SS the search space to sample a node like `spot`
/// @return boolean indicating the success (true) or fail (false) of the operation
class DeleteMutation : public MutationBase
{
public:
    explicit DeleteMutation(const SearchSpace& SS, size_t max_size=0, size_t max_depth=0)
        : MutationBase(SS, max_size, max_depth)
    {
    }

    auto operator()(tree<Node>& Tree, Iter spot) const -> bool override
    {
        // cout << "delete mutation\n";

        // sample_terminal will sample based on terminal_weights. If it succeeds, 
        // then the new terminal will be in `opt.value()`
        auto opt = SS().sample_terminal(spot.node->data.ret_type); 
        
        if (!opt) // there is no terminal with compatible arguments
            return false;

        Tree.erase_children(spot); 

        Tree.replace(spot, opt.value());

        return true;
    }
};

/// @brief toggle the node's weight ON
/// @param prog the program
/// @param Tree the program tree
/// @param spot an iterator to the node that is being mutated
/// @param SS the search space (unused)
/// @return boolean indicating the success (true) or fail (false) of the operation
class ToggleWeightOnMutation : public MutationBase
{
public:
    explicit ToggleWeightOnMutation(const SearchSpace& SS, size_t max_size=0, size_t max_depth=0)
        : MutationBase(SS, max_size, max_depth)
    {
    }

    auto find_spots(tree<Node>& Tree) const -> vector<float> override
    {
        vector<float> weights(Tree.size());

        if (size_with_weights(Tree) < max_size()) {
            std::transform(Tree.begin(), Tree.end(), weights.begin(),
                        [&](const auto& n){
                            // only weighted nodes can be toggled off
                            if (!n.get_is_weighted()
                            &&  IsWeighable(n.ret_type))
                                return n.get_prob_change();
                            else
                                return 0.0f; 
                        });
        }
        else {
            // fill the vector with zeros, since we're already at max_size
            std::fill(weights.begin(), weights.end(), 0.0f); 
        }

        return weights;
    }

    auto operator()(tree<Node>& Tree, Iter spot) const -> bool override
    {
        // cout << "toggle_weight_on mutation\n";

        if (spot.node->data.get_is_weighted()==true // cant turn on whats already on
        ||  !IsWeighable(spot.node->data.ret_type)) // does not accept weights (e.g. boolean)
            return false; // false indicates that mutation failed and should return std::nullopt

        spot.node->data.set_is_weighted(true);
        return true;
    }
};

/// @brief toggle the node's weight OFF
/// @param prog the program
/// @param Tree the program tree
/// @param spot an iterator to the node that is being mutated
/// @param SS the search space (unused)
/// @return boolean indicating the success (true) or fail (false) of the operation
class ToggleWeightOffMutation : public MutationBase
{
public:
    explicit ToggleWeightOffMutation(const SearchSpace& SS, size_t max_size=0, size_t max_depth=0)
        : MutationBase(SS, max_size, max_depth)
    {
    }

    auto find_spots(tree<Node>& Tree) const -> vector<float> override
    {
        vector<float> weights(Tree.size());

        std::transform(Tree.begin(), Tree.end(), weights.begin(),
                    [&](const auto& n){
                        if (n.get_is_weighted()
                        &&  IsWeighable(n.ret_type))
                            return n.get_prob_change();
                        else
                            return 0.0f;
                    });

        return weights;
    }

    auto operator()(tree<Node>& Tree, Iter spot) const -> bool override
    {
        // cout << "toggle_weight_off mutation\n";

        if (spot.node->data.get_is_weighted()==false)
            return false; 

        spot.node->data.set_is_weighted(false);
        return true;
    }
};

/// @brief replaces the subtree rooted in `spot`
/// @param prog the program
/// @param Tree the program tree
/// @param spot an iterator to the node that is being mutated
/// @param SS the search space to generate a compatible subtree
/// @return boolean indicating the success (true) or fail (false) of the operation
class SubtreeMutation : public MutationBase
{
public:
    explicit SubtreeMutation(const SearchSpace& SS, size_t max_size=0, size_t max_depth=0)
        : MutationBase(SS, max_size, max_depth) // TODO: change order size and depth
    {
    }

    // TODO: make different private functions to find spots and use them. theres too much copy and paste here
    auto find_spots(tree<Node>& Tree) const -> vector<float> override
    {
        vector<float> weights;

        auto node_map = SS().node_map;

        if (size_with_weights(Tree) < max_size()) {
            Iter iter = Tree.begin();
            std::transform(Tree.begin(), Tree.end(), std::back_inserter(weights),
                        [&](const auto& n){ 
                            size_t d = 1+Tree.depth(iter);
                            std::advance(iter, 1);

                            // we need to make sure there's some node to start the subtree
                            if ((d >= max_depth())
                            ||  (SS().node_map.find(n.ret_type) == SS().node_map.end())
                            ||  (SS().node_map.find(n.ret_type) == SS().node_map.end()) )
                                return 0.0f;
                            else
                                return n.get_prob_change(); 
                        });
        }
        else {
            weights.resize(Tree.size());
            std::fill(weights.begin(), weights.end(), 0.0f); 
        }
        
        return weights;
    }

    auto operator()(tree<Node>& Tree, Iter spot) const -> bool override
    {
        // cout << "subtree mutation\n";

        // check if we exceeded the size/depth constrains (without subtracting,
        // to avoid overflow cases if the user sets max_size smaller than arity
        // of smallest operator. The overflow would happen when calculating d and
        // s in the following lines, to choose the PTC2 limits)
        if ( max_size()  <= (Tree.size() - Tree.size(spot))
        ||   max_depth() <= Tree.depth(spot) )
            return false;

        auto spot_type = spot.node->data.ret_type;

        // d and s must be compatible with PTC2 --- they should be based on 
        // tree structure, not program structure
        size_t d = max_depth() - Tree.depth(spot);
        size_t s = max_size() - (Tree.size() - Tree.size(spot));

        s = r.rnd_int(1, s);

        // sample subtree uses PTC2, which operates on depth and size of the tree<Node> 
        // (and not on the program!). we shoudn't care for weights here
        auto subtree = SS().sample_subtree(spot.node->data, d, s); 

        if (!subtree) // there is no terminal with compatible arguments
            return false;

        // if optional contains a Node, we access its contained value
        Tree.erase_children(spot); 
        Tree.replace(spot, subtree.value().begin());

        return true;
    }
};

/**
 * @brief Stochastically mutate a program.
 * 
 * Types of mutation:
 * 
 *  - point mutation changes a single node. 
 *  - insertion mutation inserts a node as the parent of an existing node, and fills in the other arguments. 
 *  - deletion mutation deletes a node.
 *  - subtree mutation inserts a new subtree into the program. 
 *  - toggle_weight_on mutation turns a node's weight ON.
 *  - toggle_weight_off mutation turns a node's weight OFF.
 * 
 * Every mutation has a probability (weight) based on global parameters. The
 * spot where the mutation will take place is sampled based on attribute 
 * `get_prob_change` of each node in the tree. Inside each type of mutation, 
 * when a new node is inserted, it is sampled based on `terminal_weights`.
 * 
 * Due to the stochastic behavior, and the several sampling steps, it may come to
 * a case where the search space does not hold any possible modification to do in
 * the program. In this case, the method returns `std::nullopt` (and has overloads
 * so it can be used in a boolean context).
 * 
 * If the mutation succeeds, the mutated program can be accessed through the
 * `.value()` attribute of the `std::optional`. 
 * 
 * This means that, if you use the mutation as `auto opt = mutate(parent, SS)`,
 * either `opt==false` or `opt.value()` contains the child program.
 * 
 * @tparam T program type
 * @param parent the program to be mutated
 * @param SS a search space
 * @return `std::optional` that may contain the child program of type `T`
 */
template<ProgramType T>
std::optional<Program<T>> mutate(const Program<T>& parent, const SearchSpace& SS)
{
    auto options = PARAMS["mutation_options"].get<std::map<string,float>>();

    // whether we should write everything that happened inside the method
    if (PARAMS.value("write_mutation_trace", false)==true) {
        // Default fields of the trace. Initialize with default values, which are
        // gradually changed throughout the execution of the method.
        PARAMS["mutation_trace"] = json({
            {"parent",           parent.get_model("compact", true)},
            {"mutation_weights", options},
            // default values, to be changed in case mutation works
            {"mutation",         "not selected"},
            {"spot_weights",     "not calculated"},
            {"spot",             "not selected"},
            {"child",            "failed to generate"},
            {"status",           "initialized weight vectors"},
            {"success",          "false"}
        });
    }
    if (std::all_of(options.begin(), options.end(),
                    [](const auto& kv) { return kv.second<=0.0; })
    )
    { // No mutation can be successfully applied to this solution  
        return std::nullopt;
    }

    // choose a valid mutation option
    string choice = r.random_choice(options);

    // TODO: this could be improved
    std::unique_ptr<MutationBase> mutation;
    if (choice == "point") 
        mutation = std::make_unique<PointMutation>(
            SS, PARAMS["max_size"].get<int>(), PARAMS["max_depth"].get<int>());
    else if (choice == "insert") 
        mutation = std::make_unique<InsertMutation>(
            SS, PARAMS["max_size"].get<int>(), PARAMS["max_depth"].get<int>());
    else if (choice == "delete") 
        mutation = std::make_unique<DeleteMutation>(
            SS, PARAMS["max_size"].get<int>(), PARAMS["max_depth"].get<int>());
    else if (choice == "toggle_weight_on") 
        mutation = std::make_unique<ToggleWeightOnMutation>(
            SS, PARAMS["max_size"].get<int>(), PARAMS["max_depth"].get<int>());
    else if (choice == "toggle_weight_off") 
        mutation = std::make_unique<ToggleWeightOffMutation>(
            SS, PARAMS["max_size"].get<int>(), PARAMS["max_depth"].get<int>());
    else if (choice == "subtree") 
        mutation = std::make_unique<SubtreeMutation>(
            SS, PARAMS["max_size"].get<int>(), PARAMS["max_depth"].get<int>());
    else {
        std::string msg = fmt::format("{} not a valid mutation choice", choice);
        HANDLE_ERROR_THROW(msg);
    }

    if (PARAMS.value("write_mutation_trace", false)==true) {
        PARAMS["mutation_trace"]["mutation"] = choice;
    }

    Program<T> child(parent);

    // choose location by weighted sampling of program
    auto weights = mutation->find_spots(child.Tree);

    if (PARAMS.value("write_mutation_trace", false)==true) {
        PARAMS["mutation_trace"]["spot_weights"] = weights;
    }

    if (std::all_of(weights.begin(), weights.end(), [](const auto& w) {
        return w<=0.0;
    }))
    { // There is no spot that has a probability to be selected
        return std::nullopt;
    }

    // apply the mutation and check if it succeeded
    auto spot = r.select_randomly(child.Tree.begin(), child.Tree.end(),
                                  weights.begin(), weights.end());

    if (PARAMS.value("write_mutation_trace", false)==true) {
        PARAMS["mutation_trace"]["spot"] = spot.node->get_model(false);
        PARAMS["mutation_trace"]["status"] = "sampled the spot";
    }
    
    // Every mutation here works inplace, so they return bool instead of
    // std::optional to indicare the result of their manipulation over the
    // program tree. Here we call the mutation function and return the result
    bool success = (*mutation)(child.Tree, spot);

    if (PARAMS.value("write_mutation_trace", false)==true) {
        PARAMS["mutation_trace"]["status"] = "aplied the mutation";
        if (success)
            PARAMS["mutation_trace"]["child"] = child.get_model("compact", true);
    }

    if (success
    && ( (child.size()  <= PARAMS["max_size"].get<int>() )
    &&   (child.depth() <= PARAMS["max_depth"].get<int>()) )){

        // success is true only if mutation returned a valid program
        if (PARAMS.value("write_mutation_trace", false)==true) {
            PARAMS["mutation_trace"]["success"] = true;
        }

        return child;
    } else {
        
        // here we have a string in PARAMS["mutation_trace"]["child"],
        // but success is false since it didnt return an valid program
        if (PARAMS.value("write_mutation_trace", false)==true) {
            PARAMS["mutation_trace"]["status"] = "mutation returned child, but it exceeds max_size or max_depth";
            //fmt::print("{}\n", PARAMS["mutation_trace"].get<json>().dump());
        }
        return std::nullopt;
    }
};

/**
 * @brief Stochastically swaps subtrees between root and other, returning a new program. 
 * 
 * The spot where the cross will take place in the `root` parent is sampled
 * based on attribute `get_prob_change` of each node in the tree. After selecting
 * the cross spot, the program will iterate through the `other` parent searching
 * for all compatible sub-trees to replace.
 * 
 * Due to the stochastic behavior, it may come to a case where there is no 
 * candidate to replace the spot node.  In this case, the method returns
 * `std::nullopt` (and has overloads so it can be used in a boolean context).
 * 
 * If the cross succeeds, the child program can be accessed through the
 * `.value()` attribute of the `std::optional`. 
 * 
 * This means that, if you use the cross as `auto opt = mutate(parent, SS)`,
 * either `opt==false` or `opt.value()` contains the child.
 * 
 * @tparam T the program type
 * @param root the root parent
 * @param other the donating parent
 * @return `std::optional` that may contain the child program of type `T`
 */
template<ProgramType T>
std::optional<Program<T>> cross(const Program<T>& mom, const Program<T>& dad) 
{
    /* subtree crossover between this and other, producing new Program */
    // choose location by weighted sampling of program
    // TODO: why doesn't this copy the search space reference to child?
    Program<T> child(mom);

    // pick a subtree to replace
    vector<float> child_weights(child.Tree.size());    
    auto child_iter = child.Tree.begin();
    std::transform(child.Tree.begin(), child.Tree.end(), child_weights.begin(),
                [&](const auto& n){ 
                    auto s_at = child.size_at(child_iter);
                    auto d_at = child.depth_to_reach(child_iter);

                    std::advance(child_iter, 1);

                    if (s_at<PARAMS["max_size"].get<int>() && d_at<PARAMS["max_depth"].get<int>())
                        return n.get_prob_change(); 
                    else
                        return 0.0f;
                }
    );

    if (std::all_of(child_weights.begin(), child_weights.end(), [](const auto& w) {
        return w<=0.0;
    }))
    { // There is no spot that has a probability to be selected
        return std::nullopt;
    }

    auto child_spot = r.select_randomly(child.Tree.begin(), 
                                        child.Tree.end(), 
                                        child_weights.begin(), 
                                        child_weights.end()
                                    );

    auto child_ret_type = child_spot.node->data.ret_type;

    auto allowed_size  = PARAMS["max_size"].get<int>() -
                         ( child.size() - child.size_at(child_spot) );
    auto allowed_depth = PARAMS["max_depth"].get<int>() - 
                         ( child.depth_to_reach(child_spot) );

    Program<T> other(dad);
    // pick a subtree to insert. Selection is based on other_weights
    vector<float> other_weights(other.Tree.size());

    // iterator to get the size of subtrees inside transform
    auto other_iter = other.Tree.begin();

    // lambda function to check feasibility of solution and increment the iterator 
    const auto check_and_incrm = [other, &other_iter, allowed_size, allowed_depth]() -> bool {
        int s = other.size_at( other_iter );
        int d = other.depth_at( other_iter );

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
                return 0.0f;
        }
    );

    bool matching_spots_found = false;
    for (const auto& w: other_weights)
    {
        matching_spots_found = w > 0.0;

        if (matching_spots_found) {
            auto other_spot = r.select_randomly(
                other.Tree.begin(), 
                other.Tree.end(), 
                other_weights.begin(), 
                other_weights.end()
            );
                            
            // fmt::print("other_spot : {}\n",other_spot.node->data);
            // swap subtrees at child_spot and other_spot
            // TODO: do I need to delete the removed node?
            child.Tree.move_ontop(child_spot, other_spot);
            return child;
        }
    }

    return std::nullopt;
};
} //namespace variation
#endif