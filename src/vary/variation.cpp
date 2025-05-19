#include "variation.h"

namespace Brush {
namespace Var {


using namespace Brush;
using namespace Pop;
using namespace MAB;

/// @brief replace node with same typed node
/// @param prog the program
/// @param Tree the program tree
/// @param spot an iterator to the node that is being mutated
/// @param SS the search space to sample a node like `spot`
/// @return boolean indicating the success (true) or fail (false) of the operation
class PointMutation : public MutationBase
{
public:
    template<Brush::ProgramType T>
    static auto mutate(Program<T>& program, Iter spot, Variation<T>& variator,
                    const Parameters& params)
    {
        // get_node_like will sample a similar node based on node_map_weights or
        // terminal_weights, and maybe will return a Node.

        optional<Node> newNode = variator.bandit_get_node_like(spot.node->data);

        if (!newNode) // overload to check if newNode == nullopt
            return false;

        // if optional contains a Node, we access its contained value
        program.Tree.replace(spot, *newNode);

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
    template<Brush::ProgramType T>
    static auto find_spots(Program<T>& program, Variation<T>& variator,
                    const Parameters& params)
    {
        vector<float> weights;

        if (program.Tree.size() < params.get_max_size()) {
            Iter iter = program.Tree.begin();
            std::transform(program.Tree.begin(), program.Tree.end(), std::back_inserter(weights),
                        [&](const auto& n){ 
                            size_t d = 1+program.Tree.depth(iter);
                            std::advance(iter, 1);

                            // check if SS holds an operator to avoid failing `check` in sample_op_with_arg
                            if ((d >= params.get_max_depth())
                            ||  (variator.search_space.node_map.find(n.ret_type) == variator.search_space.node_map.end())) {
                                return 0.0f;
                            }
                            else {
                                return n.get_prob_change(); 
                            }
                        });
        }
        else {
            // fill the vector with zeros, since we're already at max_size
            weights.resize(program.Tree.size());
            std::fill(weights.begin(), weights.end(), 0.0f); 
        }
        
        return weights;
    }

    template<Brush::ProgramType T>
    static auto mutate(Program<T>& program, Iter spot, Variation<T>& variator,
                    const Parameters& params)
    {
        auto spot_type = spot.node->data.ret_type;
        
        // pick a random compatible node to insert (with probabilities given by
        // node_map_weights). The `-1` represents the node being inserted.
        // Ideally, it should always find at least one match (the same node
        // used as a reference when calling the function). However, we have a 
        // size restriction, which will be relaxed here (just as it is in the PTC2
        // algorithm). This mutation can create a new expression that exceeds the
        // maximum size by the highest arity among the operators.

        std::optional<Node> n = variator.bandit_sample_op_with_arg(
            spot_type, spot_type, params.max_size-program.Tree.size()-1); 

        if (!n) // there is no operator with compatible arguments
            return false;

        // make node n wrap the subtree at the chosen spot
        auto parent_node = program.Tree.wrap(spot, *n);

        // now fill the arguments of n appropriately
        bool spot_filled = false;
        for (auto a: (*n).arg_types)
        {
            if (spot_filled)
            {
                // if spot is in its child position, append children.
                auto opt = variator.bandit_sample_terminal(a);

                if (!opt)
                    return false;

                program.Tree.append_child(parent_node, opt.value());
            }
            // if types match, treat this spot as filled by the spot node 
            else if (a == spot_type)
                spot_filled = true;
            // otherwise, add siblings before spot node
            else {        
                auto opt = variator.bandit_sample_terminal(a);

                if (!opt)
                    return false;

                program.Tree.insert(spot, opt.value());
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
    template<Brush::ProgramType T>
    static auto mutate(Program<T>& program, Iter spot, Variation<T>& variator,
                    const Parameters& params)
    {
        // sample_terminal will sample based on terminal_weights. If it succeeds, 
        // then the new terminal will be in `opt.value()`

        auto opt = variator.bandit_sample_terminal(spot.node->data.ret_type); 
        
        if (!opt) // there is no terminal with compatible arguments
            return false;

        program.Tree.erase_children(spot); 

        program.Tree.replace(spot, opt.value());

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
    template<Brush::ProgramType T>
    static auto find_spots(Program<T>& program, Variation<T>& variator,
                    const Parameters& params)
    {
        vector<float> weights(program.Tree.size());

        if (program.Tree.size() < params.max_size) {
            std::transform(program.Tree.begin(), program.Tree.end(), weights.begin(),
                        [&](const auto& n){
                            // some nodetypes must always have a weight                            
                            if (Is<NodeType::OffsetSum>(n.node_type) || Is<NodeType::Constant>(n.node_type))
                                return 0.0f;

                            // only weighted nodes can be toggled off
                            if (!n.get_is_weighted()
                            &&  IsWeighable(n.node_type))
                            {
                                return n.get_prob_change();
                            }
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

    template<Brush::ProgramType T>
    static auto mutate(Program<T>& program, Iter spot, Variation<T>& variator,
                    const Parameters& params)
    {
        if (spot.node->data.get_is_weighted()==true // cant turn on whats already on
        ||  !IsWeighable(spot.node->data.node_type)) // does not accept weights (e.g. boolean)
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
    template<Brush::ProgramType T>
    static auto find_spots(Program<T>& program, Variation<T>& variator,
                    const Parameters& params)
    {
        vector<float> weights(program.Tree.size());

        std::transform(program.Tree.begin(), program.Tree.end(), weights.begin(),
                    [&](const auto& n){
                        // some nodetypes must always have a weight                            
                        if (Is<NodeType::OffsetSum>(n.node_type) || Is<NodeType::Constant>(n.node_type))
                            return 0.0f;
                            
                        if (n.get_is_weighted()
                        &&  IsWeighable(n.node_type))
                            return n.get_prob_change();
                        else
                            return 0.0f;
                    });

        return weights;
    }

    template<Brush::ProgramType T>
    static auto mutate(Program<T>& program, Iter spot, Variation<T>& variator,
                    const Parameters& params)
    {
        if (spot.node->data.get_is_weighted()==false) // TODO: This condition should never happen. Make sure it dont, then remove it. (this is also true for toggleweighton, also fix that)
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
    template<Brush::ProgramType T>
    static auto find_spots(Program<T>& program, Variation<T>& variator,
                    const Parameters& params)
    {
        vector<float> weights;

        auto node_map = variator.search_space.node_map;

        if (program.Tree.size() < params.max_size) {
            Iter iter = program.Tree.begin();
            std::transform(program.Tree.begin(), program.Tree.end(), std::back_inserter(weights),
                        [&](const auto& n){ 
                            size_t d = program.Tree.depth(iter);
                            std::advance(iter, 1);

                            // we need to make sure there's some node to start the subtree
                            if ((d >= params.max_depth)
                            ||  (node_map.find(n.ret_type) == node_map.end()) )
                                return 0.0f;
                            else
                                return n.get_prob_change(); 
                        });
        }
        else {
            weights.resize(program.Tree.size());
            std::fill(weights.begin(), weights.end(), 0.0f); 
        }
        
        return weights;
    }

    template<Brush::ProgramType T>
    static auto mutate(Program<T>& program, Iter spot, Variation<T>& variator,
                    const Parameters& params)
    {
        // check if we exceeded the size/depth constrains (without subtracting,
        // to avoid overflow cases if the user sets max_size smaller than arity
        // of smallest operator. The overflow would happen when calculating d and
        // s in the following lines, to choose the PTC2 limits)
        if ( params.max_size  <= (program.Tree.size() - program.Tree.size(spot))
        ||   params.max_depth <= program.Tree.depth(spot) )
            return false;

        auto spot_type = spot.node->data.ret_type;

        // d and s must be compatible with PTC2 --- they should be based on 
        // tree structure, not program structure
        size_t d = params.max_depth - program.Tree.depth(spot);
        size_t s = params.max_size - (program.Tree.size() - program.Tree.size(spot));

        s = r.rnd_int(1, s+1);

        // sample subtree uses PTC2, which operates on depth and size of the tree<Node> 
        // (and not on the program!). we shoudn't care for weights here

        auto subtree = variator.search_space.sample_subtree(spot.node->data, d, s); 


        if (!subtree) // there is no terminal with compatible arguments
            return false;


        // if optional contains a Node, we access its contained value
        program.Tree.erase_children(spot); 


        program.Tree.move_ontop(spot, subtree.value().begin());



        return true;
    }
};

/// @brief Inserts an split node in the `spot`
/// @param prog the program
/// @param Tree the program tree
/// @param spot an iterator to the node that is being mutated
/// @param SS the search space to generate a compatible subtree
/// @return boolean indicating the success (true) or fail (false) of the operation
class SplitMutation : public MutationBase
{
public:
    template<Brush::ProgramType T>
    static auto find_spots(Program<T>& program, Variation<T>& variator,
                    const Parameters& params)
    {
        vector<float> weights;

        if (program.Tree.size() < params.get_max_size()) {
            Iter iter = program.Tree.begin();
            std::transform(program.Tree.begin(), program.Tree.end(), std::back_inserter(weights),
                        [&](const auto& n){ 
                            size_t d = 1+program.Tree.depth(iter);
                            std::advance(iter, 1);

                            // check if SS holds an operator to avoid failing `check` in sample_op_with_arg
                            if (d >= params.get_max_depth()
                            ||  variator.search_space.node_map.find(n.ret_type) == variator.search_space.node_map.end()
                            // ||  check if n.ret_type can be splitted (e.g. DataType::ArrayF)
                            ) {
                                return 0.0f;
                            }
                            else {
                                return n.get_prob_change(); 
                            }
                        });
        }
        else {
            // fill the vector with zeros, since we're already at max_size
            weights.resize(program.Tree.size());
            std::fill(weights.begin(), weights.end(), 0.0f); 
        }
        
        return weights;
    }

    template<Brush::ProgramType T>
    static auto mutate(Program<T>& program, Iter spot, Variation<T>& variator,
                    const Parameters& params)
    {
        return false;
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
 * TODO: update this documentation (it doesnt take the program but the individual. also update mutation documentation)
 * This means that, if you use the cross as `auto opt = mutate(parent, SS)`,
 * either `opt==false` or `opt.value()` contains the child.
 * 
 * @tparam T the program type
 * @param root the root parent
 * @param other the donating parent
 * @return `std::optional` that may contain the child program of type `T`
 */
template<Brush::ProgramType T>
std::optional<Individual<T>> Variation<T>::cross(
    const Individual<T>& mom, const Individual<T>& dad) 
{
    /* subtree crossover between this and other, producing new Program */
    // choose location by weighted sampling of program
    // TODO: why doesn't this copy the search space reference to child?
    Program<T> child(mom.program);

    // pick a subtree to replace
    vector<float> child_weights(child.Tree.size());

    auto child_iter = child.Tree.begin();
    std::transform(child.Tree.begin(), child.Tree.end(), child_weights.begin(),
                [&](const auto& n){ 
                    auto s_at = child.size_at(child_iter);
                    auto d_at = child.depth_to_reach(child_iter);

                    std::advance(child_iter, 1);

                    if (s_at<parameters.max_size && d_at<parameters.max_depth)
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
    
    // pick a subtree to insert. Selection is based on other_weights
    Program<T> other(dad.program);

    int attempts = 0;
    while (++attempts <= 3)
    {
        auto child_spot = r.select_randomly(child.Tree.begin(), 
                                            child.Tree.end(), 
                                            child_weights.begin(), 
                                            child_weights.end()
                                        );

        auto child_ret_type = child_spot.node->data.ret_type;

        auto allowed_size  = parameters.max_size -
                            ( child.size() - child.size_at(child_spot) );
        auto allowed_depth = parameters.max_depth - 
                            ( child.depth_to_reach(child_spot) );

        vector<float> other_weights(other.Tree.size());

        // Iterator to traverse the tree during transformation
        auto other_iter = other.Tree.begin();
        std::transform(other.Tree.begin(), other.Tree.end(), other_weights.begin(),
            [&other, &other_iter, allowed_size, allowed_depth, child_ret_type](const auto& n) mutable {
                int s = other.size_at(other_iter);
                int d = other.depth_at(other_iter);

                std::advance(other_iter, 1);

                // Check feasibility and matching return type
                if (s <= allowed_size && d <= allowed_depth && n.ret_type == child_ret_type) {
                    return n.get_prob_change();
                }

                return 0.0f; // Non-feasible crossover point
            }
        );
        
        bool matching_spots_found = std::any_of(other_weights.begin(), other_weights.end(), 
                                        [](float w) { return w > 0.0f; });

        if (matching_spots_found) {

            auto other_spot = r.select_randomly(
                other.Tree.begin(), 
                other.Tree.end(), 
                other_weights.begin(), 
                other_weights.end()
            );
                 
            // fmt::print("other_spot : {}\n",other_spot.node->data);
            // swap subtrees at child_spot and other_spot
            child.Tree.move_ontop(child_spot, other_spot);
            
            Individual<T> ind(child);
            ind.set_variation("cx"); // TODO: use enum here to make it faster

            return ind;
        }
    }

    return std::nullopt;
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
template<Brush::ProgramType T>
std::optional<Individual<T>> Variation<T>::mutate(
    const Individual<T>& parent, string choice)
{
    if (choice.empty())
    {

        auto options = parameters.mutation_probs;

        bool all_zero = true;
        for (auto &it : parameters.mutation_probs) {
            if (it.second > 0.0) {
                all_zero = false;
                break;
            }
        }

        if (all_zero) { // No mutation can be successfully applied to this solution  
            return std::nullopt;
        }
        
        // picking a valid mutation option
        choice = r.random_choice(parameters.mutation_probs);
    }
    
    Program<T> copy(parent.program);

    vector<float> weights; // choose location by weighted sampling of program
    if (choice.compare("point") == 0) // TODO: use enum here to optimize
        weights = PointMutation::find_spots(copy, (*this), parameters);
    else if (choice.compare("insert") == 0)
        weights = InsertMutation::find_spots(copy, (*this), parameters);
    else if (choice.compare("delete") == 0)
        weights = DeleteMutation::find_spots(copy, (*this), parameters);
    else if (choice.compare("subtree") == 0)
        weights = SubtreeMutation::find_spots(copy, (*this), parameters);
    else if (choice.compare("toggle_weight_on") == 0)
        weights = ToggleWeightOnMutation::find_spots(copy, (*this), parameters);
    else if (choice.compare("toggle_weight_off") == 0)
        weights = ToggleWeightOffMutation::find_spots(copy, (*this), parameters);
    else {
        std::string msg = fmt::format("{} not a valid mutation choice", choice);
        HANDLE_ERROR_THROW(msg);
    }

    if (std::all_of(weights.begin(), weights.end(), [](const auto& w) {
        return w<=0.0;
    }))
    { // There is no spot that has a probability to be selected
        return std::nullopt;
    }
    int attempts = 0;
    while(attempts++ < 3)
    {
        Program<T> child(parent.program);

        // apply the mutation and check if it succeeded
        auto spot = r.select_randomly(child.Tree.begin(), child.Tree.end(),
                                      weights.begin(), weights.end());

        // Every mutation here works inplace, so they return bool instead of
        // std::optional to indicare the result of their manipulation over the
        // program tree. Here we call the mutation function and return the result
        
        bool success;
        if (choice.compare("point") == 0)
            success = PointMutation::mutate(child, spot, (*this), parameters);
        else if (choice.compare("insert") == 0)
            success = InsertMutation::mutate(child, spot, (*this), parameters);
        else if (choice.compare("delete") == 0)
            success = DeleteMutation::mutate(child, spot, (*this), parameters);
        else if (choice.compare("subtree") == 0)
            success = SubtreeMutation::mutate(child, spot, (*this), parameters);
        else if (choice.compare("toggle_weight_on") == 0)
            success = ToggleWeightOnMutation::mutate(child, spot, (*this), parameters);
        else // it must be"toggle_weight_off"
            success = ToggleWeightOffMutation::mutate(child, spot, (*this), parameters);

        if (success
        && ( (child.size()  <= parameters.max_size)
        &&   (child.depth() <= parameters.max_depth) )){

            Individual<T> ind(child);

            ind.set_variation(choice);

            // subtree performs several samplings, and it will leverate
            // what point/insert/delete mutations learned about each node utility.

            // TODO: handle subtree - it will sample too many nodes and it may
            // be hard to track which ones actually improved the expression to
            // update the bandits/ maybe we should skip it?
            // mutations that sampled from search space
            if (choice.compare("point")   == 0
            ||  choice.compare("insert")  == 0
            ||  choice.compare("delete")  == 0
            // ||  choice.compare("subtree") == 0 // TODO: disable this one
            ) {
                ind.set_sampled_nodes({spot.node->data});
            }
            
            return ind;
        }
        else { // reseting 

        }
    }

    return std::nullopt;
};

template<Brush::ProgramType T>
void Variation<T>::vary(Population<T>& pop, int island, 
                        const vector<size_t>& parents)
{    
    auto indices = pop.get_island_indexes(island);

    for (unsigned i = 0; i<indices.size(); ++i)
    {
        if (pop.individuals.at(indices.at(i)) != nullptr)
        {
            continue; // skipping if it is an individual
        }
        
        // pass check for children undergoing variation     
        std::optional<Individual<T>> opt=std::nullopt; // new individual  
            
        const Individual<T>& mom = pop[
            *r.select_randomly(parents.begin(), parents.end())];
    
        vector<Individual<T>> ind_parents;
        
        bool crossover = ( r() < parameters.cx_prob );
        if (crossover)
        {
            const Individual<T>& dad = pop[
                *r.select_randomly(parents.begin(), parents.end())];
            
            auto variation_result = cross(mom, dad);
            ind_parents = {mom, dad};
            opt = variation_result;
        }
        else
        {

            auto variation_result = mutate(mom);   

            ind_parents = {mom};
            opt = variation_result;

        }
    
        // this assumes that islands do not share indexes before doing variation
        unsigned id = parameters.current_gen*parameters.pop_size+indices.at(i);

        // mutation and crossover already perform 3 attempts. If it fails, we just fill with a random individual
        
        Individual<T> ind;
        if (opt) // variation worked, lets keep this
        {
            ind = opt.value();
            ind.set_parents(ind_parents);
        }
        else {  // no optional value was returned. creating a new random individual
            // It seems that the line below will not fix the root in clf programs
            ind.init(search_space, parameters); // ind.variation is born by default
        
            // Program<T> p = search_space.make_program<Program<T>>(parameters, 0, 0);
            // ind = Individual<T>(p);
        }

        ind.set_objectives(mom.get_objectives()); // it will have an invalid fitness

        ind.is_fitted_ = false;
        ind.set_id(id);

        // TODO: smarter way of copying the entire fitness
        // copying mom fitness to the new individual (without making the fitnes valid)
        // so the bandits can access this information. Fitness will be valid
        // only when we do set_values(). We are setting these parameters below
        // because we want to keep the previous values for the bandits, and
        // we are not updating the fitness values here.
        ind.fitness.set_loss(mom.fitness.get_loss());
        ind.fitness.set_loss_v(mom.fitness.get_loss_v());
        ind.fitness.set_size(mom.fitness.get_size());
        ind.fitness.set_complexity(mom.fitness.get_complexity());
        ind.fitness.set_linear_complexity(mom.fitness.get_linear_complexity());
        ind.fitness.set_depth(mom.fitness.get_depth());

        // dont set stuff that is not used to calculate the rewards, like crowding_dist
        // ind.fitness.set_crowding_dist(0.0);
        
        assert(ind.program.size()>0);
        assert(ind.fitness.valid()==false);

        pop.individuals.at(indices.at(i)) = std::make_shared<Individual<T>>(ind);
    }
};

template <Brush::ProgramType T>
void Variation<T>::update_ss()
{
    // propagate bandits learnt information to the search space.
    // TODO: not all arms are initialized, if the user set something to zero then we must
    // disable it. So, during update, we need to properly handle these skipped arms. --> remove this for nodes, allow it just for variations. If the user doesnt want to use a feature or op, he should not set it at the first place. We need to do this with variations because the user 
    // can choose it directly instead of letting brush to figure out.

    // variation: getting new probabilities for variation operators
    auto variation_probs = variation_bandit.sample_probs(true);

    if (variation_probs.find("cx") != variation_probs.end())
        parameters.set_cx_prob(variation_probs.at("cx"));
    
    for (const auto& variation : variation_probs)
        if (variation.first != "cx")
            parameters.mutation_probs[variation.first] = variation.second;
            
    // terminal: getting new probabilities for terminal nodes in search space
    for (auto& bandit : terminal_bandits) {
        auto datatype = bandit.first;
        
        auto terminal_probs = bandit.second.sample_probs(true);

        for (auto& [terminal_name, terminal_prob] : terminal_probs) {
            // Search for the index that matches the terminal name
            auto it = std::find_if(
                search_space.terminal_map.at(datatype).begin(),
                search_space.terminal_map.at(datatype).end(), 
                [&](auto& node) { return node.get_feature() == terminal_name; });

            // if (it != search_space.terminal_map.at(datatype).end()) {
                auto index = std::distance(search_space.terminal_map.at(datatype).begin(), it);

                // Update the terminal weights with the second value
                search_space.terminal_weights.at(datatype)[index] = terminal_prob;
            // }
        }
    }

    // operators: getting new probabilities for op nodes
    for (auto& [ret_type, bandit_map] : op_bandits) {
        for (auto& [args_type, bandit] : bandit_map) {
            auto op_probs = bandit.sample_probs(true);

            for (auto& [op_name, op_prob] : op_probs) {

                for (const auto& [node_type, node_value]: search_space.node_map.at(ret_type).at(args_type))
                {
                    if (node_value.name == op_name) {

                        search_space.node_map_weights.at(ret_type).at(args_type).at(node_type) = op_prob;
                    }
                }
            }
        }
    }
};

} //namespace Var
} //namespace Brush
