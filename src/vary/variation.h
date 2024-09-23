/* Brush

copyright 2020 William La Cava
license: GNU/GPL v3
*/
#ifndef VARIATION_H
#define VARIATION_H

#include "../ind/individual.h"

#include "../bandit/bandit.h"
#include "../bandit/bandit_operator.h"
#include "../bandit/dummy.h"

#include "../pop/population.h"
#include "../eval/evaluation.h"

#include <map>
#include <optional>

using namespace Brush::Pop;
using namespace Brush::MAB;
using namespace Brush::Eval;

/**
 * @brief Namespace for variation functions like crossover and mutation. 
 * 
 */
namespace Brush {
namespace Var {

class MutationBase {
public:
    using Iter = tree<Node>::pre_order_iterator;

    static auto find_spots(tree<Node>& Tree, const SearchSpace& SS,
                            const Parameters& params)
    {
        vector<float> weights(Tree.size());

        // by default, mutation can happen anywhere, based on node weights
        std::transform(Tree.begin(), Tree.end(), weights.begin(),
                       [&](const auto& n){ return n.get_prob_change();});
        
        // Should have same size as prog.Tree.size, even if all weights <= 0.0
        return weights;
    }

    static auto mutate(tree<Node>& Tree, Iter spot, const SearchSpace& SS,
                            const Parameters& params);
};

/*!
 * @class Variation
 * @brief Class representing the variation operators in Brush.
 * 
 * The Variation class is responsible for performing individual-level variations
 * and handling the variation of a population in Brush. It contains methods for
 * crossing individuals, mutating individuals, and varying a population.
 */
template<ProgramType T>
class Variation {
public:
    /**
     * @brief Default constructor.
     */
    Variation() = default;
    
    /**
     * @brief Constructor that initializes the Variation object with parameters and search space.
     * 
     * @param params The parameters for the variation operator.
     * @param ss The search space for the variation operator.
     */
    Variation(Parameters& params, SearchSpace& ss)
        : parameters(params)
        , search_space(ss)
    {
        init();
    };

    /**
     * @brief Destructor.
     */
    ~Variation() {};

    /**
     * @brief Initializes the Variation object with parameters and search space.
     * 
     * @param params The parameters for the variation operator.
     * @param ss The search space for the variation operator.
     */
    void init(){
        // initializing variation bandit with the probabilities of non-null variation
        map<string, float> variation_probs;
        for (const auto& mutation : parameters.get_mutation_probs())
            if (mutation.second > 0.0)
                variation_probs[mutation.first] = mutation.second;

        if (parameters.cx_prob > 0.0)
            variation_probs["cx"] = parameters.cx_prob;
        
        this->variation_bandit = Bandit<string>(parameters.bandit, variation_probs);

        // TODO: should I set C parameter based on pop size or leave it fixed?
        // TODO: update string comparisons to use .compare method
        // if (parameters.bandit.compare("dynamic_thompson")==0)
        //     this->variation_bandit.pbandit.set_C(parameters.pop_size);

        // initializing one bandit for each terminal type
        for (const auto& entry : this->search_space.terminal_weights) {
            // entry is a tuple <dataType, vector<float>> where the vector is the weights
            
            if (terminal_bandits.find(entry.first) == terminal_bandits.end())
            {
                map<string, float> terminal_probs;
                for (int i = 0; i < entry.second.size(); i++)
                    if (entry.second[i] > 0.0)
                    {
                        auto node_name = search_space.terminal_map.at(entry.first).at(i).get_feature();
                        terminal_probs[node_name] = entry.second[i];
                    }
                        
                terminal_bandits[entry.first] = Bandit<string>(parameters.bandit,
                                                               terminal_probs);
            }
        }

        // one bandit for each return type. If we look at implementation of 
        // sample op, the thing that matters is the most nested probabilities, so we will
        // learn only that
        for (auto& [ret_type, arg_w_map]: search_space.node_map) 
        {

            // TODO: this could be made much easier using user_ops
            map<string, float> node_probs;
            for (const auto& [args_type, node_map] : arg_w_map)
            {
                for (const auto& [node_type, node]: node_map)
                {
                    auto weight = search_space.node_map_weights.at(ret_type).at(args_type).at(node_type);
                    
                    // Attempt to emplace; if the key exists, do nothing
                    auto [it, inserted] = node_probs.try_emplace(node.name, weight);
                    
                    // If the key already existed, update its value
                    if (!inserted) {
                        // it->second += weight;
                    }
                }
            }
            op_bandits[ret_type] = Bandit<string>(parameters.bandit, node_probs);
        }
    };

    /**
     * @brief Performs crossover operation on two individuals.
     * 
     * @param mom The first parent individual.
     * @param dad The second parent individual.
     * @return An optional containing the offspring individual if the crossover 
     * is successful, or an empty optional otherwise.
     */
    std::tuple<std::optional<Individual<T>>, VectorXf> cross(
        const Individual<T>& mom, const Individual<T>& dad);

    /**
     * @brief Performs mutation operation on an individual.
     * 
     * @param parent The parent individual.
     * @return An optional containing the mutated individual if the mutation is
     * successful, or an empty optional otherwise.
     */
    std::tuple<std::optional<Individual<T>>, VectorXf> mutate(
        const Individual<T>& parent, string choice="");

    /**
     * @brief Handles variation of a population.
     * 
     * @param pop The population to be varied.
     * @param island The island index.
     * @param parents The indices of the parent individuals.
     * @param p The parameters for the variation operator.
     */
    void vary(Population<T>& pop, int island, const vector<size_t>& parents);

    /**
     * Calculates the rewards for the given population and island.
     *
     * @param pop The population to calculate rewards for.
     * @param island The island index.
     * @return A vector of rewards for the population.
     */
    vector<float> calculate_rewards(Population<T>& pop, int island);

    /**
     * Updates the probability distribution sampling for variation and nodes based on the given rewards.
     *
     * @param pop The population to update the selection strategy for.
     * @param rewards The rewards obtained from the evaluation of individuals.
     */
    void update_ss(Population<T>& pop);

    /**
     * @brief Varies a population and updates the selection strategy based on rewards.
     * 
     * This function performs variation on a population, calculates rewards, and updates
     * the selection strategy based on the obtained rewards.
     * 
     * @param pop The population to be varied and updated.
     * @param island The island index.
     * @param parents The indices of the parent individuals.
     */
    void vary_and_update(Population<T>& pop, int island, const vector<size_t>& parents,
                         const Dataset& data, Evaluation<T>& evaluator) {

        // TODO: rewrite this entire function to avoid repetition (this is a frankenstein)
        auto indices = pop.get_island_indexes(island);

        for (unsigned i = 0; i < indices.size(); ++i)
        {
            if (pop.individuals.at(indices.at(i)) != nullptr)
            {
                // std::cout << "Skipping individual at index " << indices.at(i) << std::endl;
                continue; // skipping if it is an individual
            }
            
            // std::cout << "Processing individual at index " << indices.at(i) << std::endl;

            // pass check for children undergoing variation     
            std::optional<Individual<T>> opt = std::nullopt; // new individual  
                
            const Individual<T>& mom = pop[
                *r.select_randomly(parents.begin(), parents.end())];
        
            vector<Individual<T>> ind_parents;
            VectorXf context = this->variation_bandit.get_context(mom.program.Tree, mom.program.Tree.begin());

            string choice = this->variation_bandit.choose(context);

            if (choice == "cx")
            {
                const Individual<T>& dad = pop[
                    *r.select_randomly(parents.begin(), parents.end())];
                
                // std::cout << "Performing crossover" << std::endl;
                auto variation_result = cross(mom, dad);
                ind_parents = {mom, dad};
                tie(opt, context) = variation_result;
            }
            else
            {
                // std::cout << "Performing mutation " << choice << std::endl;
                auto variation_result = mutate(mom, choice);  
                // cout << "finished mutation" << endl;
                ind_parents = {mom};
                tie(opt, context) = variation_result;
                // cout << "unpacked" << endl;
            }

            // this assumes that islands do not share indexes before doing variation
            unsigned id = parameters.current_gen * parameters.pop_size + indices.at(i);

            Individual<T> ind;
            if (opt) // variation worked, lets keep this
            {
                // std::cout << "Variation successful" << std::endl;
                ind = opt.value();
                ind.set_parents(ind_parents);
            }
            else {  // no optional value was returned. creating a new random individual
                // std::cout << "Variation failed, creating a new random individual" << std::endl;
                ind.init(search_space, parameters); // ind.variation is born by default
            }

            ind.set_objectives(mom.get_objectives()); // it will have an invalid fitness

            ind.is_fitted_ = false;
            ind.set_id(id);

            ind.fitness.set_loss(mom.fitness.get_loss());
            ind.fitness.set_loss_v(mom.fitness.get_loss_v());
            ind.fitness.set_size(mom.fitness.get_size());
            ind.fitness.set_complexity(mom.fitness.get_complexity());
            ind.fitness.set_linear_complexity(mom.fitness.get_linear_complexity());
            ind.fitness.set_depth(mom.fitness.get_depth());

            assert(ind.program.size() > 0);
            assert(ind.fitness.valid() == false);

            ind.program.fit(data);
            evaluator.assign_fit(ind, data, parameters, false);

            vector<float> deltas(ind.get_objectives().size(), 0.0f);
            
            float delta = 0.0f;
            float weight = 0.0f;

            for (const auto& obj : ind.get_objectives())
            {   
                if (obj.compare(parameters.scorer) == 0)
                    delta = ind.fitness.get_loss_v() - ind.fitness.get_prev_loss();
                else if (obj.compare("complexity") == 0)
                    delta = ind.fitness.get_complexity() - ind.fitness.get_prev_complexity();
                else if (obj.compare("linear_complexity") == 0)
                    delta = ind.fitness.get_linear_complexity() - ind.fitness.get_prev_linear_complexity();
                else if (obj.compare("size") == 0)
                    delta = ind.fitness.get_size() - ind.fitness.get_prev_size();
                else if (obj.compare("depth") == 0)
                    delta = ind.fitness.get_depth() - ind.fitness.get_prev_depth();
                else
                    HANDLE_ERROR_THROW(obj + " is not a known objective");

                auto it = Individual<T>::weightsMap.find(obj);
                weight = it->second;
                
                deltas.push_back(delta * weight);
            }

            bool allPositive = true;
            for (float d : deltas) {
                if (d < 0) {
                    allPositive = false;
                    break;
                }
            }

            float r = 0.0;
            if (allPositive)
                r = 1.0;

            // std::cout << "Updating variation bandit with reward: " << r << std::endl;

            if (ind.get_variation() != "born")
            {
                this->variation_bandit.update(ind.get_variation(), r, context);
            }
            else
            { // giving zero reward if the variation failed
                this->variation_bandit.update(choice, 0.0, context);
            }

            if (ind.get_variation() != "born" && ind.get_variation() != "cx")
            {                
                if (ind.get_sampled_nodes().size() > 0) {
                    const auto& changed_nodes = ind.get_sampled_nodes();
                    for (const auto& node : changed_nodes) {
                        if (node.get_arg_count() == 0) {
                            auto datatype = node.get_ret_type();
                            // std::cout << "Updating terminal bandit for node: " << node.get_feature() << std::endl;
                            this->terminal_bandits[datatype].update(node.get_feature(), r, context);
                        }
                        else {
                            auto ret_type = node.get_ret_type();
                            auto name = node.name;
                            // std::cout << "Updating operator bandit for node: " << name << std::endl;
                            this->op_bandits[ret_type].update(name, r, context);
                        }
                    }
                }
            }

            pop.individuals.at(indices.at(i)) = std::make_shared<Individual<T>>(ind);
            // std::cout << "Individual at index " << indices.at(i) << " updated successfully" << std::endl;
        }
    }
    
    // they need to be references because we are going to modify them
    SearchSpace search_space; // The search space for the variation operator.
    Parameters parameters;    // The parameters for the variation operator
private:
    // bandits will internaly work as an interface between variation and its searchspace.
    // they will sample from the SS (instead of letting the search space do it directly),
    // and also propagate what they learn back to the search space at the end of the execution.
    Bandit<string> variation_bandit;
    map<DataType, Bandit<string>> terminal_bandits; 
    map<DataType, Bandit<string>> op_bandits;    

    // these functions below will extract context and use it to choose the nodes to replace
    
    // bandit_get_node_like
    //bandit_sample_op_with_arg
    //bandit_sample_terminal
    // bandit_sample_op
    //bandit_sample_subtree

    //etc.
};

// // Explicitly instantiate the template for brush program types
// template class Variation<ProgramType::Regressor>;
// template class Variation<ProgramType::BinaryClassifier>;
// template class Variation<ProgramType::MulticlassClassifier>;
// template class Variation<ProgramType::Representer>;

} //namespace Var
} //namespace Brush
#endif