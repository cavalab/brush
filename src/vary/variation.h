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

#include <map>
#include <optional>

using namespace Brush::Pop;
using namespace Brush::MAB;

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

        // TODO: op bandit?
        // this->op_bandit = Bandit<DataType>(this->parameters.bandit,
        //                    this->search_space.node_map_weights.size() );
                 
    };

    /**
     * @brief Performs crossover operation on two individuals.
     * 
     * @param mom The first parent individual.
     * @param dad The second parent individual.
     * @return An optional containing the offspring individual if the crossover 
     * is successful, or an empty optional otherwise.
     */
    std::optional<Individual<T>> cross(const Individual<T>& mom,
                                       const Individual<T>& dad);

    /**
     * @brief Performs mutation operation on an individual.
     * 
     * @param parent The parent individual.
     * @return An optional containing the mutated individual if the mutation is
     * successful, or an empty optional otherwise.
     */
    std::optional<Individual<T>> mutate(const Individual<T>& parent);

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
    void update_ss(Population<T>& pop, const vector<float>& rewards);
    
    // they need to be references because we are going to modify them
    SearchSpace search_space; // The search space for the variation operator.
    Parameters parameters;    // The parameters for the variation operator
private:
    Bandit<string> variation_bandit;

    map<DataType, Bandit<string>> terminal_bandits; 

    // TODO: implement bandit for operators
    // Bandit<DataType> op_bandit;
    
};

// // Explicitly instantiate the template for brush program types
// template class Variation<ProgramType::Regressor>;
// template class Variation<ProgramType::BinaryClassifier>;
// template class Variation<ProgramType::MulticlassClassifier>;
// template class Variation<ProgramType::Representer>;

} //namespace Var
} //namespace Brush
#endif