/* Brush

copyright 2020 William La Cava
license: GNU/GPL v3
*/
#ifndef VARIATION_H
#define VARIATION_H

#include "../pop/population.h"

#include <map>
#include <optional>

using namespace Brush::Pop;

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
    {};

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
    void init(Parameters& params, SearchSpace& ss){
        this->parameters = params;
        this->search_space = ss;
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

private:
    SearchSpace search_space; // The search space for the variation operator.
    Parameters parameters;    // The parameters for the variation operator
};

} //namespace Var
} //namespace Brush
#endif