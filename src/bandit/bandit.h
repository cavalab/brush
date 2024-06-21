/* Brush
copyright 2024 William La Cava
license: GNU/GPL v3
*/

#ifndef BANDIT_H
#define BANDIT_H

#include "bandit_operator.h"
#include "dummy.h"

namespace Brush {
namespace MAB {

using namespace Brush;

template <typename T>
struct Bandit
{
public:
    /**
     * @brief A shared pointer to the bandit operator (policy).
     * 
     * TODO: This should be a shared pointer to allow multiple instances of Bandit to share the same operator.
     */
    std::shared_ptr<BanditOperator<T>> pbandit;

    /**
     * @brief The type (policy) of the bandit.
     */
    std::string type;

    /**
     * @brief The number of arms in the bandit.
     */
    int arms;

    Bandit();
    ~Bandit(){};

    /**
     * @brief Constructor for Bandit.
     * 
     * @param type The type (policy) of the bandit.
     * @param arms The number of arms in the bandit.
     */
    Bandit(string type, int arms);


    /**
     * @brief Set the bandit operator for the bandit.
     * 
     * This function sets the bandit operator (policy) for the bandit.
     */
    void set_bandit();

    /**
     * @brief Get the type of the bandit.
     * 
     * @return The type (policy) of the bandit.
     */
    string get_type();

    /**
     * @brief Set the type of the bandit.
     * 
     * @param type The type (policy) of the bandit.
     */
    void set_type(string type);

    /**
     * @brief Sample a probability distribution for each arm.
     * 
     * If update is true, the inner tracker of probabilities is updated.
     * 
     * @param update Flag indicating whether to update the inner tracker of probabilities.
     * @return The sampled probability distribution.
     */
    map<T, float> sample_probs(bool update);

    /**
     * @brief Update the inner state of the bandit based on the rewards obtained from the arms.
     * 
     * @param rewards The rewards obtained from the arms.
     */
    void update_with_reward(vector<float> rewards);
};

//TODO: serialization should save the type of bandit and its parameters

} // MAB
} // Brush
#endif