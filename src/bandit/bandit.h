/* Brush
copyright 2024 William La Cava
license: GNU/GPL v3
*/

#ifndef BANDIT_H
#define BANDIT_H

#include "../init.h"
#include "../types.h"
#include "../program/program.h"
#include "../vary/search_space.h"
#include "../util/utils.h"
#include "bandit_operator.h"
#include "dummy.h"
#include "thompson.h"

namespace Brush {
namespace MAB {

using namespace Brush;

/**
 * @brief The Bandit struct represents a multi-armed bandit.
 * 
 * The Bandit struct encapsulates the bandit operator (policy) and provides\
 * methods to set and get the arms, type, and probabilities of the bandit.
 * It also provides a method to update the bandit's state based on the chosen
 * arm and the received reward.
 */
struct Bandit
{
    using Iter = tree<Node>::pre_order_iterator;
    
public:
    /**
     * @brief A shared pointer to the bandit operator (policy).
    */
    std::shared_ptr<BanditOperator> pbandit;
    // TODO: This should be a shared pointer to allow multiple instances of Bandit to share the same operator.
     
    std::string type; /**< The type of the bandit. */
    vector<string> arms; /**< The arms of the bandit. */

    std::map<string, float> probabilities; /**< The probabilities associated with each arm. */

    Bandit();
    ~Bandit(){};

    /**
     * @brief Constructor for the Bandit struct.
     * @param type The type of the bandit.
     * @param arms The arms of the bandit.
     */
    Bandit(string type, vector<string> arms);

    /**
     * @brief Constructor for the Bandit struct.
     * @param type The type of the bandit.
     * @param arms_probs The arms and their associated probabilities.
     */
    Bandit(string type, map<string, float> arms_probs);

    /**
     * @brief Sets the bandit operator (policy).
     */
    void set_bandit();

    /**
     * @brief Sets the arms of the bandit.
     */
    void set_arms(vector<string> arms);

    /**
     * @brief Gets the arms of the bandit.
     */
    vector<string> get_arms();

    /**
     * @brief Gets the type of the bandit.
     */
    string get_type();

    /**
     * @brief Sets the type of the bandit.
     */
    void set_type(string type);

    /**
     * @brief Gets the probabilities associated with each arm.
     */
    map<string, float> get_probs();

    /**
     * @brief Sets the probabilities associated with each arm.
     * @param arms_probs The arms and their associated probabilities.
     */
    void set_probs(map<string, float> arms_probs);

    /**
     * @brief Samples the probabilities associated with each arm using the policy.
     * @param update Flag indicating whether to update the bandit's state, or just return the current probabilities.
     * @return The sampled probabilities associated with each arm.
     */
    map<string, float> sample_probs(bool update=false);

    /**
     * @brief Selects an arm.
     * 
     * @return T The selected arm from the tree.
     */
    string choose();

    /**
     * @brief Updates the bandit's state based on the chosen arm and the received reward.
     * @param arm The chosen arm.
     * @param reward The received reward.
     */
    void update(string arm, float reward);
};

//TODO: serialization should save the type of bandit and its parameters

} // MAB
} // Brush
#endif
