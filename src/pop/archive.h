#ifndef ARCHIVE_H
#define ARCHIVE_H

#include "../ind/individual.h"

///< nsga2 selection operator for getting the front
#include "../selection/nsga2.h"

namespace Brush{

using namespace Sel;

namespace Pop{
    
/**
 * @brief The Archive struct represents a collection of individual programs.
 * 
 * The Archive struct is used to store individual programs in a collection. It provides
 * functionality for initializing, updating, and sorting the archive based on complexity
 * or objectives. The archive can be operated on by a single thread.
 * 
 * @tparam T The program type.
 */
template<ProgramType T> 
struct Archive  
{
    vector<Individual<T>> individuals; ///< individual programs in the archive
    bool sort_complexity;    ///< whether to sort archive by complexity
    bool linear_complexity;  ///< Indicates if the user set linear_complexity instead of recursive complexity
    NSGA2<T> selector;       ///< using NSGA2 in survival mode (nsga2 does not implement selection)

    /**
     * @brief Default constructor for the Archive struct.
     */
    Archive();

    /**
     * @brief Initializes the archive with individuals from a population.
     * @param pop The population from which to initialize the archive.
     */
    void init(Population<T>& pop);

    /**
     * @brief Updates the archive with individuals from a population.
     * @param pop The population from which to update the archive.
     * @param params The parameters for the update.
     */
    void update(Population<T>& pop, const Parameters& params);

    /**
     * @brief Sets the objectives for the archive.
     * 
     * This function sets the objectives for the archive. The objectives are used for
     * sorting the archive.
     * 
     * @param objectives The objectives to set for the archive.
     */
    void set_objectives(vector<string> objectives);

    /**
     * @brief Sorts the population in increasing complexity.
     * 
     * This static function is used to sort the population in increasing complexity.
     * It is used as a comparison function for sorting algorithms.
     * 
     * @param lhs The left-hand side individual to compare.
     * @param rhs The right-hand side individual to compare.
     */
    static bool sortComplexity(const Individual<T>& lhs, const Individual<T>& rhs);

    static bool sortLinearComplexity(const Individual<T>& lhs, const Individual<T>& rhs);

    /**
     * @brief Sorts the population by the first objective.
     * 
     * This static function is used to sort the population by the first objective.
     * It is used as a comparison function for sorting algorithms.
     * 
     * @param lhs The left-hand side individual to compare.
     * @param rhs The right-hand side individual to compare.
     */
    static bool sortObj1(const Individual<T>& lhs, const Individual<T>& rhs);

    /**
     * @brief Checks if two individuals have the same fitness complexity.
     * 
     * This static function is used to check if two individuals have the same fitness complexity.
     * It is used as a comparison function for finding duplicates in the population.
     * 
     * @param lhs The left-hand side individual to compare.
     * @param rhs The right-hand side individual to compare.
     */
    static bool sameFitComplexity(const Individual<T>& lhs, const Individual<T>& rhs);

    /**
     * @brief Checks if two individuals have the same objectives.
     * 
     * This static function is used to check if two individuals have the same objectives.
     * It is used as a comparison function for finding duplicates in the population.
     * 
     * @param lhs The left-hand side individual to compare.
     * @param rhs The right-hand side individual to compare.
     */
    static bool sameObjectives(const Individual<T>& lhs, const Individual<T>& rhs);
};

//serialization
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Archive<PT::Regressor>, individuals, sort_complexity, linear_complexity);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Archive<PT::BinaryClassifier>, individuals, sort_complexity, linear_complexity);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Archive<PT::MulticlassClassifier>, individuals, sort_complexity, linear_complexity);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Archive<PT::Representer>, individuals, sort_complexity, linear_complexity);

} // Pop
} // Brush

#endif
