#ifndef ARCHIVE_H
#define ARCHIVE_H

//#include "node.h" // including node.h since definition of node is in the header
#include "../ind/individual.h"

///< nsga2 selection operator for getting the front
#include "../selection/nsga2.h"

namespace Brush{

using namespace Sel;

namespace Pop{
    
template<ProgramType T> 
struct Archive  
{
    // I dont need shared pointers here
    vector<Individual<T>> individuals; ///< individual programs in the archive
    bool sort_complexity;    ///< whether to sort archive by complexity

    // using NSGA2 in survival mode (nsga2 does not implement selection)
    NSGA2<T> selector; 

    Archive();

    void init(Population<T>& pop);

    void update(const Population<T>& pop, const Parameters& params);
   
    void set_objectives(vector<string> objectives);

    /// Sort population in increasing complexity.
    static bool sortComplexity(const Individual<T>& lhs, 
            const Individual<T>& rhs);

    /// Sort population by first objective.
    static bool sortObj(const Individual<T>& lhs, 
            const Individual<T>& rhs, const int index=0);

    /// check for repeats
    static bool sameFitComplexity(const Individual<T>& lhs, 
            const Individual<T>& rhs);
    static bool sameObjectives(const Individual<T>& lhs, 
            const Individual<T>& rhs);
};

//serialization
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Archive<PT::Regressor>, individuals);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Archive<PT::BinaryClassifier>, individuals);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Archive<PT::MulticlassClassifier>, individuals);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Archive<PT::Representer>, individuals);

} // Pop
} // Brush

#endif
