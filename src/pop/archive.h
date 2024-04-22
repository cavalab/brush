#ifndef ARCHIVE_H
#define ARCHIVE_H

//#include "node.h" // including node.h since definition of node is in the header
#include "../individual.h"

///< nsga2 selection operator for getting the front
#include "../selection/nsga2.h"

// TODO: do i really need these?
using std::vector;
using std::string;
using Eigen::Map;

namespace Brush{

using namespace Sel;

namespace Pop{
    
template<ProgramType T> 
struct Archive  
{
    // I dont need shared pointers here
    vector<Individual<T>> individuals; ///< individual programs in the archive
    bool sort_complexity;    ///< whether to sort archive by complexity

    NSGA2<T> selector; 

    Archive(){};
    ~Archive(){};

    void init(Population<T>& pop){};

    void update(const Population<T>& pop, const Parameters& params){};
   
    void set_objectives(vector<string> objectives){};

    /// Sort population in increasing complexity.
    static bool sortComplexity(const Individual<T>& lhs, 
            const Individual<T>& rhs){ return false; };

    /// Sort population by first objective.
    static bool sortObj1(const Individual<T>& lhs, 
            const Individual<T>& rhs){ return false; };

    /// check for repeats
    static bool sameFitComplexity(const Individual<T>& lhs, 
            const Individual<T>& rhs){ return false; };
    static bool sameObjectives(const Individual<T>& lhs, 
            const Individual<T>& rhs){ return false; };
};

//serialization
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Archive<PT::Regressor>, individuals);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Archive<PT::BinaryClassifier>, individuals);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Archive<PT::MulticlassClassifier>, individuals);
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(Archive<PT::Representer>, individuals);

} // Pop
} // Brush

#endif
