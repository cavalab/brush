#ifndef LEXICASE_H
#define LEXICASE_H

#include "selection_operator.h"
#include "../util/utils.h"

namespace Brush {
namespace Sel {

using namespace Brush;
using namespace Pop;
using namespace Sel;


/*!
* @class Lexicase
* @brief Lexicase selection operator.
*/

template<ProgramType T> 
class Lexicase : public SelectionOperator<T>
{
public:
    Lexicase(bool surv=false);
    ~Lexicase(){};

    /// function returns a set of selected indices from pop 
    vector<size_t> select(Population<T>& pop, int island,
            const Parameters& p);
    
    /// lexicase survival
    vector<size_t> survive(Population<T>& pop, int island, 
            const Parameters& p);

    void set_lexicase_pool(vector<size_t> s) { this->lexicase_pool = s; }

private:
        vector<size_t> lexicase_pool;
};

extern template class Lexicase<PT::Regressor>;
extern template class Lexicase<PT::BinaryClassifier>;
extern template class Lexicase<PT::MulticlassClassifier>;
extern template class Lexicase<PT::Representer>;

} // Sel
} // Brush
#endif