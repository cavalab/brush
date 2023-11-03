#include "selection.h"

namespace selection {

Selection::Selection()
{
    /*!
     * set type of selection operator.
     */
    this->type = "lexicase";
    this->survival = false;
}

Selection::~Selection(){}



SelectionOperator::~SelectionOperator(){}

vector<size_t> SelectionOperator::select() 
{   
    // THROW_INVALID_ARGUMENT("Undefined select() operation");
    return vector<size_t>();
}

vector<size_t> SelectionOperator::survive()
{
    // THROW_INVALID_ARGUMENT("Undefined select() operation");
    return vector<size_t>();
}

} // selection