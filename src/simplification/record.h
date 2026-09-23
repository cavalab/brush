#ifndef SIMPLIFICATION_RECORD_H
#define SIMPLIFICATION_RECORD_H

#include <string>
#include <vector>

namespace Brush { namespace Simpl{

/// One replacement performed by a simplifier, as written to
/// `<logfile>_simplifications.csv`.
struct SimplificationRecord
{
    unsigned    generation    = 0;
    unsigned    individual_id = 0;
    std::string simplifier;  ///< "constants" or "inexact"
    std::string ret_type;    ///< return type of the replaced subtree
    std::string original;    ///< subtree before the replacement
    std::string replacement; ///< subtree after the replacement
    float       distance = 0.0f; ///< mse between program predictions before/after (inexact only)
};

using SimplificationRecords = std::vector<SimplificationRecord>;

} // Simpl
} // Brush

#endif
