#ifndef CONSTANTS_H
#define CONSTANTS_H

#include "../init.h"
#include "../types.h"
#include "../program/program.h"
#include "../vary/search_space.h"
#include "../util/utils.h"

using namespace std;
using Brush::Node;
using Brush::DataType;

namespace Brush { namespace Simpl{
    
    class Constants_simplifier
    {
        public:
            static Constants_simplifier* initSimplifier();
            
            static void destroy();

            template<Brush::ProgramType PT>
            static Program<PT> simplify_tree(Program<PT>& program,
                                      const SearchSpace &ss, const Dataset &d);

        private:
            Constants_simplifier();
            ~Constants_simplifier();

            // private static attribute used by every instance of the class
            static Constants_simplifier* instance;
    };

    // static attribute holding an singleton instance of Constants_simplifier.
    // the instance is created by calling `initRand`, which creates
    // an instance of the private static attribute `instance`. `r` will contain
    // one generator for each thread (since it called the constructor) 
    static Constants_simplifier &constants_simplifier = *Constants_simplifier::initSimplifier();

} // Simply
} // Brush

#endif
