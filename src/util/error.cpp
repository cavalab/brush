/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/

#include "error.h"
#include "fmt/core.h"
//#include "node/node.h"
//external includes

namespace Brush{ namespace Util{
        /// prints error and throws an exception
        void HandleErrorThrow(string err, const char *file, int line )
        {
            fmt::print(stderr, "FATAL ERROR {}:{}: {}\n", file, line, err);

            // when called with no arguments, will call terminate(), which
            // throws a std::terminate_handler (and can't be handled in GTEST).
            // Here we throw a runtime_error with same information printed on
            // screen.
            throw std::runtime_error(fmt::format("FATAL ERROR {}:{}: {}\n", file, line, err)); 
        }
        
        ///prints error to stderr and returns
        void HandleErrorNoThrow(string err, const char *file, int line )
        {
            fmt::print(stderr, "WARNING {}:{}: {}\n", file, line, err);
        }
} }

