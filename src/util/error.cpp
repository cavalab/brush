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
            throw;
        }
        
        ///prints error to stderr and returns
        void HandleErrorNoThrow(string err, const char *file, int line )
        {
            fmt::print(stderr, "ERROR {}:{}: {}\n", file, line, err);
        }
} }

