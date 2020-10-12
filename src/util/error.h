/* Brush
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#ifndef ERROR_H
#define ERROR_H

#include <string>
#include <iostream>

using namespace std;

//#include "node/node.h"
//external includes

namespace BR{ namespace Util {
    /// prints error and throws an exception
    void HandleErrorThrow(string err, const char *file, int line );
    
    ///prints error to stderr and returns
    void HandleErrorNoThrow(string err, const char *file, int line );
    
    #define HANDLE_ERROR_THROW( err ) (BR::Util::HandleErrorThrow( err, __FILE__, __LINE__ ))
    #define HANDLE_ERROR_NO_THROW( err ) (BR::Util::HandleErrorNoThrow( err, __FILE__, __LINE__ ))

}}

#endif
