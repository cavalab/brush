/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/

#ifndef PARAMS_H
#define PARAMS_H
#include "init.h"
namespace ns = nlohmann;
namespace Brush
{
    extern ns::json PARAMS;
    void set_params(const ns::json& j);
    ns::json get_params();
} // Brush

#endif
