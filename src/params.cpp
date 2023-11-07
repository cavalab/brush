/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/
#include "params.h"

namespace Brush
{
    
nlohmann::json PARAMS;
void set_params(const ns::json& j) { PARAMS = j; }
ns::json get_params(){ return PARAMS;}

} // Brush
