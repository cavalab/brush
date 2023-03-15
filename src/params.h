/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/

#ifndef PARAMS_H
#define PARAMS_H

namespace Brush
{

// TODO: just make this a json object that can be controlled in Python or c++
static json PARAMS;
json PARAMS = {
    {"mutation_options", {
        {"point",   0.5},
        {"insert",  0.25},
        {"delete",  0.25}
    }},
    {"max_depth", 4},
    {"max_size", 30}
};
// struct Params
// {
//     std::map<string, float> mutation_options;
//     /// maximum program depth
//     int max_depth;
//     /// maximum program breadth (max arity of a node)
//     int max_breadth;
//     /// maximum program size (total nodes)
//     int max_size;
    
//     Params()
//     { 
//         mutation_options = {
//                             {"point",   0.5},
//                             {"insert",  0.25},
//                             {"delete",  0.25}
//                            };
//         max_depth = 4;
//         max_breadth = 5;
//         max_size = 20;
//     }

//     void set_params(const json& p)
//     {
//         for (const auto& el : p)
//         {


//         }
//     }

// };

// static Params params;

} // Brush

#endif
