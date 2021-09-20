/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/

#ifndef PARAMS_H
#define PARAMS_H

namespace Brush
{

struct Params
{
    std::map<string, float> mutation_options;
    /// maximum program depth
    int max_depth;
    /// maximum program breadth (max arity of a node)
    int max_breadth;
    /// maximum program size (total nodes)
    int max_size;
    
    Params()
    { 
        mutation_options = {
                            {"point",   0.5},
                            {"insert",  0.25},
                            {"delete",  0.25}
                           };
        max_depth = 4;
        max_breadth = 5;
        max_size = 20;
    }
};

static Params params;

} // Brush

#endif