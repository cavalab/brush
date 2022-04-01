/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/

#ifndef NODE_H
#define NODE_H

// #include "nodes/base.h"
// #include "nodes/dx.h"
// #include "nodes/split.h"
// #include "nodes/terminal.h"
////////////////////////////////////////////////////////////////////////////////
/*
Node overhaul:

- Incorporating new design principles, learning much from operon:
    - make Node trivial, so that it is easily copied around. 
    - use Enums and maps to define node information. This kind of abandons the object oriented approach taken thus far, but it should make extensibility easier and performance better in the long run. 
    - Leverage ceres for parameter optimization. No more defining analytical 
    derivatives for every function. Let ceres do that. 
        - sidenote: not sure ceres can handle the data flow of split nodes. 
        need to figure out. 
        - this also suggests turning TimeSeries back into EigenSparse matrices.
    - forget all the runtime node generation. It saves space at the cost of 
    unclear code. I might as well just define all the nodes that are available, plainly. At run-time this will be faster. 
    - keep an eye towards extensibility by defining a custom node registration function that works.

*/
namespace Brush{

enum class NodeType : uint32_t {
    //arithmetic
    Add,
    Sub,
    Mul,
    Div,
    Aq,
    Abs,

    Acos,
    Asin,
    Atan,
    Cos,
    Cosh,
    Sin,
    Sinh,
    Tan,
    Tanh,
    Cbrt,
    Ceil,
    Floor,
    Exp,
    Log,
    Logabs,
    Log1p,
    Sqrt,
    Sqrtabs,
    Square,
    Pow,

    // logic; not sure these will make it in
    And,
    Or,
    Not,
    Xor,

    // decision (same)
    Equals,
    LessThan,
    GreaterThan,
    Leq,
    Geq,

    // summary stats
    Min,
    Max,
    Mean,
    Median,
    Count,
    Sum,

    // timing masks
    Before,
    After,
    During

    //split
    SplitBest,
    SplitOn,

    // leaves
    Constant,
    Variable 

    // custom
    CustomOp,
    CustomSplit
};

// TODO:
// define NodeGroup Enum
// define a map of NodeTypes to their input-output mappings
// define the actual templated functions

/* defines groupings of nodes that share common fitting conditions. 
*/
enum class NodeGroup : __UINT32_TYPE__
{
    UnaryOperator,
    BinaryOperator,
    NaryOperator,
    Split,
    Leaf,
};

}
#endif
