/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/

#ifndef NODE_H
#define NODE_H

#include "data/data.h"
#include "nodemap.h"
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

using Brush::data::DataType;
using Brush::data::Data;

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
    During,

    //split
    SplitBest,
    SplitOn,

    // leaves
    Constant,
    Variable,

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
/* enum class NodeGroup : __UINT32_TYPE__ */
/* { */
/*     UnaryOperator, */
/*     BinaryOperator, */
/*     NaryOperator, */
/*     Split, */
/*     Leaf, */
/* }; */

enum class ExecType : uint32_t {
    Mapper, // maps child nodes to output via function call.
    Splitter, // splits data and returns the output of the children on each split.
    Terminal    // returns a data element.
}; 

struct Node {

    /// full name of the node, with types
    string name;
    // whether to center the operator in pretty printing
    bool center_op;
    // chance of node being selected for variation
    float prob_change; 
    // /// unique id
    // int ID;
    // static int sNextId;
    // inline int getNextId() { return ++sNextId; };

    NodeType op_type;
    ExecType exec_type;
    DataType ret_type;
    std::vector<DataType> arg_types;
    bool is_differentiable;
    bool is_weighted;
    bool optimize;
    vector<float> W; 
    float threshold; // just use W.at(0)? 
    string feature; // feature for terminals or splitting nodes 


    Node() = default; 

    explicit Node(NodeType type, DataType output_type) noexcept
        : op_type(type)
        , ret_type(output_type)
        , exec_type(NodeSchema[type]["ExecType"])
        , arg_types(NodeSchema[type]["Mapping"][output_type])
    {
    }

    explicit Node(NodeType type, bool weighted) noexcept
        : op_type(type)
        , is_weighted(weighted)
    {
        // if (Type < NodeType::Abs) // Add, Mul, Sub, Div, Aq, Pow
        // {
        //     Arity = 2;
        // } else if (Type < NodeType::Dynamic) // Log, Exp, Sin, Cos, Tan, Tanh, Sqrt, Cbrt, Square
        // {
        //     Arity = 1;
        // }
        // Length = Arity;
        // IsEnabled = true;
        // Optimize = IsLeaf(); // we only optimize leaf nodes
        // Value = 1.;
    }

    /* static auto Constant(double value) */
    /* { */
    /*     Node node(NodeType::Constant); */
    /*     node.Value = static_cast<Operon::Scalar>(value); */
    /*     return node; */
    /* } */

    [[nodiscard]] auto Name() const noexcept -> std::string const&;
    [[nodiscard]] auto Desc() const noexcept -> std::string const&;

    // get return type and argument types. 
    // these should come from a mapping. 
    DataType get_ret_type() const { return ret_type; }; 
    std::type_index args_type() const { 
        return typeid(make_tuple(arg_types.begin(), arg_types.end()));
    }; 
    size_t get_arg_count() const { return arg_types.size(); };
    // comparison operators
    // inline auto operator==(const Node& rhs) const noexcept -> bool
    // {
    //     return CalculatedHashValue == rhs.CalculatedHashValue;
    // }

    // inline auto operator!=(const Node& rhs) const noexcept -> bool
    // {
    //     return !((*this) == rhs);
    // }

    // inline auto operator<(const Node& rhs) const noexcept -> bool
    // {
    //     return std::tie(HashValue, CalculatedHashValue) < std::tie(rhs.HashValue, rhs.CalculatedHashValue);
    // }

    // inline auto operator<=(const Node& rhs) const noexcept -> bool
    // {
    //     return ((*this) == rhs || (*this) < rhs);
    // }

    // inline auto operator>(const Node& rhs) const noexcept -> bool
    // {
    //     return !((*this) <= rhs);
    // }

    // inline auto operator>=(const Node& rhs) const noexcept -> bool
    // {
    //     return !((*this) < rhs);
    // }

    template <NodeType... T>
    [[nodiscard]] inline auto Is() const -> bool { return ((op_type == T) || ...); }

    [[nodiscard]] inline auto IsLeaf() const noexcept -> bool { 
        return Is<NodeType::Constant, NodeType::Variable>(); 
    }
    [[nodiscard]] inline auto IsCommutative() const noexcept -> bool { 
        return Is<NodeType::Add,
                  NodeType::Mul,
                  NodeType::Min,
                  NodeType::Max>(); 
    }


    // need to figure out how to define these for NodeTypes. 
    // different operators need different flow through fit and predict - 
    // for example, split nodes need to run a function on the data, then
    // pass different data chunks to the children. meanwhile math ops mostly
    // pull their children first and then do a computation on the arguments.
    /* auto fit(const Data&, TreeNode*&, TreeNode*&); */
    /* auto predict(const Data&, TreeNode*&, TreeNode*&); */
    /* void grad_descent(const ArrayXf&, const Data&, */ 
    /*                             TreeNode*&, TreeNode*&); */
    /* string get_model(bool pretty, */ 
    /*                             TreeNode*& first_child, */
    /*                             TreeNode*& last_child) const; */
    /* string get_tree_model(bool pretty, string offset, */ 
    /*                                 TreeNode *&first_child, */
    /*                                 TreeNode *&last_child) const ; */
    /* // naming */
    /* string get_name() const {return this->name;}; */
    /* string get_op_name() const {return this->op_name;}; */
    /* void set_name(string n){this->name = n;}; */
    /* void set_op_name(string n){this->op_name = n;}; */
    /* // changing */
    /* float get_prob_change(){ return this->prob_change;}; */
    /* void set_prob_change(float w){ this->prob_change = w;}; */
    /* float get_prob_keep(){ return 1-this->prob_change;}; */
};

}
#endif
