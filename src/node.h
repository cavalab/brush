/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/

#ifndef NODE_H
#define NODE_H

#include "data/data.h"
#include "nodemap.h"
#include "util/utils.h"
#include <iostream>
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
using Brush::DataType;
using Brush::ExecType;
using Brush::data::Data;

namespace Brush{

struct uint32_vector_hasher {
    std::size_t operator()(std::vector<uint32_t> const& vec) const {
      std::size_t seed = vec.size();
      for(auto& i : vec) {
        seed ^= i + 0x9e3779b9 + (seed << 6) + (seed >> 2);
      }
      return seed;
    }
    std::size_t operator()(std::vector<Brush::DataType> const& vec) const {
      std::size_t seed = vec.size();
      for(auto& i : vec) {
        seed ^= uint32_t(i) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
      }
      return seed;
    }
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

    NodeType node_type;
    /* ExecType exec_type; */
    std::size_t sig_hash;
    DataType ret_type;
    std::vector<DataType> arg_types;
    bool is_differentiable;
    bool is_weighted;
    bool optimize;
    vector<float> W; 
    float threshold; // just use W.at(0)? 
    string feature; // feature for terminals or splitting nodes 


    Node() = default; 

    explicit Node(NodeType type, std::size_t sig, DataType output_type, bool weighted=false) noexcept
        : node_type(type)
        , name(NodeTypeName[type])
        , sig_hash(sig)
        , ret_type(output_type)
        , is_weighted(weighted)
    {
        cout << "instantiated " << name << "with sig hash " << sig_hash << " and return type " << DataTypeName.at(ret_type) << endl;
    }

    explicit Node(NodeType type, DataType output_type, string feature_name) noexcept
        : node_type(type)
        , name(NodeTypeName[type])
        /* , sig_hash(SigType::) */
        , ret_type(output_type)
        , feature(feature_name)
        /* , exec_type(NodeSchema[NodeTypeName[type]]["ExecType"]) */
        /* , arg_types(NodeSchema[NodeTypeName[type]]["Signature"][DataTypeName[output_type]]) */
        , is_weighted(false)
    {
        //TODO: set sig_hash
        cout << "instantiated " << name << " from feature " << feature << " with output type " << DataTypeName.at(ret_type) << endl;
        sig_hash=typeid(void).hash_code();
        
    }
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

    /* static auto Constant(double value) */
    /* { */
    /*     Node node(NodeType::Constant); */
    /*     node.Value = static_cast<Operon::Scalar>(value); */
    /*     return node; */
    /* } */

    auto get_name() const noexcept -> std::string const&;
    /* auto get_desc() const noexcept -> std::string const&; */

    // get return type and argument types. 
    // these should come from a mapping. 
    DataType get_ret_type() const { return ret_type; }; 
    std::size_t args_type() const { 
        return uint32_vector_hasher()(arg_types);
    }; 
    size_t get_arg_count() const { return arg_types.size(); };

    //comparison operators
    inline auto operator==(const Node& rhs) const noexcept -> bool
    {
        /* return CalculatedHashValue == rhs.CalculatedHashValue; */
        return (*this) == rhs;
    }

    inline auto operator!=(const Node& rhs) const noexcept -> bool
    {
        return !((*this) == rhs);
    }

    inline auto operator<(const Node& rhs) const noexcept -> bool
    {
        /* return std::tie(HashValue, CalculatedHashValue) < std::tie(rhs.HashValue, rhs.CalculatedHashValue); */
        return (*this) < rhs;
    }

    inline auto operator<=(const Node& rhs) const noexcept -> bool
    {
        return ((*this) == rhs || (*this) < rhs);
    }

    inline auto operator>(const Node& rhs) const noexcept -> bool
    {
        return !((*this) <= rhs);
    }

    inline auto operator>=(const Node& rhs) const noexcept -> bool
    {
        return !((*this) < rhs);
    }

    template <NodeType... T>
    inline auto Is() const -> bool { return ((node_type == T) || ...); }

    inline auto IsLeaf() const noexcept -> bool { 
        return Is<NodeType::Constant, NodeType::Terminal>(); 
    }

    inline auto IsCommutative() const noexcept -> bool { 
        return Is<NodeType::Add,
                  NodeType::Mul,
                  NodeType::Min,
                  NodeType::Max>(); 
    }

    inline auto IsDifferentiable() const noexcept -> bool { 
        return !Is<
                    NodeType::Ceil,
                    NodeType::Floor,
                    NodeType::Not,              
                    NodeType::Before,       
                    NodeType::After,          
                    NodeType::During,
                    NodeType::Count,
                    NodeType::And, 
                    NodeType::Or,
                    NodeType::Xor, 
                    NodeType::Equals,
                    NodeType::LessThan,
                    NodeType::Leq,
                    NodeType::Geq
                    >();                
    }

    inline auto IsWeighable() const noexcept -> bool { 
        return !Is<
                    NodeType::Ceil,
                    NodeType::Floor,
                    NodeType::Not,              
                    NodeType::Before,       
                    NodeType::After,          
                    NodeType::During,
                    NodeType::Count,
                    NodeType::And, 
                    NodeType::Or,
                    NodeType::Xor, 
                    NodeType::Equals,
                    NodeType::LessThan,
                    NodeType::Leq,
                    NodeType::Geq,
                    NodeType::SplitOn,
                    NodeType::SplitBest
                    >();                
    }

    inline decltype(auto) signature() const { 
        return NodeSchema[NodeTypeName[node_type]]["Signature"][DataTypeName[ret_type]]; 
    };
    /* template<ExecType E> auto tupleargs() const; */

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
    float get_prob_change() const { return this->prob_change;};
    void set_prob_change(float w){ this->prob_change = w;};
    float get_prob_keep() const { return 1-this->prob_change;};
    };

    ostream& operator<<(ostream& os, const Node& n);
}

#endif
