/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3

Node class design heavily inspired by Operon, (c) Heal Research
https://github.com/heal-research/operon/
*/

#ifndef NODE_H
#define NODE_H

#include "../data/data.h"
#include "nodemap.h"
#include "../util/utils.h"
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
using Brush::Data::Dataset;

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

/**
 * @brief class holding the data for a node in a tree.
 * 
 */
struct Node {

    /// full name of the node, with types
    string name;
    // whether to center the operator in pretty printing
    bool center_op;
    // chance of node being selected for variation
    float prob_change; 
    // whether node is modifiable
    bool fixed;
    NodeType node_type;
    std::size_t sig_hash;
    std::size_t sig_dual_hash;
    DataType ret_type;
    std::vector<DataType> arg_types;
    bool is_weighted;
    bool optimize;
    vector<float> W; 
    string feature; // feature for terminals or splitting nodes 
    size_t complete_hash; 


    Node() = default; 


    template<typename S>
    explicit Node(NodeType type, S signature, bool weighted=false) noexcept
        : node_type(type)
        , name(NodeTypeName[type])
        , arg_types(S::get_arg_types())
        , sig_hash(S::hash())
        , sig_dual_hash(S::Dual::hash())
        , ret_type(DataTypeEnum<typename S::RetType>::value)
        , is_weighted(weighted)
    {
        /* cout << "instantiated " << name << " with sig hash " << sig_hash << " and return type " << DataTypeName.at(ret_type) << endl; */
        optimize=false;

        if (weighted){   
            optimize=true;
            W.resize(arg_types.size());
            for (int i = 0; i < W.size(); ++i)
                W.at(i) = 1.0;  
            optimize=true;
        }
        else if (Util::in(vector<NodeType>{NodeType::SplitOn, NodeType::SplitBest}, type))
            W.resize(1); // W.at(0) represents the threshold of the split
        else

        set_complete_hash();
        set_prob_change(1.0);
        fixed=false;
    }
    // explicit Node(NodeType type, const vector<DataType>& args, std::size_t sig, DataType output_type, bool weighted=false) noexcept
    //     : node_type(type)
    //     , name(NodeTypeName[type])
    //     , arg_types(args)
    //     , sig_hash(sig)
    //     , ret_type(output_type)
    //     , is_weighted(weighted)
    // {
    //     /* cout << "instantiated " << name << " with sig hash " << sig_hash << " and return type " << DataTypeName.at(ret_type) << endl; */
    //     optimize=false;

    //     if (weighted){   
    //         optimize=true;
    //         W.resize(args.size());
    //         for (int i = 0; i < W.size(); ++i)
    //             W.at(i) = 1.0;  
    //         optimize=true;
    //     }
    //     else if (Util::in(vector<NodeType>{NodeType::SplitOn, NodeType::SplitBest}, type))
    //         W.resize(1); // W.at(0) represents the threshold of the split
    //     else

    //     set_complete_hash();
    //     set_prob_change(1.0);
    //     fixed=false;
    // }

    template<typename S>
    explicit Node(NodeType type, string feature_name, S signature) noexcept
        : node_type(type)
        , name(NodeTypeName[type])
        , feature(feature_name)
        , ret_type(DataTypeEnum<typename S::RetType>::value)
        , sig_hash(S::hash())
        , sig_dual_hash(S::Dual::hash())
        , is_weighted(false)
    {
        /* cout << "instantiated " << name << " from feature " << feature << " with output type " << DataTypeName.at(ret_type) << endl; */
        optimize=false;
        arg_types = vector<DataType>{};
        set_complete_hash();
        set_prob_change(1.0);
        fixed=false;
    }

    auto get_name() const noexcept -> std::string; 
    /* auto get_desc() const noexcept -> std::string const&; */

    // get return type and argument types. 
    inline DataType get_ret_type() const { return ret_type; }; 
    inline std::size_t args_type() const { return sig_hash; };
    inline auto get_arg_types() const { return arg_types; };
    inline size_t get_arg_count() const { return arg_types.size(); };

    void set_complete_hash(){
        using Tuple = std::tuple< UnderlyingNodeType, size_t, bool, bool, string >;
        complete_hash = std::hash<Tuple>{}(Tuple{
                NodeTypes::GetIndex(node_type),
                sig_hash,
                is_weighted,
                optimize,
                feature
                });
    }
    ////////////////////////////////////////////////////////////////////////////////
    //comparison operators
    inline auto operator==(const Node& rhs) const noexcept -> bool
    {
        /* return CalculatedHashValue == rhs.CalculatedHashValue; */
        return complete_hash == rhs.complete_hash;
        /* return (*this) == rhs; */
    }

    inline auto operator!=(const Node& rhs) const noexcept -> bool
    {
        return !((*this) == rhs);
    }

    inline auto operator<(const Node& rhs) const noexcept -> bool
    {
        /* return std::tie(HashValue, CalculatedHashValue) < std::tie(rhs.HashValue, rhs.CalculatedHashValue); */
        return complete_hash < complete_hash; 
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

    ////////////////////////////////////////////////////////////////////////////////
    // getters and setters
    //TODO revisit
    float get_prob_change() const { return this->prob_change;};
    void set_prob_change(float w){ if (!fixed) this->prob_change = w;};
    float get_prob_keep() const { return 1-this->prob_change;};
};

//TODO: add nt to template as first argument, make these constexpr
template <NodeType... T>
inline auto Is(NodeType nt) -> bool { return ((nt == T) || ...); }

inline auto IsLeaf(NodeType nt) noexcept -> bool { 
    return Is<NodeType::Constant, NodeType::Terminal>(nt); 
}

inline auto IsCommutative(NodeType nt) noexcept -> bool { 
    return Is<NodeType::Add,
              NodeType::Mul,
              NodeType::Min,
              NodeType::Max>(nt); 
}

inline auto IsDifferentiable(NodeType nt) noexcept -> bool { 
    return !Is<
                NodeType::Ceil,
                NodeType::Floor,
                // NodeType::Not,              
                NodeType::Before,       
                NodeType::After,          
                NodeType::During,
                NodeType::Count
                // NodeType::And, 
                // NodeType::Or,
                // NodeType::Xor 
                /* NodeType::Equals, */
                /* NodeType::LessThan, */
                /* NodeType::Leq, */
                /* NodeType::Geq */
                >(nt);                
}
template<NodeType NT>
inline auto IsWeighable() noexcept -> bool { 
        return !Is<
                    NodeType::Ceil,
                    NodeType::Floor,
                    // NodeType::Not,              
                    NodeType::Before,       
                    NodeType::After,          
                    NodeType::During,
                    NodeType::Count
                    // NodeType::And, 
                    // NodeType::Or,
                    // NodeType::Xor
                    /* NodeType::Equals, */
                    /* NodeType::LessThan, */
                    /* NodeType::Leq, */
                    /* NodeType::Geq, */
                    /* NodeType::SplitOn, */
                    /* NodeType::SplitBest */
                    >(NT);                
    }
ostream& operator<<(ostream& os, const Node& n);
ostream& operator<<(ostream& os, const NodeType& nt);


} // namespace Brush

// format overload for Nodes
template <> struct fmt::formatter<Brush::Node>: formatter<string_view> {
  // parse is inherited from formatter<string_view>.
  template <typename FormatContext>
  auto format(Brush::Node x, FormatContext& ctx) const {
    return formatter<string_view>::format(x.get_name(), ctx);
  }
};

#endif
