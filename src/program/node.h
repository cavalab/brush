/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3

Node class design heavily inspired by Operon, (c) Heal Research
https://github.com/heal-research/operon/
*/

#ifndef NODE_H
#define NODE_H

#include "../data/data.h"
#include "nodetype.h"
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

// TODO: should I move this declaration to another place?
template <DataType... T>
inline auto Isnt(DataType dt) -> bool { return !((dt == T) || ...); }

template<DataType DT>
inline auto IsWeighable() noexcept -> bool { 
        return Isnt<DataType::ArrayB,
                    DataType::MatrixB,
                    DataType::TimeSeriesB,
                    DataType::ArrayBJet,
                    DataType::MatrixBJet
                    >(DT);                
}
inline auto IsWeighable(DataType dt) noexcept -> bool { 
        return Isnt<DataType::ArrayB,
                    DataType::MatrixB,
                    DataType::TimeSeriesB,
                    DataType::ArrayBJet,
                    DataType::MatrixBJet
                    >(dt);                
}

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
    /// whether to center the operator in pretty printing
    bool center_op;
    /// chance of node being selected for variation
    float prob_change; 
    /// whether node is modifiable
    bool fixed;
    /// @brief the node type
    NodeType node_type;
    /// @brief a hash of the signature
    std::size_t sig_hash;
    /// @brief a hash of the dual of the signature (for NLS)
    std::size_t sig_dual_hash;
    /// @brief return data type
    DataType ret_type;
    /// @brief argument data types
    std::vector<DataType> arg_types;
    /// @brief whether this node is weighted
    bool is_weighted;
    /// @brief the weights of the node. also used for splitting thresholds.
    float W; 
    // /// @brief a node hash / unique ID for the node, except weights
    // size_t node_hash; 
    /// @brief tuple type for hashing
    using HashTuple = std::tuple< 
        UnderlyingNodeType,     // node type
        size_t,                 // sig_hash
        bool,                   // is_weighted
        string,                 // feature
        bool,                   // fixed
        int                     // rounded W
        // float                   // prob_change
    >;

    // Node(){init();}; 
    Node() = default;


    /// @brief Constructor used by search space 
    /// @tparam S signature 
    /// @param type node type
    /// @param feature_name name of the terminal 
    /// @param signature signature 
    template<typename S>
    explicit Node(NodeType type, S signature, bool weighted=false, string feature_name="") noexcept
        : node_type(type)
        , name(NodeTypeName[type])
        , ret_type(S::get_ret_type())
        , arg_types(S::get_arg_types())
        , sig_hash(S::hash())
        , sig_dual_hash(S::Dual::hash())
        , is_weighted(weighted)
        , feature(feature_name)
    {
        init();
    }

    template<typename S>
    void set_signature()
    {
        ret_type = S::get_ret_type();
        arg_types = S::get_arg_types(); 
        sig_hash = S::hash();
        sig_dual_hash = S::Dual::hash();
        // set_node_hash();
    }

    void init(){

        W = 1.0;
        // set_node_hash();
        fixed=false;
        set_prob_change(1.0);

        // cant weight an boolean terminal
        if (!IsWeighable(this->ret_type)) 
            this->is_weighted = false;
    }

    /// @brief gets a string version of the node for printing.
    /// @param include_weight whether to include the node's weight in the output.
    /// @return string version of the node.
    string get_name(bool include_weight=true) const noexcept; 
    string get_model(const vector<string>&) const noexcept; 

    // get return type and argument types. 
    inline DataType get_ret_type() const { return ret_type; }; 
    inline std::size_t args_type() const { return sig_hash; };
    inline auto get_arg_types() const { return arg_types; };
    inline size_t get_arg_count() const { return arg_types.size(); };

    // void set_node_hash(){
    //     node_hash = std::hash<HashTuple>{}(HashTuple{
    //             NodeTypes::GetIndex(node_type),
    //             sig_hash,
    //             is_weighted,
    //             feature,
    //             fixed,
    //             W,
    //             prob_change
    //     });
    //     // fmt::print("nodetype:{}; hash tuple:{}; node_hash={}\n", node_type, tmp, node_hash);
    // }
    size_t get_node_hash() const {
        return std::hash<HashTuple>{}(HashTuple{
                NodeTypes::GetIndex(node_type),
                sig_hash,
                is_weighted,
                feature,
                fixed,
                int(W*100)
        });
    }
    ////////////////////////////////////////////////////////////////////////////////
    //comparison operators
    inline auto operator==(const Node& rhs) const noexcept -> bool
    {
        /* return CalculatedHashValue == rhs.CalculatedHashValue; */
        return get_node_hash() == rhs.get_node_hash();
        /* return (*this) == rhs; */
    }

    inline auto operator!=(const Node& rhs) const noexcept -> bool
    {
        return !((*this) == rhs);
    }

    inline auto operator<(const Node& rhs) const noexcept -> bool
    {
        /* return std::tie(HashValue, CalculatedHashValue) < std::tie(rhs.HashValue, rhs.CalculatedHashValue); */
        return get_node_hash() < rhs.get_node_hash(); 
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
    float get_prob_change() const { return fixed ? 0.0 : this->prob_change;};
    void set_prob_change(float w){ this->prob_change = w;};
    float get_prob_keep() const { return fixed ? 1.0 : 1.0-this->prob_change;};

    inline void set_feature(string f){ feature = f; };
    inline string get_feature() const { return feature; };

    inline bool get_is_weighted() const {return this->is_weighted;};
    inline void set_is_weighted(bool is_weighted){
        // cant change the weight of a boolean terminal
        if (IsWeighable(this->ret_type)) 
            this->is_weighted = is_weighted;
    };

    private:

        /// @brief feature name for terminals or splitting nodes
        string feature; 
};

template <NodeType... T>
inline auto Is(NodeType nt) -> bool { return ((nt == T) || ...); }

template <NodeType... T>
inline auto Isnt(NodeType nt) -> bool { return !((nt == T) || ...); }

inline auto IsLeaf(NodeType nt) noexcept -> bool { 
    return Is<NodeType::Constant, NodeType::Terminal, NodeType::MeanLabel>(nt); 
}

inline auto IsCommutative(NodeType nt) noexcept -> bool { 
    return Is<NodeType::Add,
              NodeType::Mul,
              NodeType::Min,
              NodeType::Max
              >(nt); 
}

inline auto IsDifferentiable(NodeType nt) noexcept -> bool { 
    return Isnt<
                NodeType::Ceil,
                NodeType::Floor,
                NodeType::Before,       
                NodeType::After,          
                NodeType::During,
                NodeType::Count,
                NodeType::And, 
                NodeType::Or,
                NodeType::Not
                >(nt);                
}
template<NodeType NT>
inline auto IsWeighable() noexcept -> bool { 
        return Isnt<
                    NodeType::Ceil,
                    NodeType::Floor,
                    NodeType::Before,       
                    NodeType::After,          
                    NodeType::During,
                    NodeType::Count,
                    NodeType::SplitOn,
                    NodeType::SplitBest,
                    NodeType::And, 
                    NodeType::Or,
                    NodeType::Not 
                    >(NT);                
}
inline auto IsWeighable(NodeType nt) noexcept -> bool { 
        return Isnt<
                    NodeType::Ceil,
                    NodeType::Floor,
                    NodeType::Before,       
                    NodeType::After,          
                    NodeType::During,
                    NodeType::Count,
                    NodeType::SplitOn,
                    NodeType::SplitBest,
                    NodeType::And, 
                    NodeType::Or,
                    NodeType::Not
                    >(nt);                
}

ostream& operator<<(ostream& os, const Node& n);
ostream& operator<<(ostream& os, const NodeType& nt);



void from_json(const json &j, Node& p);
void to_json(json& j, const Node& p);
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
