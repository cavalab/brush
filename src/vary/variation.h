/* Brush

copyright 2020 William La Cava
license: GNU/GPL v3
*/
#ifndef VARIATION_H
#define VARIATION_H


// #include "util/error.h"
// #include "util/utils.h"

//#include "search_space.h"
#include "../pop/population.h"

#include <map>
#include <optional>

// namespace Brush{

// typedef tree<Node>::pre_order_iterator Iter; 

////////////////////////////////////////////////////////////////////////////
// Mutation & Crossover

using namespace Brush::Pop;

/**
 * @brief Namespace for variation functions like crossover and mutation. 
 * 
 */
namespace Brush {
namespace Var {

// base for MUTATION variators
class MutationBase {
public:
    using Iter = tree<Node>::pre_order_iterator;

    // TODO: static methods, without storing information, and using just SS and params as arguments
    MutationBase(const SearchSpace& SS, size_t max_size, size_t max_depth)
        : SS_(SS)
        , max_size_(max_size)
        , max_depth_(max_depth)
    {}
        
    virtual auto find_spots(tree<Node>& Tree) const -> vector<float>
    {
        vector<float> weights(Tree.size());

        // by default, mutation can happen anywhere, based on node weights
        std::transform(Tree.begin(), Tree.end(), weights.begin(),
                       [&](const auto& n){ return n.get_prob_change();});
        
        // Should have same size as prog.Tree.size, even if all weights <= 0.0
        return weights;
    }

    virtual auto operator()(tree<Node>& Tree, Iter spot) const -> bool = 0;

    auto SS() const -> SearchSpace { return SS_; }
    auto max_size() const -> size_t { return max_size_; }
    auto max_depth() const -> size_t{ return max_depth_; }
protected:
    static size_t size_with_weights(tree<Node>& Tree, bool include_weight=true)
    {
        // re-implementation of int Node::size(bool include_weight=true) meant
        // to work with the tree<Node> instead of brush's programs.
        // TODO: find a better way to have this function available to mutations
        // and avoid repeated functions
        size_t acc = 0;

        std::for_each(Tree.begin(), Tree.end(), 
            [include_weight, &acc](auto& node){ 
                ++acc; // the node operator or terminal
                
                // TODO: the same size check occurs in search_space.cpp and program.h. Make a function (stop doing hardcoded)
                // SplitBest has an optimizable decision tree consisting of 3 nodes
                // (terminal, arithmetic comparison, value) that needs to be taken
                // into account
                if (Is<NodeType::SplitBest>(node.node_type))
                    acc += 3;
                else if (Is<NodeType::SplitOn>(node.node_type))
                    acc += 2;

                if ( (include_weight && node.get_is_weighted()==true)
                &&   Isnt<NodeType::Constant, NodeType::MeanLabel>(node.node_type) )
                    // Taking into account the weight and multiplication, if enabled.
                    // weighted constants still count as 1 (simpler than constant terminals)
                    acc += 2;
             });

        return acc;
    }

private:
    SearchSpace SS_; // where to sample nodes to change the program

    // constrains TODO: use params to get this values, stop storing it
    size_t max_size_;
    size_t max_depth_;
};

// TODO: make sure every method doesnt store information, instead they retrieve it from parameters (so there's no side effect)

template<ProgramType T>
class Variation 
{
private:
    SearchSpace search_space;
    Parameters parameters; // stop using this thing here and get parameter as argument
public:
    Variation() = default;
    
    Variation(Parameters& params, SearchSpace& ss)
        : parameters(params)
        , search_space(ss)
    {};

    ~Variation() {};

    void init(Parameters& params, SearchSpace& ss){
        this->parameters = params;
        this->search_space = ss;
    };

    // individual-level variations
    std::optional<Individual<T>> cross(const Individual<T>& mom, const Individual<T>& dad);
    std::optional<Individual<T>> mutate(const Individual<T>& parent);

    /// method to handle variation of population
    void vary(Population<T>& pop, int island, const vector<size_t>& parents,
              const Parameters& p);
};

} //namespace Var
} //namespace Brush
#endif