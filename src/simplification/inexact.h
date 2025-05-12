// simplification maps are based on trainingdata
// should ignore fixed nodes ---> does not change subtrees if they contain fixed nodes (also this should be applied to the constants simplification)
// number of tables and max_samples_to_use are parameters, set default values enough to make it work off-the-shelf
// should I implement json serialization?
// maybe this function should be templated. should handle constant predictions
        

#ifndef INEXACT_H
#define INEXACT_H

#include "../init.h"
#include "../types.h"
#include "../program/program.h"
#include "../vary/search_space.h"
#include "../util/utils.h"

using namespace std;
using Brush::Node;
using Brush::DataType;

namespace Brush { namespace Simpl{
    
class HashStorage {
public:
    HashStorage() {};
    ~HashStorage() {};

    void append(const string& key, const ArrayXf& value) {
        if (storage.find(key) == storage.end())
            storage[key] = vector<ArrayXf>{value};
        else
            storage[key].push_back(value);
    }

    vector<ArrayXf> getList(const string& key) {
        auto it = storage.find(key);
        if (it != storage.end())
            return it->second;

        static vector<ArrayXf> empty_list;
        return empty_list;
    }

    void clear() {
        storage.clear();
    }

    vector<string> keys() {
        vector<string> result;
        for (const auto& pair : storage) {
            result.push_back(pair.first);
        }
        return result;
    }

private:
    map<string, vector<ArrayXf>> storage;
};

class Inexact_simplifier
{
    public:
        // static Inexact_simplifier* initSimplifier();
        void initUniformPlanes(int hashSize, int inputDim, int numPlanes);
        
        // static void destroy();

        // TODO: mode templated stuff to cpp. right now they are in the header
        // because of the templating, that does not work with testing. 
        // This is happening in inexact, constants, variation.
        template<Brush::ProgramType PT>
        Program<PT> simplify_tree(Program<PT>& program,
                                    const SearchSpace &ss, const Dataset &d)
        {
            Program<PT> simplified_program(program);

            // iterate over the tree, trying to replace each node with a constant, and keeping the change if the pred does not change.
            TreeIter spot = simplified_program.Tree.begin();
            while(spot != simplified_program.Tree.end())
            {
                // we dont index or simplify fixed stuff.
                // non-wheightable nodes are not simplified. TODO: revisit this and see if they should (then implement it)
                // This is avoiding using booleans.
                if (spot.node->data.get_prob_change() > 0
                // &&  IsWeighable(spot.node->data.ret_type) && IsWeighable(spot.node->data.node_type)
                ) {
                    // indexing only small subtrees or non-constant-terminal nodes
                    if (simplified_program.size_at(spot) < 10
                    ||  Isnt<NodeType::Constant, NodeType::MeanLabel, NodeType::Terminal>(spot.node->data.node_type)) {
                        index(spot, d);
                    }
                    
                    // TODO: use IsLeaf here instead of checking for each possible nodetype. also search throughout the code and replace it
                    if (Isnt<NodeType::Constant, NodeType::MeanLabel, NodeType::Terminal>(spot.node->data.node_type)){
                        // res will return the closest within the threshold, so we dont have to check distance here
                        auto res = query(spot, d); // optional<pair<size_t, string>>

                        if (res){
                            auto key = res.value(); // table index and hash
                            const tree<Node> branch(spot);
                                
                            if (equivalentExpression.find(key) == equivalentExpression.end()) {
                                equivalentExpression[key] = branch;
                            } else if (spot.node->get_size(false) < equivalentExpression[key].begin().node->get_size(false)){                
                                equivalentExpression[key] = branch;
                            } else if (spot.node->get_size(false) > equivalentExpression[key].begin().node->get_size(false)){                         
                                const tree<Node> simplified_branch(equivalentExpression[key]);
                                simplified_program.Tree.erase_children(spot); 
                                spot = simplified_program.Tree.move_ontop(spot, simplified_branch.begin());
                            }
                        }
                    }
                }
                ++spot;
            }    
            program.Tree = simplified_program.Tree;

            return simplified_program;
        }
        Inexact_simplifier();
        ~Inexact_simplifier();
        
        // TODO: make index private and initUniformPlanes to add terminals from search space
        void index(TreeIter& spot, const Dataset &d);
    private:

        vector<string> hash(const ArrayXf& inputPoint); // one string for each plane


        // will return the hash and the distance to the queryPoint
        optional<pair<size_t, string>> query(TreeIter& spot, const Dataset &d);

        // one storage instance for each datatype/rettype.
        // the storage will be used to calculate the hash and query the
        // collection of hashes, returning the closest ones,
        // and the list will contain equivalent expressions, ordered by size
        // (or linear complexity). So we dont store pairs in the storage
        // TODO: improve how I handle different return types (should I use a map?)
        HashStorage storageBool;
        HashStorage storageInt;
        HashStorage storageFloat;

        map<pair<size_t, string>, tree<Node>> equivalentExpression;

        vector<MatrixXf> uniformPlanes;

        // private static attribute used by every instance of the class
        // static Inexact_simplifier* instance;
};

// static Inexact_simplifier &inexact_simplifier = *Inexact_simplifier::initSimplifier();

} // Simply
} // Brush

#endif
