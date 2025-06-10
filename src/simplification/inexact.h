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
    HashStorage(int numPlanes=10) { 
        storage.clear();
        storage.reserve(numPlanes);
        for (int i = 0; i < numPlanes; ++i)
            storage.push_back(map<size_t, vector<tree<Node>>>());        
    };
    ~HashStorage() {};

    void append(const int& storage_n, const size_t& key, const tree<Node> Tree) {
        // we initialize the list of equivalent vectors if it does not exist

        if (storage[storage_n].find(key) == storage[storage_n].end())
            storage[storage_n][key] = vector<tree<Node>>();
        
        // if it is smaller we add to the front, otherwise we add to the back.
        // This way we know the first element is the smallest one, and we don't care
        // about the order of the rest of the elements.
        auto& storage_it = storage[storage_n][key];

        // Insert Tree in order by size, smallest first
        auto it = std::find_if(storage_it.begin(), storage_it.end(),
            [&](const tree<Node>& t) {
            return Tree.begin().node->get_size() < t.begin().node->get_size();
            });
        storage_it.insert(it, Tree);
    }

    vector<tree<Node>> getList(const int& storage_n, const size_t& key) {
        auto it = storage[storage_n].find(key);
        if (it != storage[storage_n].end())
            return it->second;

        return {};
    }

    void clear() {
        int numPlanes = storage.size();
        storage.clear();
        storage.resize(numPlanes);
    }

    vector<size_t> keys(const int& storage_n) {
        vector<size_t> result;
        for (const auto& pair : storage[storage_n])
            result.push_back(pair.first);

        return result;
    }

private:
    // one storage instance for each plane
    vector<map<size_t, vector<tree<Node>>>> storage;
};

class Inexact_simplifier
{
    public:
        // static Inexact_simplifier* initSimplifier();
        void init(int hashSize, const Dataset &data, int numPlanes);
        
        // static void destroy();

        template<ProgramType P>
        Program<P> simplify_tree(Program<P>& program,
                                    const SearchSpace &ss, const Dataset &d)
        {
            // using RetType =
            //     typename // std::conditional_t<P == PT::Regressor, ArrayXf,
            //                 // std::conditional_t<P == PT::Representer, ArrayXXf, ArrayXf
            //     >>;
            

            Program<P> simplified_program(program);
            

            // prediction at the root already performs template cast and always returns a float
            auto original_predictions = simplified_program.predict(d);
            

            // iterate over the tree, trying to replace each node with a constant, and keeping the change if the pred does not change.
            TreeIter spot = simplified_program.Tree.begin_post();
            while(spot != simplified_program.Tree.end_post())
            {
                // we dont index or simplify fixed stuff.
                // non-wheightable nodes are not simplified. TODO: revisit this and see if they should (then implement it)
                // This is avoiding using booleans.
                if (spot.node->data.get_prob_change() > 0
                // &&  IsWeighable(spot.node->data.ret_type) && IsWeighable(spot.node->data.node_type)
                ) {
                    // indexing only small subtrees
                    if (simplified_program.size_at(spot, true) <= 15
                    &&  Isnt<NodeType::Constant, NodeType::MeanLabel, NodeType::Terminal>(spot.node->data.node_type)) {
                        index<P>(spot, d);
                    }
                    
                    // TODO: use IsLeaf here instead of checking for each possible nodetype. also search throughout the code and replace it
                    if (Isnt<NodeType::Constant, NodeType::MeanLabel, NodeType::Terminal>(spot.node->data.node_type)){

                        // res will return the closest within the threshold, so we dont have to check distance here
                        
                        auto res = query<P>(spot, d); // optional<pair<int, size_T>>

                        if (res){
                            // for each res we replace the subtree and pick the one with the smallest error.
                            // we know they will be smaller because query only returns smaller trees. We also include
                            // the current node in the list of candidates so the model does not get worse.
                            
                            float threshold = 1e-6;

                            float best_distance = threshold;
                            tree<Node> best_branch;
                            for (const auto& cand : res.value()) {
                                
                                const tree<Node> original_branch(spot);
                                const tree<Node> simplified_branch(cand);

                                // auto original_predictions = simplified_program.predict(d);
                                // auto spot_pred = spot.node->template predict<spot.node->data.ret_type>(d);
                                // using RetType = decltype(spot_pred);

                                simplified_program.Tree.erase_children(spot);

                                
                                spot = simplified_program.Tree.move_ontop(spot, simplified_branch.begin());
                                
                                auto new_predictions = simplified_program.predict(d);

                                float diff = (original_predictions.template cast<float>() - new_predictions.template cast<float>()).square().mean();
                                
                                
                                if (diff < best_distance) {
                                    
                                    best_distance = diff;
                                    best_branch = cand;
                                }

                                // rollback
                                simplified_program.Tree.erase_children(spot);
                                spot = simplified_program.Tree.move_ontop(spot, original_branch.begin());
                            }
                            if (best_distance < threshold) {

                                // cout << "replacing " << spot.node->get_model();
                                simplified_program.Tree.erase_children(spot);

                                const tree<Node> best_branch_copy(best_branch);
                                // cout << " with " << best_branch_copy.begin().node->get_model() << endl;
                                
                                spot = simplified_program.Tree.move_ontop(spot, best_branch_copy.begin());
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
                

        template<ProgramType P>
        void index(TreeIter& spot, const Dataset &d)
        {
            const tree<Node> tree_copy(spot);

            // cout << "indexing ...";
            // cout << tree_copy.begin().node->get_model(true) 
            //     << " with datatype " << static_cast<int>(tree_copy.begin().node->data.ret_type) << endl;
            
            
            auto hashes = hash<P>(spot, d);
            
            
            for (size_t i = 0; i < hashes.size(); ++i)
            {
                
                // hash() will clip the prediction to the inputDim, but here we store the full
                // predictions so we can calculate the distance to the query point later in query()
                equivalentExpressions[spot.node->data.ret_type].append(i, hashes[i], tree_copy);
                
            }
            
        }

    private:
        int inputDim = 1;

        // template<DataType D>
        // float compare_predictions_at_node(TreeIter spot,
        //                                 const tree<Node>& replacement_branch,
        //                                 const Dataset& d)
        // {
        //     using RetType = typename DataEnumType<D>::type;

        //     auto original_pred = spot.node->predict<RetType>(d, nullptr);
        //     auto replacement_pred = replacement_branch.begin().node->predict<RetType>(d, nullptr);

        //     return (original_pred.array() - replacement_pred.array()).square().mean();
        // }

        template<ProgramType P>
        vector<size_t> hash(TreeIter& spot, const Dataset &d)
        {
            // returns one hash for each plane
            
            using RetType =
                typename std::conditional_t<P == PT::Regressor, ArrayXf,
                            std::conditional_t<P == PT::Representer, ArrayXXf, ArrayXf
                >>;
            
            // we cast to float because hash and query are based on matrix multiplications,
            // but we will store the hash only on the corresponding storage instance
            ArrayXf floatClippedInput;

            if constexpr (P == PT::Representer) {
                
                ArrayXXf inputPoint = (*spot.node).template predict<ArrayXXf>(d);
                
                floatClippedInput = Eigen::Map<ArrayXf>(inputPoint.data(), inputPoint.size()).head(inputDim).template cast<float>();
                
            } else {
                
                if (spot.node->data.ret_type == DataType::ArrayB) {
                    
                    ArrayXb inputPointB = (*spot.node).template predict<ArrayXb>(d);
                    
                    floatClippedInput = inputPointB.template cast<float>();
                    
                }
                else if (spot.node->data.ret_type == DataType::ArrayI) {
                    
                    ArrayXi inputPointI = (*spot.node).template predict<ArrayXi>(d);
                    
                    floatClippedInput = inputPointI.template cast<float>();
                    
                }
                else {
                    
                    floatClippedInput = (*spot.node).template predict<ArrayXf>(d);
                    
                }
            }
            
            assert(floatClippedInput.size() >= inputDim && 
                "data must have at least inputDim elements");

            

            floatClippedInput = floatClippedInput.head(inputDim);

            

            vector<size_t> hashes;
            for (size_t planeIdx = 0; planeIdx < uniformPlanes.size(); ++planeIdx)
            {
                
                // TODO: handle nan predictions?

                const auto& plane = uniformPlanes[planeIdx];
                Eigen::ArrayXf projection = plane * floatClippedInput.matrix();
                Eigen::Array<bool, Eigen::Dynamic, 1> comparison = (projection > 0);

                size_t input_hash = 0;
                for (int i = 0; i < comparison.size(); ++i) {
                    input_hash <<= 1;
                    input_hash |= comparison(i) ? 1 : 0;
                }
                
                hashes.push_back(input_hash);
                
                
            }

            return hashes;
        }


        template<ProgramType P>
        optional<vector<tree<Node>>> query(TreeIter& spot, const Dataset &d)
        {
            // will return the hash and the distance to the queryPoint.        

            int spot_size = spot.node->get_size();

            // first argument is the index of the plane, second is the hash
            vector<tree<Node>> matches = {};

            vector<size_t> hashes = hash<P>(spot, d);

            for (int i = 0; i < hashes.size(); ++i){
                // cout << "querying hashes index " << i 
                //   << " with datatype " << static_cast<int>(spot.node->data.ret_type) << endl;

                vector<tree<Node>> newCandidates = equivalentExpressions[spot.node->data.ret_type].getList(i, hashes[i]);

                if (newCandidates.size() == 0)
                    continue;
                    int count = 0;
                    for (const auto& cand : newCandidates) {
                        if (cand.begin().node->get_size() < spot_size) {
                            matches.push_back(cand);
                            if (++count >= 10) break; // returning only top 10
                        } else {
                            // Since candidates are ordered by size, we can break early
                            break;
                        }
                    }
            }

            if (matches.size() > 0)
                return matches;
            return std::nullopt;
        }

        // one storage instance for each datatype/rettype.
        // the storage will be used to calculate the hash and query the
        // collection of hashes, returning the closest ones,
        // and the list will contain equivalent expressions, ordered by size
        // (or linear complexity). So we dont store pairs in the storage
        std::unordered_map<DataType, HashStorage> equivalentExpressions;

        vector<MatrixXf> uniformPlanes;
};

} // Simply
} // Brush

#endif
