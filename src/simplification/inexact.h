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
            //     typename std::conditional_t<P == PT::Regressor, ArrayXf,
            //                 std::conditional_t<P == PT::Representer, ArrayXXf, ArrayXf
            //     >>;
            // std::cout << "[DEBUG] inside simplify" << std::endl;

            Program<P> simplified_program(program);
            // std::cout << "[DEBUG] created copy" << std::endl;

            // prediction at the root already performs template cast and always returns a float
            auto original_predictions = simplified_program.predict(d);
            // std::cout << "[DEBUG] initial predict" << std::endl;

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
                    // indexing only small subtrees
                    if (simplified_program.size_at(spot, true) <= 10
                    &&  Isnt<NodeType::Constant, NodeType::MeanLabel, NodeType::Terminal>(spot.node->data.node_type)) {
                        index<P>(spot, d);
                    }
                    
                    // TODO: use IsLeaf here instead of checking for each possible nodetype. also search throughout the code and replace it
                    if (Isnt<NodeType::Constant, NodeType::MeanLabel, NodeType::Terminal>(spot.node->data.node_type)){

                        // res will return the closest within the threshold, so we dont have to check distance here
                        // std::cout << "[DEBUG] Querying replacements..." << std::endl;
                        auto res = query<P>(spot, d); // optional<pair<int, size_T>>

                        if (res){
                            // for each res we replace the subtree and pick the one with the smallest error.
                            // we know they will be smaller because query only returns smaller trees. We also include
                            // the current node in the list of candidates so the model does not get worse.
                            
                            for (const auto& cand : res.value()) {
                                // std::cout << "[DEBUG] Trying candidate for simplification..." << std::endl;
                                const tree<Node> original_branch(spot);

                                const tree<Node> simplified_branch(cand);
                                simplified_program.Tree.erase_children(spot);

                                // std::cout << "[DEBUG] Moved candidate onto spot, predicting..." << std::endl;
                                spot = simplified_program.Tree.move_ontop(spot, simplified_branch.begin());
                                
                                auto new_predictions = simplified_program.predict(d);

                                float diff = (original_predictions - new_predictions).square().mean();
                                // std::cout << "[DEBUG] Prediction diff: " << diff << std::endl;

                                if (diff > 1e-4){
                                    // std::cout << "[DEBUG] Diff too large, rolling back..." << std::endl;
                                    // rollback
                                    simplified_program.Tree.erase_children(spot);
                                    spot = simplified_program.Tree.move_ontop(spot, original_branch.begin());
                                }
                                else {
                                    // std::cout << "[DEBUG] Simplification accepted." << std::endl;
                                    // we have changed the tree, so we need to update the predictions
                                    original_predictions = new_predictions;
                                }
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
            
            // std::cout << "[DEBUG] Entering index()" << std::endl;
            auto hashes = hash<P>(spot, d);
            // std::cout << "[DEBUG] After hash()" << std::endl;
            
            for (size_t i = 0; i < hashes.size(); ++i)
            {
                // std::cout << "[DEBUG] In index() loop" << std::endl;
                // hash() will clip the prediction to the inputDim, but here we store the full
                // predictions so we can calculate the distance to the query point later in query()
                equivalentExpressions[spot.node->data.ret_type].append(i, hashes[i], tree_copy);
                // std::cout << "[DEBUG] After append()" << std::endl;
            }
            // std::cout << "[DEBUG] Exiting index()" << std::endl;
        }

    private:
        int inputDim = 1;

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
            
            // std::cout << "[DEBUG] Entering hash()" << std::endl;

            if constexpr (P == PT::Representer) {
                // std::cout << "[DEBUG] ProgramType is Representer." << std::endl;
                ArrayXXf inputPoint = (*spot.node).template predict<ArrayXXf>(d);
                // std::cout << "[DEBUG] After predict<ArrayXXf>" << std::endl;
                floatClippedInput = Eigen::Map<ArrayXf>(inputPoint.data(), inputPoint.size()).head(inputDim).template cast<float>();
                // std::cout << "[DEBUG] After mapping and casting ArrayXXf" << std::endl;
            } else {
                // std::cout << "[DEBUG] ProgramType is not Representer." << std::endl;
                if (spot.node->data.ret_type == DataType::ArrayB) {
                    // std::cout << "[DEBUG] ret_type is ArrayB." << std::endl;
                    ArrayXb inputPointB = (*spot.node).template predict<ArrayXb>(d);
                    // std::cout << "[DEBUG] After predict<ArrayXb>" << std::endl;
                    floatClippedInput = inputPointB.template cast<float>();
                    // std::cout << "[DEBUG] After casting ArrayXb" << std::endl;
                }
                else if (spot.node->data.ret_type == DataType::ArrayI) {
                    // std::cout << "[DEBUG] ret_type is ArrayI." << std::endl;
                    ArrayXi inputPointI = (*spot.node).template predict<ArrayXi>(d);
                    // std::cout << "[DEBUG] After predict<ArrayXi>" << std::endl;
                    floatClippedInput = inputPointI.template cast<float>();
                    // std::cout << "[DEBUG] After casting ArrayXi" << std::endl;
                }
                else {
                    // std::cout << "[DEBUG] ret_type is default (ArrayXf)." << std::endl;
                    floatClippedInput = (*spot.node).template predict<ArrayXf>(d);
                    // std::cout << "[DEBUG] After predict<ArrayXf>" << std::endl;
                }
            }
            // std::cout << "[DEBUG] After preparing floatClippedInput" << std::endl;
            
            // std::cout << "[DEBUG] Starting hash computation..." << std::endl;

            assert(floatClippedInput.size() >= inputDim && 
                "data must have at least inputDim elements");

            // std::cout << "[DEBUG] After assert, preparing input..." << std::endl;

            floatClippedInput = floatClippedInput.head(inputDim);

            // std::cout << "[DEBUG] Prepared floatClippedInput, entering hash loop..." << std::endl;

            vector<size_t> hashes;
            for (size_t planeIdx = 0; planeIdx < uniformPlanes.size(); ++planeIdx)
            {
                // std::cout << "[DEBUG] In hash loop..." << std::endl;
                // TODO: handle nan predictions?

                const auto& plane = uniformPlanes[planeIdx];
                Eigen::ArrayXf projection = plane * floatClippedInput.matrix();
                Eigen::Array<bool, Eigen::Dynamic, 1> comparison = (projection > 0);

                size_t input_hash = 0;
                for (int i = 0; i < comparison.size(); ++i) {
                    input_hash <<= 1;
                    input_hash |= comparison(i) ? 1 : 0;
                }
                // std::cout << "[DEBUG] Hash for plane " << planeIdx << ": " << input_hash << endl;
                hashes.push_back(input_hash);
                // std::cout << "[DEBUG] pushing..." << endl;
                
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
