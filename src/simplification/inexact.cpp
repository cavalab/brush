#include "inexact.h"


// simplification maps are based on trainingdata
// should ignore fixed nodes ---> does not change subtrees if they contain fixed nodes 
// should I implement json serialization?
        
namespace Brush { namespace Simpl{
    
Inexact_simplifier* Inexact_simplifier::instance = NULL;
    
Inexact_simplifier::Inexact_simplifier()
{                   
}

Inexact_simplifier* Inexact_simplifier::initSimplifier()
{
    // creates the static random generator by calling the constructor
    if (!instance)
    {
        instance = new Inexact_simplifier();
    }

    return instance;
}

template<Brush::ProgramType PT>
Program<PT> Inexact_simplifier::simplify_tree(Program<PT>& program,
                            const SearchSpace &ss, const Dataset &d)
{
    Program<PT> simplified_program(program);

    // iterate over the tree, trying to replace each node with a constant, and keeping the change if the pred does not change.
    TreeIter spot = simplified_program.Tree.begin();
    while(spot != simplified_program.Tree.end())
    {
        // we dont index or simplify fixed stuff
        if (spot.node->data.get_prob_change() > 0) {
            // indexing only small subtrees or non-constant-terminal nodes
            if (simplified_program.size_at(spot) < 10
            ||  Isnt<NodeType::Constant, NodeType::MeanLabel>(spot.node->data.node_type)) {
                inexact_simplifier.index(spot, d);
            }

            if (Isnt<NodeType::Constant, NodeType::MeanLabel, NodeType::Terminal>(spot.node->data.node_type)){
                // res will return the closest within the threshold, so we dont have to check distance here
                auto res = inexact_simplifier.query(spot, d); // optional<pair<size_t, string>>

                if (res){
                    auto key = res.value(); // table index and hash
                    const tree<Node> branch(spot);
                        
                    if (inexact_simplifier.equivalentExpression.find(key) == inexact_simplifier.equivalentExpression.end()) {
                        inexact_simplifier.equivalentExpression[key] = branch;
                    } else if (spot.node->get_size(false) < inexact_simplifier.equivalentExpression[key].begin().node->get_size(false)){                
                            inexact_simplifier.equivalentExpression[key] = branch;
                    } else if (spot.node->get_size(false) > inexact_simplifier.equivalentExpression[key].begin().node->get_size(false)){                         
                        const tree<Node> simplified_branch(inexact_simplifier.equivalentExpression[key]);
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

void Inexact_simplifier::initUniformPlanes(int hashSize, int inputDim, int numPlanes)
{
    // TODO: inputDim cutoff at 100 datapoints?
    // int numPlanes = 1? The bigger the number of planes, the more accurate the hash, but the slower the search

    uniformPlanes.clear();
    
    // create random planes
    for (int i=0; i<numPlanes; ++i)
    {
        MatrixXf plane = MatrixXf::Random(hashSize, inputDim);
        // plane /= plane.norm();
        uniformPlanes.push_back(plane);
    }
}

vector<string> Inexact_simplifier::hash(const ArrayXf& inputPoint)
{
    vector<string> hashes;
    for (size_t planeIdx = 0; planeIdx < uniformPlanes.size(); ++planeIdx)
    {
        // TODO: handle nan predictions here and on the index functions
        // std::cout << "Processing plane " << planeIdx << std::endl;
        const auto& plane = uniformPlanes[planeIdx];
        ArrayXf projections = (plane * inputPoint.matrix());
        // std::cout << "Projections: " << projections.transpose() << std::endl;

        ArrayXb comparison = (projections.array() > 0);
        // std::cout << "Comparisons: " << comparison.transpose() << std::endl;

        string hashString = ""; // TODO: size_t instead of string
        // hashString.reserve(hashSize);
        
        for (bool v : comparison){
            // std::cout << v << ", ";
            hashString += v ? "1" : "0";
        }

        // std::cout << std::endl << "Generated hash string: " << hashString << std::endl;
        hashes.push_back(hashString);
    }
    // std::cout << "Returning hashes" << std::endl;
    return hashes;
}

void Inexact_simplifier::index(TreeIter& spot, const Dataset &d)
{
    // we cast to float because hash and query are based on matrix multiplications,
    // but we will store the hash only on the corresponding storage instance
    // std::cout << "Predicting node value" << std::endl;
    ArrayXf v_float = (*spot.node).template predict<ArrayXf>(d);

    // std::cout << "Hashing node value" << std::endl;
    auto hashes = hash(v_float);
    for (size_t i = 0; i < hashes.size(); ++i)
    {
        // std::cout << "Processing hash " << i << ": " << hashes[i] << std::endl;
        if (spot.node->data.ret_type==DataType::ArrayB) {
            // std::cout << "Appending to storageBool" << std::endl;
            storageBool.append(hashes[i], v_float);
        } else if (spot.node->data.ret_type==DataType::ArrayI) {
            // std::cout << "Appending to storageInt" << std::endl;
            storageInt.append(hashes[i], v_float);
        } else { // otherwise we store it as floats
            // std::cout << "Appending to storageFloat" << std::endl;
            storageFloat.append(hashes[i], v_float); // TODO: should throw an error
        }
    }
}

// will return the hash and the distance to the queryPoint
optional<pair<size_t, string>> Inexact_simplifier::query(TreeIter& spot, const Dataset &d)
{
    float threshold = 1e-8; // TODO: calculate threshold based on variance of dataset

    ArrayXf v_float = (*spot.node).template predict<ArrayXf>(d);

    vector<pair<size_t, string>> candidates;
    vector<float> distances;

    HashStorage *storage;
    if (spot.node->data.ret_type==DataType::ArrayB) {
        storage = (&storageBool);
        // std::cout << "Using storageBool" << std::endl;
    } else if (spot.node->data.ret_type==DataType::ArrayI) {
        storage = (&storageInt);
        // std::cout << "Using storageInt" << std::endl;
    } else { // otherwise we store it as floats
        storage = (&storageFloat); // TODO: should throw an error
        // std::cout << "Using storageFloat" << std::endl;
    }

    vector<string> hashes = hash(v_float);
    // std::cout << "Hashes: ";
    for (const auto& h : hashes) {
        // std::cout << h << " ";
    }
    // std::cout << std::endl;

    for (size_t i = 0; i < hashes.size(); ++i){
        auto newCandidates = storage->getList(hashes[i]);
        // std::cout << "Candidates for hash " << hashes[i] << ": " << newCandidates.size() << std::endl;
        
        for (const auto& cand : newCandidates) {
            float d = (v_float - cand).array().pow(2).mean();
            if (std::isnan(d) || std::isinf(d))
                d = MAX_FLT;

            // std::cout << "Distance: " << d << std::endl;

            if (d<threshold){
                candidates.push_back(make_pair(i, hashes[i]));
                distances.push_back(d);
                // std::cout << "Candidate added with distance: " << d << std::endl;
            }
        }
    }

    if (distances.size() > 0){
        auto min_idx = std::distance(std::begin(distances),
            std::min_element(std::begin(distances), std::end(distances)));
        // std::cout << "Minimum distance index: " << min_idx << std::endl;
        return candidates[min_idx];
    } else {
        // std::cout << "No candidates found within threshold" << std::endl;
    }

    return std::nullopt;
}

void Inexact_simplifier::destroy()
{
    if (instance)
        delete instance;
        
    instance = NULL;
}

Inexact_simplifier::~Inexact_simplifier() {}

} // Simply
} // Brush
