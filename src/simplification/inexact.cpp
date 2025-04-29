#include "inexact.h"


// simplification maps are based on trainingdata
// should ignore fixed nodes ---> does not change subtrees if they contain fixed nodes 
// should I implement json serialization?
        
namespace Brush { namespace Simpl{
    
// Inexact_simplifier* Inexact_simplifier::instance = NULL;
    
Inexact_simplifier::Inexact_simplifier()
{                   
}

// Inexact_simplifier* Inexact_simplifier::initSimplifier()
// {
//     // creates the static random generator by calling the constructor
//     if (!instance)
//     {
//         instance = new Inexact_simplifier();
//     }

//     return instance;
// }


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

        const auto& plane = uniformPlanes[planeIdx];
        ArrayXf projections = (plane * inputPoint.matrix());


        ArrayXb comparison = (projections.array() > 0);


        string hashString = ""; // TODO: size_t instead of string
        // hashString.reserve(hashSize);
        
        for (bool v : comparison){

            hashString += v ? "1" : "0";
        }


        hashes.push_back(hashString);
    }

    return hashes;
}

void Inexact_simplifier::index(TreeIter& spot, const Dataset &d)
{
    // we cast to float because hash and query are based on matrix multiplications,
    // but we will store the hash only on the corresponding storage instance

    ArrayXf v_float; // = (*spot.node).template predict<ArrayXf>(d);
    if (spot.node->data.ret_type==DataType::ArrayB) { // TODO: make this function templated?
        auto temp = (*spot.node).predict<ArrayXb>(d);
        v_float = temp.template cast<float>();
    }
    else if (spot.node->data.ret_type==DataType::ArrayI) {
        auto temp = (*spot.node).predict<ArrayXi>(d);
        v_float = temp.template cast<float>();
    } else { // otherwise we store it as floats
        v_float = (*spot.node).template predict<ArrayXf>(d);
    }


    auto hashes = hash(v_float);
    for (size_t i = 0; i < hashes.size(); ++i)
    {

        if (spot.node->data.ret_type==DataType::ArrayB) {

            storageBool.append(hashes[i], v_float);
        } else if (spot.node->data.ret_type==DataType::ArrayI) {

            storageInt.append(hashes[i], v_float);
        } else { // otherwise we store it as floats

            storageFloat.append(hashes[i], v_float); // TODO: should throw an error
        }
    }
}

// will return the hash and the distance to the queryPoint
optional<pair<size_t, string>> Inexact_simplifier::query(TreeIter& spot, const Dataset &d)
{
    float threshold = 1e-8; // TODO: calculate threshold based on variance of dataset

    // TODO: this block below filling v_float is repeated in the function above. Maybe I should implement it in a separate function?
    ArrayXf v_float; // = (*spot.node).template predict<ArrayXf>(d);
    if (spot.node->data.ret_type==DataType::ArrayB) { // TODO: make this function templated?
        auto temp = (*spot.node).predict<ArrayXb>(d);
        v_float = temp.template cast<float>();
    }
    else if (spot.node->data.ret_type==DataType::ArrayI) {
        auto temp = (*spot.node).predict<ArrayXi>(d);
        v_float = temp.template cast<float>();
    } else { // otherwise we store it as floats
        v_float = (*spot.node).template predict<ArrayXf>(d);
    }

    vector<pair<size_t, string>> candidates;
    vector<float> distances;

    HashStorage *storage;
    if (spot.node->data.ret_type==DataType::ArrayB) {
        storage = (&storageBool);

    } else if (spot.node->data.ret_type==DataType::ArrayI) {
        storage = (&storageInt);

    } else { // otherwise we store it as floats
        storage = (&storageFloat); 

    }
    // TODO: should throw an error if no storage matches

    vector<string> hashes = hash(v_float);

    for (const auto& h : hashes) {

    }


    for (size_t i = 0; i < hashes.size(); ++i){
        auto newCandidates = storage->getList(hashes[i]);

        
        for (const auto& cand : newCandidates) {
            float d = (v_float - cand).array().pow(2).mean();
            if (std::isnan(d) || std::isinf(d))
                d = MAX_FLT;



            if (d<threshold){
                candidates.push_back(make_pair(i, hashes[i]));
                distances.push_back(d);

            }
        }
    }

    if (distances.size() > 0){
        auto min_idx = std::distance(std::begin(distances),
            std::min_element(std::begin(distances), std::end(distances)));

        return candidates[min_idx];
    } else {

    }

    return std::nullopt;
}

// void Inexact_simplifier::destroy()
// {
//     if (instance)
//         delete instance;
        
//     instance = NULL;
// }

Inexact_simplifier::~Inexact_simplifier() {}

} // Simply
} // Brush
