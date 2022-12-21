/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "io.h"
#include "../util/utils.h"
/* #include "rnd.h" */
#include <unordered_set>

namespace Brush::Data{
    
/// read csv file into Data. 
Dataset read_csv (
    const std::string& path,
    const std::string& target,
    char sep
)
{
    std::ifstream indata;
    indata.open(path);
    if (!indata.good())
        HANDLE_ERROR_THROW("Invalid input file " + path + "\n"); 
        
    string line;
    map<string,vector<float>> values;
    vector<float> targets;

    std::vector<string> names;
    unsigned rows=0, target_col_num = 0;
    
    while (std::getline(indata, line)) 
    {
        std::stringstream lineStream(line);
        std::string cell;
        
        unsigned col_num=0;   
        while (std::getline(lineStream, cell, sep)) 
        {
            cell = Util::trim(cell);
              
            if (rows==0) // read in header
            {
                if (!cell.compare(target))
                    target_col_num = col_num;                    
                else
                    names.push_back(cell);
            }
            else if (col_num != target_col_num) 
            {
                auto col_name = names.at(col_num);
                if (!values.contains(col_name))
                    values[col_name] = {};

                values.at(names.at(col_num)).push_back(std::stod(cell));
            }
            else
                targets.push_back(std::stod(cell));
            
            ++col_num;
        }
        ++rows;
    }
    
    auto y = ArrayXf::Map(targets.data(), targets.size());
    // for (int i = 0; i < targets.size(); ++i)
    //     y(i) = targets.at(i);

    // infer types of features
    map<string, State> features;
    for (auto& [key, value] : values) 
    {
        auto tmp = Map<ArrayXf>(value.data(), value.size());

        if (tmp.size() != y.size())
            HANDLE_ERROR_THROW("different numbers of samples in X and y");
        features[key] = check_type(tmp); 
        
    }

    // check if endpoint is binary
    bool binary_endpoint = (y.array() == 0 || y.array() == 1).all();

    auto result = Dataset(features,y,binary_endpoint);
    return result;
    
}

} // Brush

