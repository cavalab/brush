/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#include "io.h"
#include "../util/utils.h"
/* #include "rnd.h" */
#include <unordered_set>

namespace Brush::data{
    
/// read csv file into Data. 
Data read_csv (
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
    
    auto y = Map<VectorXf>(targets.data(), targets.size());

    // infer types of features
    map<string, State> features;
    for (auto& [key, value] : values) 
    {
        auto tmp = Map<ArrayXf>(value.data(), value.size());

        if (tmp.size() != y.size())
            HANDLE_ERROR_THROW("different numbers of samples in X and y");
        features[key] = check_type(tmp); 
        
    }
    // auto X = Map<MatrixXf>(values.data(), values.size()/(rows-1), rows-1);
    // auto XT = X.transpose();
    
    // if (X.cols() != names.size())
    // {
    //     string error_msg = "header missing or incorrect number of "
    //                        "feature names\n";
    //     error_msg += "X size: " + to_string(X.rows()) + "x" 
    //         + to_string(X.cols()) +"\n";
    //     error_msg += "feature names: ";
    //     for (auto fn: names)
    //         error_msg += fn + ",";
    //     HANDLE_ERROR_THROW(error_msg);
    // }
   
    // dtypes = find_dtypes(X);

    // string print_dtypes = "dtypes: "; 
    // for (unsigned i = 0; i < dtypes.size(); ++i) 
    //     print_dtypes += (names.at(i) + " (" + to_string(dtypes.at(i)) 
    //             + "), ");
    // print_dtypes += "\n";
    // cout << print_dtypes;

    // check if endpoint is binary
    bool binary_endpoint = (y.array() == 0 || y.array() == 1).all();

    auto result = Data(features,y,binary_endpoint);
    return result;
    
}

} // Brush

