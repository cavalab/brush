/* Brush
copyright 2022 William La Cava
license: GNU/GPL v3
*/

#ifndef OMOP_H
#define OMOP_H

// #include "../init.h"
#include <string>
#include <filesystem>
#include <fstream>
// #include "../../thirdparty/json.hpp"
#include "nlohmann/json.hpp"

namespace fs = std::filesystem;

namespace Brush::Data
{


enum class TimeValues {
    offset,
    delta,
    timestamp
};

enum class StringFeatures {
    categorical,
    onehot
};

struct OmopData
{
    std::string cdm_version;
    TimeValues tv = TimeValues::timestamp;
    StringFeatures sf = StringFeatures::categorical;

    /// Initialize OMOP Dataset from a directory of CSVs
    OmopData(fs::directory_iterator omop_dir);

    /// Initialize OMOP Dataset from a JSON file
    OmopData(fs::path json_filename);

    /// Initialize OMOP Dataset from a SQL database connection
    // TODO
};

}

#endif