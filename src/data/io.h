/* FEAT
copyright 2017 William La Cava
license: GNU/GPL v3
*/

#ifndef IO_H
#define IO_H

#include <Eigen/Dense>
#include <vector>
#include <fstream>
#include <chrono>
#include <ostream>
#include <map>
#include "../init.h"
#include "../util/error.h"
#include "data.h"

using namespace Eigen;

namespace Brush::data
{

///  read csv file into Data. 
Data read_csv (
    const std::string& path,
    const std::string& target,
    char sep=','
);

// ///  load longitudinal csv file into matrix. 
// void load_longitudinal(const std::string & path,
//                         std::map<string, std::pair<vector<ArrayXf>, vector<ArrayXf> > > &Z,
//                         char sep=',');

// /// load partial longitudinal csv file into matrix according to idx vector
// void load_partial_longitudinal(const std::string & path,
//                         std::map<string, std::pair<vector<ArrayXf>, vector<ArrayXf> > > &Z,
//                         char sep, const vector<int>& idx);
// }
}

#endif
