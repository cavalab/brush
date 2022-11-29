/* Brush
copyright 2022 William La Cava
license: GNU/GPL v3
*/
#include "omop.h"


namespace Brush::Data {


OmopData::OmopData(fs::directory_iterator omop_dir) {
    for (fs::directory_entry dir_entry : omop_dir) {
        if (dir_entry.is_regular_file()) {

        }
    }
};

OmopData::OmopData(fs::path json_filename) {

};

}