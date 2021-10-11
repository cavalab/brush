/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/

//internal includes
#include "data.h"

using namespace Brush::Util;
using std::min;

namespace Brush{ 
namespace data{

std::vector<std::type_index> StateTypeMap = {
                      typeid(ArrayXb),
                      typeid(ArrayXi), 
                      typeid(ArrayXf), 
                      typeid(ArrayXXb),
                      typeid(ArrayXXi), 
                      typeid(ArrayXXf), 
                      typeid(TimeSeries)
};
// /// returns the typeid held in arg
std::type_index StateType(const State& arg)
{
    return StateTypeMap.at(arg.index());
}
State check_type(const ArrayXf& x)
{
    // get feature types (binary or continuous/categorical)
    bool isBinary = true;
    bool isCategorical = true;
    std::map<float, bool> uniqueMap;
    for(int i = 0; i < x.size(); i++)
    {
        
        if(x(i) != 0 && x(i) != 1)
            isBinary = false;
        if(x(i) != floor(x(i)) && x(i) != ceil(x(i)))
            isCategorical = false;
        else
            uniqueMap[x(i)] = true;
    } 
    
    State tmp; // = x;

    if (isBinary)
    {
        tmp = ArrayXb(x.cast<bool>());
    }
    else
    {
        if(isCategorical && uniqueMap.size() < 10)
        {
            tmp = ArrayXi(x.cast<int>());
        }
        else
        {
            tmp = x;
        }
    }
    return tmp;

}

/// return a slice of the data using indices idx
Data Data::operator()(const vector<size_t>& idx) const
{
    std::map<std::string, State> new_d;
    for (auto& [key, value] : this->features) 
    {
        std::visit([&](auto&& arg) 
        {
            using T = std::decay_t<decltype(arg)>;
            if constexpr (std::is_same_v<T, ArrayXb> 
                          || std::is_same_v<T, ArrayXi> 
                          || std::is_same_v<T, ArrayXf> 
                          || std::is_same_v<T, TimeSeries> 
                         )
                new_d[key] = T(arg(idx));
            else if constexpr (std::is_same_v<T, ArrayXXb> 
                               || std::is_same_v<T, ArrayXXi> 
                               || std::is_same_v<T, ArrayXXf> 
                              )
                new_d[key] = T(arg(idx, Eigen::all));
            else 
                static_assert(always_false_v<T>, "non-exhaustive visitor!");
        },
        value
        );
    }
    auto new_y = this->y(idx);
    return Data(new_d, new_y, this->classification);
}

Data Data::get_batch(int batch_size) const
{

    batch_size =  std::min(batch_size,int(this->n_samples));
    return (*this)(r.shuffled_index(n_samples));
}

array<Data, 2> Data::split(const ArrayXb& mask) const
{
    // split data into two based on mask. 
    auto idx1 = Util::mask_to_index(mask);
    auto idx2 = Util::mask_to_index((!mask));
    return std::array<Data, 2>{ (*this)(idx1), (*this)(idx2) };
}

    
} // data
} // Brush
