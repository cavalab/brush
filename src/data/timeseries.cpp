#include "timeseries.h"

namespace Brush {
std::ostream &operator<<( std::ostream& output, const Brush::data::TimeSeries<bool>& ts ) { 
    return output << ts.print();
};
std::ostream &operator<<( std::ostream& output, const Brush::data::TimeSeries<float>& ts ) { 
    return output << ts.print();
};
std::ostream &operator<<( std::ostream& output, const Brush::data::TimeSeries<int>& ts ) { 
    return output << ts.print();
};
}
