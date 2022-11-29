#include "timeseries.h"

namespace Brush {
std::ostream &operator<<( std::ostream& output, const Brush::Data::TimeSeries<bool>& ts ) { 
    return output << ts.print();
};
std::ostream &operator<<( std::ostream& output, const Brush::Data::TimeSeries<float>& ts ) { 
    return output << ts.print();
};
std::ostream &operator<<( std::ostream& output, const Brush::Data::TimeSeries<int>& ts ) { 
    return output << ts.print();
};
}
