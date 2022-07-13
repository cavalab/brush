#include "tree_node.h"

namespace Brush {

DispatchTable<
              ArrayXf
              /* ArrayXb, */
              /* ArrayXi, */ 
              /* ArrayXf, */ 
              /* ArrayXXb, */
              /* ArrayXXi, */ 
              /* ArrayXXf, */ 
              /* TimeSeriesb, */
              /* TimeSeriesi, */
              /* TimeSeriesf */
             > dtable;
        /* template<typename T> */
        /* auto tree_node_<Node>::eval(const Data& d) */
        /* { */ 
        /*     auto F = DTable.TryGet<T>(n); */
        /*     return F(d, n); */
        /* }; */
        /* template<typename T> */
        /* auto tree_node_<Node>::fit(const Data& d){ State s; return std::get<T>(s);}; */
        /* template<typename T> */
        /* auto tree_node_<Node>::predict(const Data& d){ State s; return std::get<T>(s);}; */
}
