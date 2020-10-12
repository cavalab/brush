#include <algorithm>
#include <iostream>
#include "template_nodes.h"
using namespace std;

/* template<typename T> */
/* T add(T x, T y){ return x + y; }; */

namespace BR{
template<typename T>
std::function<bool(T,T)> less = std::less<T>(); 

template<typename T>
std::function<T(T,T)> plus = std::plus<T>();

}
int main(int, char **)
{
    cout << "declaring node...\n";
    NodeBase<int, int, int> node_plus(BR::plus<int>);
    cout << "3+5 = " << node_plus.op(3, 5) << endl;
    cout << "3.1+5.7 = " << node_plus.op(3.1, 5.7) << endl;

    NodeBase<float, float, float> node_plus_float(BR::plus<float>);

    cout << "3+5 = " << node_plus_float.op(3, 5) << endl;
    cout << "3.1+5.7 = " << node_plus_float.op(3.1, 5.7) << endl;

    NodeBase<bool, float, float> node_lt(BR::less<float>);
    cout << "3<5 = " << node_lt.op(3, 5) << endl;
    cout << "3.1<5.7 = " << node_lt.op(3.1, 5.7) << endl;

    cout << "dune.\n";
}
