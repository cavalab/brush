#include <algorithm>
#include <iostream>
#include "template_nodes.h"
using namespace std;

/* template<typename T> */
/* T add(T x, T y){ return x + y; }; */

namespace Op{
template<typename T>
std::function<bool(T,T)> less = std::less<T>(); 

template<typename T>
std::function<T(T,T)> plus = std::plus<T>();

template<typename T>
std::function<T(T,T)> minus = std::minus<T>();

template<typename T>
std::function<T(T,T)> multiplies = std::multiplies<T>();
}

int main(int, char **)
{
    cout << "declaring node...\n";
    Node<int(int, int)> node_plus(Op::plus<int>);
    cout << "3+5 = " << node_plus.op(3, 5) << endl;
    cout << "3.1+5.7 = " << node_plus.op(3.1, 5.7) << endl;

    Node<float(float, float)> node_plus_float(Op::plus<float>);

    cout << "3+5 = " << node_plus_float.op(3, 5) << endl;
    cout << "3.1+5.7 = " << node_plus_float.op(3.1, 5.7) << endl;

    Node<bool(float, float)> node_lt(Op::less<float>);
    cout << "3<5 = " << node_lt.op(3, 5) << endl;
    cout << "3.1<5.7 = " << node_lt.op(3.1, 5.7) << endl;

    Node<float(float, float)> node_minus(Op::minus<float>);
    cout << "3-5 = " << node_minus.op(3, 5) << endl;
    cout << "3.1-5.7 = " << node_minus.op(3.1, 5.7) << endl;

    Node<float(float, float)> node_multiplies(Op::multiplies<float>);
    cout << "3*5 = " << node_multiplies.op(3, 5) << endl;
    cout << "3.1*5.7 = " << node_multiplies.op(3.1, 5.7) << endl;
    cout << "dune.\n";
}
