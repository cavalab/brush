#include <algorithm>
#include <iostream>
#include "template_nodes.h"
#include "operators.h"
#include "program.h"
#include "data.h"

using namespace Eigen;
using namespace std;
using namespace BR;
using namespace BR::Dat;

/* template<typename T> */
/* T add(T x, T y){ return x + y; }; */
std::map<std::string, NodeBase* > NodeMap = {
    { "+", new Node<ArrayXf(ArrayXf,ArrayXf)>(Op::plus<ArrayXf>, "ADD") },
    { "-", new Node<ArrayXf(ArrayXf,ArrayXf)>(Op::minus<ArrayXf>, "MINUS") },
    { "<", new Node<ArrayXb(ArrayXf, ArrayXf)>(Op::lt<ArrayXf>, "LESS THAN") },
    { "*", new Node<ArrayXf(ArrayXf,ArrayXf)>(Op::multiplies<ArrayXf>, "TIMES") },
};

int main(int, char **)
{
    cout << "declaring node...\n";
    Node<int(int, int)> node_plus(Op::plus<int>, "ADD");
    cout << "3+5 = " << node_plus.op(3, 5) << endl;
    cout << "3.1+5.7 = " << node_plus.op(3.1, 5.7) << endl;

    Node<float(float, float)> node_plus_float(Op::plus<float>, "ADD");

    cout << "3+5 = " << node_plus_float.op(3, 5) << endl;
    cout << "3.1+5.7 = " << node_plus_float.op(3.1, 5.7) << endl;

    Node<bool(float, float)> node_lt(Op::less<float>, "LESS THAN");
    cout << "3<5 = " << node_lt.op(3, 5) << endl;
    cout << "3.1<5.7 = " << node_lt.op(3.1, 5.7) << endl;

    Node<float(float, float)> node_minus(Op::minus<float>, "MINUS");
    cout << "3-5 = " << node_minus.op(3, 5) << endl;
    cout << "3.1-5.7 = " << node_minus.op(3.1, 5.7) << endl;

    Node<float(float, float)> node_multiplies(Op::multiplies<float>, "TIMES");
    cout << "3*5 = " << node_multiplies.op(3, 5) << endl;
    cout << "3.1*5.7 = " << node_multiplies.op(3.1, 5.7) << endl;
    cout << "dune.\n";

    // construct tree
    // P = LESS ( ADD (X1, X2) , MULTIPLY( X1, X2) )
    //
    Program tree;
    Program::iterator top = tree.begin();
    auto root = tree.insert(tree.begin(), NodeMap["<"]);
    auto add = tree.append_child(root, NodeMap["+"]);
    auto times = tree.append_child(root, NodeMap["*"]);
    tree.append_child(add, new Node<ArrayXf>("x_1", 0));
    tree.append_child(add, new Node<ArrayXf>("x_2", 1));
    tree.append_child(times, new Node<ArrayXf>("x_1", 0));
    tree.append_child(times, new Node<ArrayXf>("x_2", 1));

    auto loc = tree.begin(); 

    while(loc!=tree.end()) 
    {
        for(int i=0; i<tree.depth(loc); ++i)
            cout << "--";
        cout << (*loc)->name << endl;
        ++loc;
        /* cout << endl; */
    }

    cout << "===\n";
    MatrixXf X(2,10);
    VectorXf y(10);
    X << 1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,
         2.0,1.0,6.0,4.0,5.0,8.0,7.0,5.0,9.0,10.0,
    y << 1.0,0.0,1.0,1.0,0.0,1.0,1.0,0.0,0.0,0.0;
    Longitudinal Z;
    Data d(X,y,Z);
    cout << "fitting tree...\n";
    State out = tree.fit(d);
    cout << "output: " << get<ArrayXb>(out) << endl;
}
