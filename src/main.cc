#include <algorithm>
#include <string>
#include <iostream>
/* #include "tree.h" */
/* #include "state.h" */
/* #include "data.h" */
#include "nodes/node.h"
/* #include "nodes/nodemap.h" */
/* #include "program.h" */

using namespace std;
using namespace BR;

int main(int, char **)
{
    
    cout << "oh hey\n";
    Program tr;
    Program::iterator top, one, two, loc, sum, times;

    top=tr.begin();
    one=tr.insert(top, new NodeAdd());
    /* cout << "top: " << (*top)->name << endl; */ 

    cout << "one: " << (*one)->name << endl; 
    two=tr.append_child(one, new NodeVariable(10));
    sum = tr.append_child(one, new NodeSum());
    tr.append_child(sum, new NodeVariable(7));
    tr.append_child(sum, new NodeVariable(3));
    tr.append_child(sum, new NodeVariable(2));
    tr.append_child(sum, new NodeVariable(-1));
    times = tr.append_child(sum, new NodeTimes());
    tr.append_child(times, new NodeVariable(5));
    tr.append_child(times, new NodeVariable(4));

    loc = tr.begin(); 

    while(loc!=tr.end()) 
    {
        for(int i=0; i<tr.depth(loc); ++i)
            cout << " ";
        cout << (*loc)->name << endl;
        ++loc;
        /* cout << endl; */
    }
    cout << "===\n";
    Program::pre_order_iterator poi = tr.begin();
    MatrixXf X(2,10);
    VectorXf y(10);
    X << 1.0,2.0,3.0,4.0,5.0,6.0,7.0,8.0,9.0,10.0,
         2.0,1.0,6.0,4.0,5.0,8.0,7.0,5.0,9.0,10.0,
    y << 1.0,0.0,1.0,1.0,0.0,1.0,1.0,0.0,0.0,0.0;
    BR::Dat::Longitudinal Z;
    Data d(X,y,Z);
    State out = poi.node->eval(d);
    cout << "output: " << out.get<float>().get() << endl;
    auto node_plus = Node<std::plus<float>()>;
}
