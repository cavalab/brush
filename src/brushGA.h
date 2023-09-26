/* Brush
copyright 2020 William La Cava
license: GNU/GPL v3
*/

#ifndef BrushGA_H
#define BrushGA_H

#include "init.h"
#include "taskflow/taskflow.hpp"

// TODO: improve the includes (why does this lines below does not work?)
// #include "variation.h"
// #include "selection.h"

// using namespace selection;
// using namespace variation;

namespace Brush
{

class BrushGA{
public:

    BrushGA(){}
    /// destructor             
    ~BrushGA(){} 
    
    void init();

    //getters and setters for GA configuration.
    // getters and setters for the best solution found after evolution
    // predict, transform, predict_proba, etc.
    // get statistics
    // load and save best individuals
    // logger, save to file
    // execution archive
    // random state control
    // score functions
    // fit methods (this will run the evolution), run a single generation 
private:
    // attributes (hyperparameters)
    // update best
    // calculate/print stats
};

int main(){
  
  tf::Executor executor;
  tf::Taskflow taskflow;

  auto [A, B, C, D] = taskflow.emplace(  // create four tasks
    [] () { std::cout << "TaskA\n"; },
    [] () { std::cout << "TaskB\n"; },
    [] () { std::cout << "TaskC\n"; },
    [] () { std::cout << "TaskD\n"; } 
  );                                  
                                      
  A.precede(B, C);  // A runs before B and C
  D.succeed(B, C);  // D runs after  B and C
                                      
  executor.run(taskflow).wait(); 

  return 0;
}

} // Brush

#endif
