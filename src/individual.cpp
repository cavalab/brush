#include "individual.h"

namespace Brush{   
namespace Pop{
        
template<ProgramType T>
int Individual<T>::check_dominance(const Individual<T>& b) const
{
    int flag1 = 0, // to check if this has a smaller objective
        flag2 = 0; // to check if b    has a smaller objective

    for (int i=0; i<obj.size(); ++i) {
        if (obj.at(i) < b.obj.at(i)) 
            flag1 = 1;
        else if (obj.at(i) > b.obj.at(i)) 
            flag2 = 1;                       
    }

    if (flag1==1 && flag2==0)   
        // there is at least one smaller objective for this and none 
        // for b
        return 1;               
    else if (flag1==0 && flag2==1) 
        // there is at least one smaller objective for b and none 
        // for this
        return -1;
    else             
        // no smaller objective or both have one smaller
        return 0;
}

template<ProgramType T>
void Individual<T>::set_obj(const vector<string>& objectives)
{
    obj.clear();
    
    for (const auto& n : objectives)
    {
        if (n.compare("fitness")==0)
            obj.push_back(fitness); // fitness on training data, not validation.
                                    // if you use batch, this value will change every generation
        else if (n.compare("complexity")==0)
            obj.push_back(set_complexity());
        else if (n.compare("size")==0)
            obj.push_back(program.size());
        else
            HANDLE_ERROR_THROW(n+" is not a known objective");
    }

}

} // Pop
} // Brush