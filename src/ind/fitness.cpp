#include "fitness.h"

namespace Brush
{

void to_json(json &j, const Fitness &f)
{
    j = json{
        {"values",  f.values},
        {"weights", f.weights},
        {"wvalues", f.wvalues},
        {"loss", f.loss},
        {"loss_v", f.loss_v},
        {"complexity", f.complexity},
        {"linear_complexity", f.linear_complexity},
        {"size", f.size},
        {"depth", f.depth},
        {"dcounter", f.dcounter},
        {"dominated", f.dominated},
        {"rank", f.rank},
        {"crowding_dist", f.crowding_dist}
    };
}

void from_json(const json &j, Fitness& f)
{
    j.at("values").get_to(  f.values );
    j.at("weights").get_to( f.weights );
    j.at("wvalues").get_to( f.wvalues );
    j.at("loss").get_to( f.loss );
    j.at("loss_v").get_to( f.loss_v );
    j.at("complexity").get_to( f.complexity );
    j.at("linear_complexity").get_to( f.linear_complexity );
    j.at("size").get_to( f.size );
    j.at("depth").get_to( f.depth );
    j.at("dcounter").get_to( f.dcounter );
    j.at("dominated").get_to( f.dominated );
    j.at("rank").get_to( f.rank );
    j.at("crowding_dist").get_to( f.crowding_dist );
}


int Fitness::dominates(const Fitness& b) const
{
    int flag1 = 0, // to check if this has a better objective
        flag2 = 0; // to check if b    has a better objective

    // TODO: replace comparison of individual values by using the overloaded operators (here and in nsga2)
    for (int i=0; i<get_wvalues().size(); ++i) {
        if (get_wvalues().at(i) > b.get_wvalues().at(i)
        ||  std::isnan(b.get_wvalues().at(i)) 
        ) 
            flag1 = 1;
        if (get_wvalues().at(i) < b.get_wvalues().at(i)
        ||  std::isnan(get_wvalues().at(i))
        ) 
            flag2 = 1;                       
    }

    // the proper way of comparing weighted values is considering everything as a MAXIMIIZATIION problem
    // (this is like deap does, and our fitness is inspired by them)
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

} // Brush