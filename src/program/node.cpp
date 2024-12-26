#include "node.h"

namespace Brush {

ostream& operator<<(ostream& os, const NodeType& nt)
{
    os << "nt: " << nt << endl;
    os << NodeTypeName.at(nt);
    return os;
}

ostream& operator<<(ostream& os, const Node& n)
{
    os << n.get_name();
    return os;
}

/// @brief get the name of the node. 
/// @return name 
auto Node::get_name(bool include_weight) const noexcept -> std::string 
{

    if (Is<NodeType::Terminal>(node_type))
    {
        if (is_weighted && W != 1.0 && include_weight)
            return fmt::format("{:.2f}*{}",W,feature);
        else
            return feature;
    }
    else if (Is<NodeType::Constant>(node_type) && include_weight)
    {
        return fmt::format("{:.2f}", W);
    }
    else if (Is<NodeType::MeanLabel>(node_type))
    {
        // this will show (MeanLabel) in the terminal name
        // return fmt::format("{:.2f} ({})", W, feature);

        return fmt::format("{:.2f}", W);
    }
    else if (Is<NodeType::OffsetSum>(node_type)){
        // if (W != 1.0)
        return fmt::format("{:.2f}+Sum", W);

        // return fmt::format("Sum");
    }
    else if (is_weighted && include_weight)
        return fmt::format("{:.2f}*{}",W,name);

    return name;
}

string Node::get_model(const vector<string>& children) const noexcept
{
    if (children.empty())
        return get_name();
    else if (Is<NodeType::SplitBest>(node_type)){
        return fmt::format("If({}>{:.2f},{},{})",
            feature,
            W,
            children.at(0),
            children.at(1)
            );
    }
    else if (Is<NodeType::SplitOn>(node_type)){
        if (arg_types.at(0) == DataType::ArrayB)
        {
            // booleans dont use thresholds (they are used directly as mask in split)
            return fmt::format("If({},{},{})",
                children.at(0),
                children.at(1),
                children.at(2)
            );
        }
        // integers or floating points (they have a threshold)
        return fmt::format("If({}>{:.2f},{},{})",
            children.at(0),
            W,
            children.at(1),
            children.at(2)
            );
    }
    else if (Is<NodeType::OffsetSum>(node_type)){
        string args = "";

        // if (W != 1.0)
        args = fmt::format("{:.2f},", W);
    
        for (int i = 0; i < children.size(); ++i){
            args += children.at(i);
            if (i < children.size()-1)
                args += ",";
        }

        return fmt::format("Sum({})", args);
    }
    else{
        string args = "";
        for (int i = 0; i < children.size(); ++i){
            args += children.at(i);
            if (i < children.size()-1)
                args += ",";
        }

        return fmt::format("{}({})", get_name(), args);
    }

}

////////////////////////////////////////
// serialization
// serialization for Node
// using json = nlohmann::json;

void to_json(json& j, const Node& p) 
{
    j = json{
        {"name", p.name},
        {"center_op", p.center_op}, 
        {"fixed", p.fixed}, 
        {"prob_change", p.prob_change}, 
        {"is_weighted", p.is_weighted}, 
        {"W", p.W}, 
        {"node_type", p.node_type}, 
        {"sig_hash", p.sig_hash}, 
        {"sig_dual_hash", p.sig_dual_hash}, 
        {"ret_type", p.ret_type}, 
        {"arg_types", p.arg_types}, 
        {"feature", p.get_feature()} 
        // {"node_hash", p.get_node_hash()} 
    };
}

using NT = NodeType;
void init_node_with_default_signature(Node& node)
{
    // if (Is<
    //     NT::Add,
    //     NT::Mul,
    //     NT::Min,
    //     NT::Max
    //     >(nt)) 
    //     return Signature<ArrayXf(ArrayXf,ArrayXf)>{};
    NT n = node.node_type;
    if (Is<
        NT::Abs,
        NT::Acos,
        NT::Asin,
        NT::Atan,
        NT::Cos,
        NT::Cosh,
        NT::Sin,
        NT::Sinh,
        NT::Tan,
        NT::Tanh,
        NT::Ceil,
        NT::Floor,
        NT::Exp,
        NT::Log,
        NT::Logabs,
        NT::Log1p,
        NT::Sqrt,
        NT::Sqrtabs,
        NT::Square,
        NT::Logistic,
        NT::OffsetSum, // unary version
        NT::CustomUnaryOp
        >(n)) 
    {
        node.set_signature<Signature<ArrayXf(ArrayXf)>>();
    }
    else if (Is<
        NT::Add,
        NT::Sub,
        NT::Mul,
        NT::Div,
        NT::Pow,
        NT::SplitBest,
        NT::CustomSplit
        >(n))
    {
        node.set_signature<Signature<ArrayXf(ArrayXf,ArrayXf)>>();
    }
    else if (Is<
        NT::And,
        NT::Or
        >(n))
    {
        node.set_signature<Signature<ArrayXb(ArrayXb,ArrayXb)>>();
    }  
    // else if (Is<
    //     NT::Not
    //     >(n))
    // {
    //     node.set_signature<Signature<ArrayXb(ArrayXb)>>();
    // }  
    else if (Is<
        NT::Min,
        NT::Max,
        NT::Mean,
        NT::Median,
        NT::Sum,
        // NT::OffsetSum, // n-ary version
        NT::Prod,
        NT::Softmax
        >(n))
    {
        auto msg = fmt::format("Can't infer arguments for {} from json."
            " Please provide them.\n",n);
        HANDLE_ERROR_THROW(msg);
    }
    else if (Is<
        NT::SplitOn
        >(n))
    {
        node.set_signature<Signature<ArrayXf(ArrayXb,ArrayXf,ArrayXf)>>();
    }
    else{
        node.set_signature<Signature<ArrayXf()>>();
    }

}

void from_json(const json &j, Node& p)
{
    if (j.contains("node_type"))
        j.at("node_type").get_to(p.node_type);
    else
        HANDLE_ERROR_THROW("Node json must contain node_type");

    if (j.contains("name"))
        j.at("name").get_to(p.name);
    else        
        p.name = NodeTypeName[p.node_type];

    if (j.contains("center_op"))
        j.at("center_op").get_to(p.center_op);

    if (j.contains("feature"))
    {
        // j.at("feature").get_to(p.feature);
        p.set_feature(j.at("feature"));
    }
    if (j.contains("feature_type"))
    {
        p.set_feature_type(j.at("feature_type"));
    }

    // if node has a ret_type and arg_types, get them. if not we need to make 
    // a signature
    bool make_signature=false;

    if (j.contains("ret_type"))
        j.at("ret_type").get_to(p.ret_type);
    else
        make_signature=true;
    if (j.contains("arg_types"))
        j.at("arg_types").get_to(p.arg_types);
    else
        make_signature=true;
    if (j.contains("sig_hash"))
        j.at("sig_hash").get_to(p.sig_hash);
    else
        make_signature=true;
    if (j.contains("sig_dual_hash"))
        j.at("sig_dual_hash").get_to(p.sig_dual_hash);
    else
        make_signature=true;

    if (make_signature){
        p.is_weighted = false;
        init_node_with_default_signature(p);
    }
    p.init();
    
    // these 4 below needs to be set after init(), since it resets these values
    if (j.contains("fixed"))
    {
        j.at("fixed").get_to(p.fixed);
    }

    if (j.contains("is_weighted"))
    {
        j.at("is_weighted").get_to(p.is_weighted);
    }

    if (j.contains("prob_change"))
    {
        j.at("prob_change").get_to(p.prob_change);
    }
    
    if (j.contains("W"))
    {
        j.at("W").get_to(p.W);
    }

    json new_json = p;
}


}
