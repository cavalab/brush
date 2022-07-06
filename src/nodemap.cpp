#include "nodemap.h"

namespace Brush{

std::map<std::string, NodeType> NodeNameType = {
    //arithmetic
    {"Add", NodeType::Add},
    {"Sub", NodeType::Sub},
    {"Mul", NodeType::Mul},
    {"Div", NodeType::Div},
    {"Aq", NodeType::Aq},
    {"Abs", NodeType::Abs},

    {"Acos", NodeType::Acos},
    {"Asin", NodeType::Asin},
    {"Atan", NodeType::Atan},
    {"Cos", NodeType::Cos},
    {"Cosh", NodeType::Cosh},
    {"Sin", NodeType::Sin},
    {"Sinh", NodeType::Sinh},
    {"Tan", NodeType::Tan},
    {"Tanh", NodeType::Tanh},
    {"Cbrt", NodeType::Cbrt},
    {"Ceil", NodeType::Ceil},
    {"Floor", NodeType::Floor},
    {"Exp", NodeType::Exp},
    {"Log", NodeType::Log},
    {"Logabs", NodeType::Logabs},
    {"Log1p", NodeType::Log1p},
    {"Sqrt", NodeType::Sqrt},
    {"Sqrtabs", NodeType::Sqrtabs},
    {"Square", NodeType::Square},
    {"Pow", NodeType::Pow},

    // logic; not sure these will make it in
    {"And", NodeType::And},
    {"Or", NodeType::Or},
    {"Not", NodeType::Not},
    {"Xor", NodeType::Xor},

    // decision (same)
    {"Equals", NodeType::Equals},
    {"LessThan", NodeType::LessThan},
    {"GreaterThan", NodeType::GreaterThan},
    {"Leq", NodeType::Leq},
    {"Geq", NodeType::Geq},

    // summary stats
    {"Min", NodeType::Min},
    {"Max", NodeType::Max},
    {"Mean", NodeType::Mean},
    {"Median", NodeType::Median},
    {"Count", NodeType::Count},
    {"Sum", NodeType::Sum},

    // timing masks
    {"Before", NodeType::Before},
    {"After", NodeType::After},
    {"During", NodeType::During},

    //split
    {"SplitBest", NodeType::SplitBest},
    {"SplitOn", NodeType::SplitOn},

    // leaves
    {"Constant", NodeType::Constant},
    {"Terminal", NodeType::Terminal},

    // custom
    {"CustomOp", NodeType::CustomOp},
    {"CustomSplit", NodeType::CustomSplit},
};

std::map<NodeType,std::string> NodeTypeName = Util::reverse_map(NodeNameType);

json BinaryFFtoF  = {
    { "ArrayF", {DataType::ArrayF, DataType::ArrayF }},
    { "MatrixF", {DataType::MatrixF,DataType::MatrixF }},
    { "TimeSeriesF", {DataType::TimeSeriesF,DataType::TimeSeriesF }},
};

json UnaryFtoF  = {
    { "ArrayF", {DataType::ArrayF }},
    { "MatrixF", {DataType::MatrixF }},
    { "TimeSeriesF", {DataType::TimeSeriesF }},
};

json NodeSchema = {
//arithmetic
    {"Add", { 
                {"ExecType", ExecType::Binary}, 
                {"Signature", BinaryFFtoF}, 
            },
    },
    {"Abs", { 
                {"ExecType", ExecType::Unary}, 
                {"Signature", UnaryFtoF}, 
            },
    },
};
}
