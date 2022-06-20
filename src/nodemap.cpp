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
/* auto BinaryFF  = { */
/*     { DataType::ArrayF, tuple<Eigen::ArrayXf, Eigen::ArrayXf>() }, */
/*     { DataType::MatrixF, tuple<Eigen::ArrayXXf,Eigen::ArrayXXf>() }, */
/*     { DataType::TimeSeriesF, tuple<TimeSeriesf,TimeSeriesf>() }, */
/* }; */
/* auto BinaryFF  = { */
/*     { DataType::ArrayF, tuple<ArrayF,ArrayF>() }, */
/*     { DataType::MatrixF, tuple<MatrixF,MatrixF>() }, */
/*     { DataType::TimeSeriesF, tuple<TimeSeriesF,TimeSeriesF>() }, */
/* }; */

/* auto BinaryBB  = { */
/*     { DataType::ArrayB, tuple<Eigen::ArrayXb,Eigen::ArrayXb>() }, */
/*     { DataType::MatrixB, tuple<ArrayXXb,ArrayXXb>() }, */
/*     { DataType::TimeSeriesB, tuple<TimeSeriesb,TimeSeriesb>() }, */
/* }; */

/* auto BinaryBF  = { */
/*     { DataType::ArrayB, tuple<Eigen::ArrayXf, Eigen::ArrayXf>() }, */
/*     { DataType::MatrixB, tuple<Eigen::ArrayXXf,Eigen::ArrayXXf>() }, */
/*     { DataType::ArrayB, tuple<TimeSeriesf,TimeSeriesf>() }, */
/* }; */

/* auto UnaryFF  = { */
/*     { DataType::ArrayF, tuple<Eigen::ArrayXf>() }, */
/*     { DataType::MatrixF, tuple<Eigen::ArrayXXf>() }, */
/*     { DataType::TimeSeriesF, tuple<TimeSeriesXf>() }, */
/* }; */
/* auto UnaryBB  = { */
/*     { DataType::ArrayB, tuple<ArrayXb>() }, */
/*     { DataType::MatrixB, tuple<Eigen::ArrayXXb>() }, */
/*     { DataType::TimeSeriesB, tuple<TimeSeriesb>() }, */
/* }; */

/* auto ReduceFF  = { */
/*     { DataType::ArrayF, tuple<Eigen::ArrayXXf>() }, */
/*     { DataType::ArrayF, tuple<TimeSeriesf>() }, */
/* }; */
/* auto ReduceIA  = { */
/*     { DataType::ArrayI, tuple<Eigen::ArrayXXf>() }, */
/*     { DataType::ArrayI, tuple<MatrixI>() }, */
/*     { DataType::ArrayI, tuple<TimeSeriesI>() }, */
/*     { DataType::ArrayI, tuple<TimeSeriesXf>() }, */
/* }; */
/* auto ReduceAll  = { */
/*     { DataType::ArrayF, tuple<Eigen::ArrayXXf>() }, */
/*     { DataType::ArrayF, tuple<TimeSeriesXf>() }, */
/*     { DataType::ArrayI, tuple<Eigen::ArrayXXi>() }, */
/*     { DataType::ArrayI, tuple<TimeSeriesI>() }, */
/* }; */


/* auto NodeSchema = { */

//arithmetic
    /* {NodeType::Add, { */
    /*     {"ExecType", ExecType::Reducer}, */
    /*     {"Signature", BinaryF}, */
    /*                 }, */
    /* }, */
    /* {NodeType::Sub, { */
    /*     {"Arity", 2}, */
    /*     {"ExecType", ExecType::Applier}, */
    /*     {"Signature", BinaryFF}, */
    /*                 }, */
    /* }, */
    /* {NodeType::Mul, { */
    /*     {"Arity", 2}, */
    /*     {"ExecType", ExecType::Reducer}, */
    /*     {"Signature", BinaryFF}, */
    /*                 }, */
    /* }, */
    /* {NodeType::Div, { */
    /*     {"Arity", 2}, */
    /*     {"ExecType", ExecType::Applier}, */
    /*     {"Signature", BinaryFF}, */
    /*                 }, */
    /* }, */
    /* {NodeType::Aq, { */
    /*     {"Arity", 2}, */
    /*     {"ExecType", ExecType::Applier}, */
    /*     {"Signature", BinaryFF}, */
    /*                }, */
    /* }, */
    /* {NodeType::Abs, { */
    /*     {"Arity", 1}, */
    /*     {"ExecType", ExecType::Transformer}, */
    /*     {"Signature", UnaryFF}, */
    /*                 }, */
    /* }, */
    /* {NodeType::Cbrt, { */
    /*     {"Arity", 1}, */
    /*     {"ExecType", ExecType::Transformer}, */
    /*     {"Signature", UnaryFF}, */
    /*                  }, */
    /* }, */
    /* {NodeType::Ceil, { */
    /*     {"Arity", 1}, */
    /*         {"ExecType", ExecType::Transformer}, */
    /*         {"Signature", UnaryIF}, */
    /*                  }, */
    /* }, */
    /* {NodeType::Floor, { */
    /*     {"Arity", 1}, */
    /*         {"ExecType", ExecType::Transformer}, */
    /*         {"Signature", UnaryIF}, */
    /*                   }, */
    /* }, */
    /* {NodeType::Exp, { */
    /*     {"Arity", 1}, */
    /*     {"ExecType", ExecType::Transformer}, */
    /*     {"Signature", UnaryFF}, */
    /*                 }, */
    /* }, */
    /* {NodeType::Log, { */
    /*     {"Arity", 1}, */
    /*     {"ExecType", ExecType::Transformer}, */
    /*     {"Signature", UnaryFF}, */
    /*                 }, */
    /* }, */
    /* {NodeType::Logabs, { */
    /*     {"Arity", 1}, */
    /*         {"ExecType", ExecType::Transformer}, */
    /*         {"Signature", UnaryFF}, */
    /*                    }, */
    /* }, */
    /* {NodeType::Log1p, { */
    /*     {"Arity", 1}, */
    /*         {"ExecType", ExecType::Transformer}, */
    /*         {"Signature", UnaryFF}, */
    /*                   }, */
    /* }, */
    /* {NodeType::Sqrt, { */
    /*     {"Arity", 1}, */
    /*         {"ExecType", ExecType::Transformer}, */
    /*         {"Signature", UnaryFF}, */
    /*                  }, */
    /* }, */
    /* {NodeType::Sqrtabs, { */
    /*     {"Arity", 1}, */
    /*         {"ExecType", ExecType::Transformer}, */
    /*         {"Signature", UnaryFF}, */
    /*                     }, */
    /* }, */
    /* {NodeType::Square, { */
    /*     {"Arity", 1}, */
    /*         {"ExecType", ExecType::Transformer}, */
    /*         {"Signature", UnaryFF}, */
    /*                    }, */
    /* }, */
    /* /1* {NodeType::Pow, { *1/ */
    /* /1*     {"Arity", 1}, *1/ */
    /* /1*     {"ExecType", ExecType::Transformer}, *1/ */
    /* /1*     {"Signature", UnaryFF}, *1/ */
    /* /1*                 }, *1/ */
    /* /1* }, *1/ */
/* // trigonometry */
    /* {NodeType::Acos, { */
    /*     {"Arity", 1}, */
    /*         {"ExecType", ExecType::Transformer}, */
    /*         {"Signature", UnaryFF}, */
    /*                  }, */
    /* }, */
    /* {NodeType::Asin, { */
    /*     {"Arity", 1}, */
    /*         {"ExecType", ExecType::Transformer}, */
    /*         {"Signature", UnaryFF}, */
    /*                  }, */
    /* }, */
    /* {NodeType::Atan, { */
    /*     {"Arity", 1}, */
    /*         {"ExecType", ExecType::Transformer}, */
    /*         {"Signature", UnaryFF}, */
    /*                  }, */
    /* }, */
    /* {NodeType::Cos, { */
    /*     {"Arity", 1}, */
    /*     {"ExecType", ExecType::Transformer}, */
    /*     {"Signature", UnaryFF}, */
    /*                 }, */
    /* }, */
    /* {NodeType::Cosh, { */
    /*     {"Arity", 1}, */
    /*         {"ExecType", ExecType::Transformer}, */
    /*         {"Signature", UnaryFF}, */
    /*                  }, */
    /* }, */
    /* {NodeType::Sin, { */
    /*     {"Arity", 1}, */
    /*     {"ExecType", ExecType::Transformer}, */
    /*     {"Signature", UnaryFF}, */
    /*                 }, */
    /* }, */
    /* {NodeType::Sinh, { */
    /*     {"Arity", 1}, */
    /*         {"ExecType", ExecType::Transformer}, */
    /*         {"Signature", UnaryFF}, */
    /*                  }, */
    /* }, */
    /* {NodeType::Tan, { */
    /*     {"Arity", 1}, */
    /*     {"ExecType", ExecType::Transformer}, */
    /*     {"Signature", UnaryFF}, */
    /*                 }, */
    /* }, */
    /* {NodeType::Tanh, { */
    /*     {"Arity", 1}, */
    /*         {"ExecType", ExecType::Transformer}, */
    /*         {"Signature", UnaryFF}, */
    /*                  }, */
    /* }, */

/* // logic; not sure these will make it in */
    /* {NodeType::And, { */
    /*     {"Arity", 1}, */
    /*     {"ExecType", ExecType::Reducer}, */
    /*     {"Signature", BinaryBB}, */
    /*                 }, */
    /* }, */
    /* {NodeType::Or, { */
    /*     {"ExecType", ExecType::Reducer}, */
    /*     {"Signature", BinaryBB}, */
    /*                }, */
    /* }, */
    /* {NodeType::Not, { */
    /*     {"ExecType", ExecType::Transformer}, */
    /*     {"Signature", UnaryBB}, */
    /*                 }, */
    /* }, */
    /* {NodeType::Xor, { */
    /*     {"ExecType", ExecType::Applier}, */
    /*     {"Signature", BinaryBB}, */
    /*                 }, */
    /* }, */

/* // decision (same) */
    /* {NodeType::Equals, { */
    /*         {"ExecType", ExecType::Reducer}, */
    /*         {"Signature", BinaryBF}, */
    /*                    }, */
    /* }, */
    /* {NodeType::LessThan, { */
    /*             {"ExecType", ExecType::Applier}, */
    /*             {"Signature", BinaryBF}, */
    /*                      }, */
    /* }, */
    /* {NodeType::GreaterThan, { */
    /*             {"ExecType", ExecType::Applier}, */
    /*             {"Signature", BinaryBF}, */
    /*                         }, */
    /* }, */
    /* {NodeType::Leq, { */
    /*     {"ExecType", ExecType::Applier}, */
    /*     {"Signature", BinaryBF}, */
    /*                 }, */
    /* }, */
    /* {NodeType::Geq, { */
    /*     {"ExecType", ExecType::Applier}, */
    /*     {"Signature", BinaryBF}, */
    /*                 }, */
    /* }, */

/* // summary stats */
    /* {NodeType::Min, { */
    /*     {"ExecType", ExecType::Reducer}, */
    /*     {"Signature", ReduceAll}, */
    /*                 }, */
    /* }, */
    /* {NodeType::Max, { */
    /*     {"ExecType", ExecType::Reducer}, */
    /*     {"Signature", ReduceAll}, */
    /*                 }, */
    /* }, */
    /* {NodeType::Mean, { */
    /*         {"ExecType", ExecType::Applier}, */
    /*         {"Signature", ReduceFF}, */
    /*                  }, */
    /* }, */
    /* {NodeType::Median, { */
    /*         {"ExecType", ExecType::Applier}, */
    /*         {"Signature", ReduceFF}, */
    /*                    }, */
    /* }, */
    /* {NodeType::Count, { */
    /*         {"ExecType", ExecType::Applier}, */
    /*         {"Signature", ReduceIA}, */
    /*                   }, */
    /* }, */
    /* {NodeType::Sum, { */
    /*     {"ExecType", ExecType::Applier}, */
    /*     {"Signature", ReduceAll}, */
    /*                 }, */
    /* }, */

/* // timing masks */
    /* {NodeType::Before, { */
    /*         {"ExecType", ExecType::Mapper}, */
    /*         {"Signature", BinaryFF}, */
    /*                    }, */
    /* }, */
    /* {NodeType::After, { */
    /*         {"ExecType", ExecType::Mapper}, */
    /*         {"Signature", BinaryFF}, */
    /*                   }, */
    /* }, */
    /* {NodeType::During, { */
    /*         {"ExecType", ExecType::Mapper}, */
    /*         {"Signature", BinaryFF}, */
    /*                    }, */
    /* }, */

/* //split */
    /* {NodeType::SplitBest, { */
    /*             {"ExecType", ExecType::Mapper}, */
    /*             {"Signature", {DataType::ArrayF, tuple<Eigen::ArrayXf,Eigen::ArrayXf>}, */
    /*                       }, */
    /* }, */
    /* {NodeType::SplitOn, { */
    /*         {"ExecType", ExecType::Mapper}, */
    /*         {"Signature", {DataType::ArrayF, {DataType::ArrayF, */
    /*                                         DataType::ArrayF, */
    /*                                         DataType::ArrayF} */
    /*                     }, */
    /*                     }, */
    /* }, */

/* // leaves */
    /* {NodeType::Constant, { */
    /*             {"ExecType", ExecType::Mapper}, */
    /*             {"Signature", BinaryFF}, */
    /*                      }, */
    /* }, */
    /* {NodeType::Variable, { */
    /*             {"ExecType", ExecType::Mapper}, */
    /*             {"Signature", BinaryFF}, */
    /*                      }, */
    /* }, */

/* // custom */
    /* {NodeType::CustomOp, { */
    /*             {"ExecType", ExecType::Mapper}, */
    /*             {"Signature", BinaryFF}, */
    /*                      }, */
    /* }, */
    /* {NodeType::CustomSplit { */
    /*             {"ExecType", ExecType::Mapper}, */
    /*             {"Signature", BinaryFF}, */
    /*                        }, */
    /* }, */
/* }; */
}
