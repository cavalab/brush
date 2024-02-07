/* Brush

copyright 2020 William La Cava
license: GNU/GPL v3
*/
#ifndef TYPES_H
#define TYPES_H

#include <variant>
#include <ceres/jet.h>
#include <concepts>

namespace Brush {

/// @brief checks whether all the types match.
/// @tparam First 
/// @tparam ...Next 
template<typename First, typename ... Next>
struct all_same{
    static constexpr bool value {(std::is_same_v<First,Next> && ...)};
};
template<typename First, typename ... Next>
static constexpr bool all_same_v = all_same<First, Next...>::value;

/// @brief checks whether any of the types match the first one.
/// @tparam First 
/// @tparam ...Next 
template<typename First, typename ... Next>
struct is_one_of{
    static constexpr bool value {(std::is_same_v<First,Next> || ...)};
};
template<typename First, typename ... Next>
static constexpr bool is_one_of_v = is_one_of<First, Next...>::value;
// // see https://en.cppreference.com/w/cpp/concepts/same_as 
template<typename T, typename ... U>
concept IsAnyOf = (std::same_as<T, U> || ...);
////////////////////////////////////////////////////////////////////////////////
// Eigen types
typedef Eigen::Array<bool,Eigen::Dynamic,1> ArrayXb;
typedef Eigen::Array<int,Eigen::Dynamic,1> ArrayXi;
typedef Eigen::Array<bool,Eigen::Dynamic,Eigen::Dynamic> ArrayXXb;
typedef Eigen::Array<int,Eigen::Dynamic,Eigen::Dynamic> ArrayXXi;
// Ceres types
static constexpr int stride = 8;
typedef ceres::Jet<float, stride> fJet;
typedef ceres::Jet<int, stride> iJet;
typedef ceres::Jet<bool, stride> bJet;

typedef Eigen::Array<fJet,Eigen::Dynamic,1> ArrayXfJet;
typedef Eigen::Array<iJet,Eigen::Dynamic,1> ArrayXiJet;
typedef Eigen::Array<bJet,Eigen::Dynamic,1> ArrayXbJet;
typedef Eigen::Array<fJet,Eigen::Dynamic,Eigen::Dynamic> ArrayXXfJet;
typedef Eigen::Array<iJet,Eigen::Dynamic,Eigen::Dynamic> ArrayXXiJet;
typedef Eigen::Array<bJet,Eigen::Dynamic,Eigen::Dynamic> ArrayXXbJet;

/// @brief Returns the weight type associated with the scalar type underlying T.
/// @tparam T one of the State types, e.g. ArrayXf
template <typename T>
struct WeightType {
    typedef std::conditional_t<
        is_one_of_v<typename T::Scalar, fJet, iJet, bJet>,
        fJet, 
        float
    > type; 
};
template<typename T> 
using WeightType_t = typename WeightType<T>::type;
////////////////////////////////////////////////////////////////////////////////
// Program 
enum class ProgramType : uint32_t {
    Regressor,
    BinaryClassifier,
    MulticlassClassifier,
    Representer
};
using PT = ProgramType;
template<PT T> struct Program;
typedef Program<PT::Regressor> RegressorProgram;
typedef Program<PT::BinaryClassifier> ClassifierProgram;
typedef Program<PT::MulticlassClassifier> MulticlassClassifierProgram;
typedef Program<PT::Representer> RepresenterProgram;

////////////////////////////////////////////////////////////////////////////////
// Individual 
namespace Pop {
    template<Brush::ProgramType T> class Individual;
}
typedef Pop::Individual<PT::Regressor> RegressorIndividual;
typedef Pop::Individual<PT::BinaryClassifier> ClassifierIndividual;
typedef Pop::Individual<PT::MulticlassClassifier> MulticlassClassifierIndividual;
typedef Pop::Individual<PT::Representer> RepresenterIndividual;

////////////////////////////////////////////////////////////////////////////////
// Estimator 
using PT = ProgramType;
template<PT T> class Estimator;
typedef Estimator<PT::Regressor> RegressorEstimator;
typedef Estimator<PT::BinaryClassifier> ClassifierEstimator;
typedef Estimator<PT::MulticlassClassifier> MulticlassClassifierEstimator;
typedef Estimator<PT::Representer> RepresenterEstimator;

////////////////////////////////////////////////////////////////////////////////
// Data 
namespace Data{
    template<class T> struct TimeSeries;
    /**
     * @brief TimeSeries convenience typedefs.
     * 
     */
    typedef TimeSeries<bool> TimeSeriesb;
    typedef TimeSeries<int> TimeSeriesi;
    typedef TimeSeries<float> TimeSeriesf;
    typedef TimeSeries<bJet> TimeSeriesbJet;
    typedef TimeSeries<iJet> TimeSeriesiJet;
    typedef TimeSeries<fJet> TimeSeriesfJet;

    ////////////////////////////////////////////////////////////////////////////
    /// @brief defines the possible types of data flowing thru nodes.
    typedef std::variant<
        ArrayXb,
        ArrayXi,
        ArrayXf,
        ArrayXXb,
        ArrayXXi,
        ArrayXXf,
        TimeSeriesb,
        TimeSeriesi,
        TimeSeriesf,
        // Jet types
        ArrayXbJet,
        ArrayXiJet,
        ArrayXfJet,
        ArrayXXbJet,
        ArrayXXiJet,
        ArrayXXfJet,
        TimeSeriesbJet,
        TimeSeriesiJet,
        TimeSeriesfJet
        >
        State;
}
/// @brief data types. 
enum class DataType : uint32_t {
    ArrayB, 
    ArrayI, 
    ArrayF, 
    MatrixB, 
    MatrixI, 
    MatrixF, 
    TimeSeriesB, 
    TimeSeriesI,
    TimeSeriesF,
    ArrayBJet, 
    ArrayIJet, 
    ArrayFJet, 
    MatrixBJet, 
    MatrixIJet, 
    MatrixFJet, 
    TimeSeriesBJet, 
    TimeSeriesIJet,
    TimeSeriesFJet
};

#ifndef DOXYGEN_SKIP
// map DataType values to JSON as strings
NLOHMANN_JSON_SERIALIZE_ENUM( DataType, {
    {DataType::ArrayB, "ArrayB"},
    {DataType::ArrayI, "ArrayI"},
    {DataType::ArrayF, "ArrayF"},
    {DataType::MatrixB, "MatrixB"},
    {DataType::MatrixI, "MatrixI"},
    {DataType::MatrixF, "MatrixF"},
    {DataType::TimeSeriesB, "TimeSeriesB"},
    {DataType::TimeSeriesI,"TimeSeriesI"},
    {DataType::TimeSeriesF,"TimeSeriesF"},
    {DataType::ArrayBJet, "ArrayBJet"},
    {DataType::ArrayIJet, "ArrayIJet"},
    {DataType::ArrayFJet, "ArrayFJet"},
    {DataType::MatrixBJet, "MatrixBJet"},
    {DataType::MatrixIJet, "MatrixIJet"},
    {DataType::MatrixFJet, "MatrixFJet"},
    {DataType::TimeSeriesBJet, "TimeSeriesBJet"},
    {DataType::TimeSeriesIJet,"TimeSeriesIJet"},
    {DataType::TimeSeriesFJet, "TimeSeriesFJet"},
})
#endif

using DT = DataType; 

template<DT D> struct DataEnumType; 
template<> struct DataEnumType<DT::ArrayB>{ using type = ArrayXb; };
template<> struct DataEnumType<DT::ArrayI>{ using type = ArrayXi; };
template<> struct DataEnumType<DT::ArrayF>{ using type = ArrayXf; };
template<> struct DataEnumType<DT::MatrixB>{ using type = ArrayXXb; };
template<> struct DataEnumType<DT::MatrixI>{ using type = ArrayXXi; };
template<> struct DataEnumType<DT::MatrixF>{ using type = ArrayXXf; };
template<> struct DataEnumType<DT::TimeSeriesB>{ using type = Data::TimeSeriesb; };
template<> struct DataEnumType<DT::TimeSeriesI>{ using type = Data::TimeSeriesi; }; 
template<> struct DataEnumType<DT::TimeSeriesF>{ using type = Data::TimeSeriesf; };
template<> struct DataEnumType<DT::ArrayBJet>{ using type = ArrayXbJet; };
template<> struct DataEnumType<DT::ArrayIJet>{ using type = ArrayXiJet; };
template<> struct DataEnumType<DT::ArrayFJet>{ using type = ArrayXfJet; };
template<> struct DataEnumType<DT::MatrixBJet>{ using type = ArrayXXbJet; };
template<> struct DataEnumType<DT::MatrixIJet>{ using type = ArrayXXiJet; };
template<> struct DataEnumType<DT::MatrixFJet>{ using type = ArrayXXfJet; };

template<typename T> struct DataTypeEnum;
template <> struct DataTypeEnum<ArrayXb> { static constexpr DT value = DT::ArrayB; };
template <> struct DataTypeEnum<ArrayXi> { static constexpr DT value = DT::ArrayI; };
template <> struct DataTypeEnum<ArrayXf> { static constexpr DT value = DT::ArrayF; };
template <> struct DataTypeEnum<ArrayXXb> { static constexpr DT value = DT::MatrixB; };
template <> struct DataTypeEnum<ArrayXXi> { static constexpr DT value = DT::MatrixI; };
template <> struct DataTypeEnum<ArrayXXf> { static constexpr DT value = DT::MatrixF; };
template <> struct DataTypeEnum<Data::TimeSeriesb> { static constexpr DT value = DT::TimeSeriesB; };
template <> struct DataTypeEnum<Data::TimeSeriesi> { static constexpr DT value = DT::TimeSeriesI; };
template <> struct DataTypeEnum<Data::TimeSeriesf> { static constexpr DT value = DT::TimeSeriesF; };
template <> struct DataTypeEnum<ArrayXbJet> { static constexpr DT value = DT::ArrayBJet; };
template <> struct DataTypeEnum<ArrayXiJet> { static constexpr DT value = DT::ArrayIJet; };
template <> struct DataTypeEnum<ArrayXfJet> { static constexpr DT value = DT::ArrayFJet; };
template <> struct DataTypeEnum<ArrayXXbJet> { static constexpr DT value = DT::MatrixBJet; };
template <> struct DataTypeEnum<ArrayXXiJet> { static constexpr DT value = DT::MatrixIJet; };
template <> struct DataTypeEnum<ArrayXXfJet> { static constexpr DT value = DT::MatrixFJet; };
template <>
struct DataTypeEnum<Data::TimeSeriesbJet>
{
    static constexpr DT value = DT::TimeSeriesBJet;
};
template <>
struct DataTypeEnum<Data::TimeSeriesiJet>
{
    static constexpr DT value = DT::TimeSeriesIJet;
};
template <>
struct DataTypeEnum<Data::TimeSeriesfJet>
{
    static constexpr DT value = DT::TimeSeriesFJet;
};

// check for std::array types
template <typename T>
struct is_std_array : std::false_type {}; 
template <typename V, size_t n>
struct is_std_array<std::array<V, n>> : std::true_type {}; 
template <typename T>
static constexpr bool is_std_array_v = is_std_array<T>::value;

// check for Eigen::Array types
template<typename T>
struct is_eigen_array
 : std::is_base_of<Eigen::ArrayBase<std::decay_t<T> >, std::decay_t<T> > {};

template<typename T>
static constexpr bool is_eigen_array_v = is_eigen_array<T>::value;

// check for tuple types
template <typename> struct is_tuple: std::false_type {};
template <typename ...T> struct is_tuple<std::tuple<T...>>: std::true_type {};
template <typename T>
static constexpr bool is_tuple_v = is_tuple<T>::value;
} // Brush

#endif
