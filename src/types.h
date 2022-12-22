/* Brush

copyright 2020 William La Cava
license: GNU/GPL v3
*/
#ifndef TYPES_H
#define TYPES_H

#include "init.h"
#include <variant>

namespace Brush {

////////////////////////////////////////////////////////////////////////////////
// Program 
template<typename T> struct Program;
typedef Program<ArrayXf> RegressorProgram;
typedef Program<ArrayXb> ClassifierProgram;
typedef Program<ArrayXi> MulticlassClassifierProgram;
typedef Program<ArrayXXf> RepresenterProgram;

enum class ProgramType : uint32_t {
    Regressor,
    BinaryClassifier,
    MulticlassClassifier,
    Representer
};

template<typename T> struct ProgramTypeEnum;
template <>
struct ProgramTypeEnum<RegressorProgram>
{
    static constexpr ProgramType value = ProgramType::Regressor;
};
template <>
struct ProgramTypeEnum<ClassifierProgram>
{
    static constexpr ProgramType value = ProgramType::BinaryClassifier;
};
template <>
struct ProgramTypeEnum<MulticlassClassifierProgram>
{
    static constexpr ProgramType value = ProgramType::MulticlassClassifier;
};
template <>
struct ProgramTypeEnum<RepresenterProgram>
{
    static constexpr ProgramType value = ProgramType::Representer;
};

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
        TimeSeriesf
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
    TimeSeriesF
};
template<DataType D> struct DataEnumType; 
template<> struct DataEnumType<DataType::ArrayB>{ using type = ArrayXb; };
template<> struct DataEnumType<DataType::ArrayI>{ using type = ArrayXi; };
template<> struct DataEnumType<DataType::ArrayF>{ using type = ArrayXf; };
template<> struct DataEnumType<DataType::MatrixB>{ using type = ArrayXXb; };
template<> struct DataEnumType<DataType::MatrixI>{ using type = ArrayXXi; };
template<> struct DataEnumType<DataType::MatrixF>{ using type = ArrayXXf; };
template<> struct DataEnumType<DataType::TimeSeriesB>{ using type = Data::TimeSeriesb; };
template<> struct DataEnumType<DataType::TimeSeriesI>{ using type = Data::TimeSeriesi; }; 
template<> struct DataEnumType<DataType::TimeSeriesF>{ using type = Data::TimeSeriesf; };

template<typename T> struct DataTypeEnum; 
template<> struct DataTypeEnum<ArrayXb>{ static constexpr DataType value = DataType::ArrayB; };
template<> struct DataTypeEnum<ArrayXi>{ static constexpr DataType value = DataType::ArrayI; };
template<> struct DataTypeEnum<ArrayXf>{ static constexpr DataType value = DataType::ArrayF; };
template<> struct DataTypeEnum<ArrayXXb>{ static constexpr DataType value = DataType::MatrixB; };
template<> struct DataTypeEnum<ArrayXXi>{ static constexpr DataType value = DataType::MatrixI; };
template<> struct DataTypeEnum<ArrayXXf>{ static constexpr DataType value = DataType::MatrixF; };
template<> struct DataTypeEnum<Data::TimeSeriesb>{ static constexpr DataType value = DataType::TimeSeriesB; };
template<> struct DataTypeEnum<Data::TimeSeriesi>{ static constexpr DataType value = DataType::TimeSeriesI; };
template<> struct DataTypeEnum<Data::TimeSeriesf>{ static constexpr DataType value = DataType::TimeSeriesF; };

} // Brush

#endif