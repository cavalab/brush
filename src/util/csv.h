/* Brush
copyright 2026 William La Cava
license: GNU/GPL v3
*/

#ifndef CSV_H
#define CSV_H

#include <cstdio>
#include <fstream>
#include <memory>
#include <sstream>
#include <string>
#include <type_traits>
#include <vector>

#include "error.h"

namespace Brush { namespace Util {

/*!
 * @class CsvWriter
 * @brief Minimal RFC 4180 CSV writer used by the run logs.
 *
 * - Fields containing a separator, a quote or a line break are quoted, and
 *   inner quotes are doubled, so expressions such as `Pow(x0,x1)` are safe.
 * - Files are opened in append mode. The header is written only when the file
 *   is new or empty; appending to a file whose header differs throws, instead
 *   of silently mixing two schemas in the same file.
 * - Every row is flushed, so logs are usable while (or after a crash during)
 *   a run.
 */
class CsvWriter
{
public:
    CsvWriter() = default;

    CsvWriter(const std::string& path, const std::vector<std::string>& header)
    {
        open(path, header);
    }

    void open(const std::string& path, const std::vector<std::string>& header)
    {
        close();

        n_columns = header.size();
        const std::string header_line = format_row(header);

        std::string existing;
        {
            std::ifstream in(path);
            if (in.is_open())
                std::getline(in, existing);
        }

        if (!existing.empty() && existing != header_line)
            HANDLE_ERROR_THROW(
                "Log file '" + path + "' already exists with different columns.\n"
                "  expected: " + header_line + "\n"
                "  found   : " + existing + "\n"
                "Use a different logfile or remove the old one.");

        out = std::make_shared<std::ofstream>(path, std::ofstream::app);
        if (!out->is_open())
            HANDLE_ERROR_THROW("Failed to open log file: " + path);

        if (existing.empty())
            *out << header_line << '\n' << std::flush;
    }

    bool is_open() const { return out && out->is_open(); }

    void close()
    {
        if (is_open())
            out->close();
        out.reset();
    }

    void write_row(const std::vector<std::string>& fields)
    {
        if (!is_open())
            HANDLE_ERROR_THROW("CsvWriter: writing to a closed file");

        if (fields.size() != n_columns)
            HANDLE_ERROR_THROW(
                "CsvWriter: row has " + std::to_string(fields.size())
                + " fields, header has " + std::to_string(n_columns));

        *out << format_row(fields) << '\n' << std::flush;
    }

    /// quotes a field if it contains a separator, a quote or a line break
    static std::string escape(const std::string& field)
    {
        if (field.find_first_of(",\"\n\r") == std::string::npos)
            return field;

        std::string quoted = "\"";
        for (char c : field)
        {
            if (c == '"')
                quoted += '"';
            quoted += c;
        }
        quoted += '"';
        return quoted;
    }

    /// converts a value to a field. Floats use enough digits to round-trip.
    template <typename V>
    static std::string field(const V& value)
    {
        if constexpr (std::is_same_v<V, bool>)
            return value ? "1" : "0";
        else if constexpr (std::is_floating_point_v<V>)
        {
            char buf[32];
            std::snprintf(buf, sizeof(buf), "%.9g", static_cast<double>(value));
            return buf;
        }
        else if constexpr (std::is_arithmetic_v<V>)
            return std::to_string(value);
        else
        {
            std::ostringstream ss;
            ss << value;
            return ss.str();
        }
    }

private:
    static std::string format_row(const std::vector<std::string>& fields)
    {
        std::string line;
        for (size_t i = 0; i < fields.size(); ++i)
        {
            if (i > 0)
                line += ',';
            line += escape(fields[i]);
        }
        return line;
    }

    // shared so that the owner (e.g. Engine) stays copyable
    std::shared_ptr<std::ofstream> out;
    size_t n_columns = 0;
};

} } // Util, Brush

#endif
