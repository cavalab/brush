#include "testsHeader.h"
#include "../../src/util/csv.h"

#include <cstdio>
#include <fstream>

using Brush::Util::CsvWriter;

namespace {
    vector<string> read_lines(const string& path)
    {
        std::ifstream in(path);
        vector<string> lines;
        for (string line; std::getline(in, line); )
            lines.push_back(line);
        return lines;
    }
}

TEST(Logging, CsvEscape)
{
    ASSERT_EQ(CsvWriter::escape("x0"), "x0");
    ASSERT_EQ(CsvWriter::escape("Pow(x0,x1)"), "\"Pow(x0,x1)\"");
    ASSERT_EQ(CsvWriter::escape("a\"b"), "\"a\"\"b\"");
    ASSERT_EQ(CsvWriter::escape("a\nb"), "\"a\nb\"");
    ASSERT_EQ(CsvWriter::escape(""), "");
}

TEST(Logging, CsvField)
{
    ASSERT_EQ(CsvWriter::field(3), "3");
    ASSERT_EQ(CsvWriter::field(true), "1");
    ASSERT_EQ(CsvWriter::field(0.1f), "0.100000001");  // round-trips float precision
    ASSERT_EQ(CsvWriter::field(string("abc")), "abc");
}

TEST(Logging, CsvWriterHeaderOnceAndQuoting)
{
    const string path = "./tests/cpp/__test_csv_writer.csv";
    std::remove(path.c_str());

    {
        CsvWriter w(path, {"id", "model"});
        w.write_row({"1", "Add(x0,x1)"});
    }
    {
        // appending with the same header must not repeat it
        CsvWriter w(path, {"id", "model"});
        w.write_row({"2", "x0"});
        ASSERT_THROW(w.write_row({"too", "many", "fields"}), std::runtime_error);
    }

    auto lines = read_lines(path);
    ASSERT_EQ(lines.size(), 3);
    ASSERT_EQ(lines[0], "id,model");
    ASSERT_EQ(lines[1], "1,\"Add(x0,x1)\"");
    ASSERT_EQ(lines[2], "2,x0");

    // a different schema must not be mixed into the same file
    ASSERT_THROW(CsvWriter(path, {"id", "other"}), std::runtime_error);

    std::remove(path.c_str());
}
