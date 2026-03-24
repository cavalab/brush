#include "testsHeader.h"

using namespace Brush::Pop;
using namespace Brush::Sel;
using namespace Brush::Eval;
using namespace Brush::Sel;

TEST(Params, ParamsTests)
{
	std::cout << "int: min=" << std::numeric_limits<int>::min() << " max=" << std::numeric_limits<int>::max() << "\n";
	std::cout << "unsigned int: min=" << std::numeric_limits<unsigned int>::min() << " max=" << std::numeric_limits<unsigned int>::max() << "\n";
	std::cout << "short: min=" << std::numeric_limits<short>::min() << " max=" << std::numeric_limits<short>::max() << "\n";
	std::cout << "unsigned short: min=" << std::numeric_limits<unsigned short>::min() << " max=" << std::numeric_limits<unsigned short>::max() << "\n";
	std::cout << "long: min=" << std::numeric_limits<long>::min() << " max=" << std::numeric_limits<long>::max() << "\n";
	std::cout << "unsigned long: min=" << std::numeric_limits<unsigned long>::min() << " max=" << std::numeric_limits<unsigned long>::max() << "\n";
	std::cout << "long long: min=" << std::numeric_limits<long long>::min() << " max=" << std::numeric_limits<long long>::max() << "\n";
	std::cout << "unsigned long long: min=" << std::numeric_limits<unsigned long long>::min() << " max=" << std::numeric_limits<unsigned long long>::max() << "\n";

    Parameters params;
	
    params.set_max_size(12);
    ASSERT_EQ(params.max_size, 12);
    ASSERT_EQ(params.get_max_size(), 12);
	
	params.set_max_depth(4);
	ASSERT_EQ(params.max_depth, 4);
	ASSERT_EQ(params.get_max_depth(), 4);

	params.set_max_depth(6);
	ASSERT_EQ(params.max_depth, 6);
	ASSERT_EQ(params.get_max_depth(), 6);
	
	params.set_objectives({"scorer","complexity"});
	ASSERT_EQ(params.get_objectives().size(), 2);
	ASSERT_STREQ(params.get_objectives()[0].c_str(), params.get_scorer().c_str());
	ASSERT_STREQ(params.get_objectives()[1].c_str(), "complexity");
	
    // TODO: implement logger and verbosity and make this work
	// string str1 = "Hello\n";
	// string str2 = logger.log("Hello", 0);
	// ASSERT_STREQ(str1.c_str(), str2.c_str());
	
	// str2 = logger.log("Hello", 2);
	// ASSERT_STREQ(str1.c_str(), str2.c_str());
	
	// str2 = logger.log("Hello", 3);
	// ASSERT_STREQ(str1.c_str(), str2.c_str());
	
	// ft.params.set_verbosity(2);
	// ASSERT_EQ(ft.params.verbosity, 2);
	// ASSERT_STREQ("", logger.log("Hello", 3).c_str());  
}
