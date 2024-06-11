#include "testsHeader.h"

using namespace Brush::Pop;
using namespace Brush::Sel;
using namespace Brush::Eval;
using namespace Brush::Sel;

TEST(Params, ParamsTests)
{

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
	
	params.set_objectives({"fitness","complexity"});
	ASSERT_EQ(params.get_objectives().size(), 2);
	ASSERT_STREQ(params.get_objectives()[0].c_str(), "fitness");
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
