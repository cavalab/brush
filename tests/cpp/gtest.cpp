#include "testsHeader.h"

int main(int argc, char **argv) {
    Brush::Util::r.set_seed(42);
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
