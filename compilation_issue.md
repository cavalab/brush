Error overview: I am having compilation error in the linking step. Specifically, the error message is "Undefined symbols for architecture x86_64" and "ld: symbols not found for architecture x86_64". It looks like it has to do with string format, and a related package is fmt.  

Error detail: Originally I was trying on my M1 chip mac with Arm architecture, but it didn't work out, so I switched to a different mac using x86 architecture, and I encountered the following error:

Undefined symbols for architecture x86_64:
  "__ZN3fmt2v97vformatB5cxx11ENS0_17basic_string_viewIcEENS0_17basic_format_argsINS0_20basic_format_contextINS0_8appenderEcEEEE", referenced from:
      __ZNK5Brush13DispatchTableILb1EE3GetIN5Eigen5ArrayIfLin1ELi1ELi0ELin1ELi1EEEEERKSt8functionIFT_RKNS_4data4DataERNS_10tree_node_INS_4NodeEEEEENS_8NodeTypeEm in brushgp.cpp.o
      __ZNK5Brush13DispatchTableILb0EE3GetIN5Eigen5ArrayIfLin1ELi1ELi0ELin1ELi1EEEEERKSt8functionIFT_RKNS_4data4DataERNS_10tree_node_INS_4NodeEEEEENS_8NodeTypeEm in brushgp.cpp.o
      __ZNK5Brush13DispatchTableILb1EE3GetIN5Eigen5ArrayIfLin1ELi1ELi0ELin1ELi1EEEEERKSt8functionIFT_RKNS_4data4DataERNS_10tree_node_INS_4NodeEEEEENS_8NodeTypeEm in dispatch_table.cpp.o
      __ZNK5Brush13DispatchTableILb1EE3GetIN5Eigen5ArrayIfLin1ELin1ELi0ELin1ELin1EEEEERKSt8functionIFT_RKNS_4data4DataERNS_10tree_node_INS_4NodeEEEEENS_8NodeTypeEm in dispatch_table.cpp.o
      __ZNK5Brush13DispatchTableILb1EE3GetIN5Eigen5ArrayIbLin1ELi1ELi0ELin1ELi1EEEEERKSt8functionIFT_RKNS_4data4DataERNS_10tree_node_INS_4NodeEEEEENS_8NodeTypeEm in dispatch_table.cpp.o
      __ZNK5Brush13DispatchTableILb1EE3GetIN5Eigen5ArrayIbLin1ELin1ELi0ELin1ELin1EEEEERKSt8functionIFT_RKNS_4data4DataERNS_10tree_node_INS_4NodeEEEEENS_8NodeTypeEm in dispatch_table.cpp.o
      __ZNK5Brush13DispatchTableILb1EE3GetINS_4data10TimeSeriesIfEEEERKSt8functionIFT_RKNS3_4DataERNS_10tree_node_INS_4NodeEEEEENS_8NodeTypeEm in dispatch_table.cpp.o
      ...
ld: symbol(s) not found for architecture x86_64 

Error attempt: The things I have tried so far:

initial attemps:
1. change compiler from Apple Clang 13 to gcc-11
2. include fmt header in brushgp and dispatch_table
 
suggested attempts:
3. include fmt header in third party under src  
4. brew install fmt and link to it

further attemps:
5. followed the stackoverflow post "https://stackoverflow.com/questions/56608684/how-to-use-the-fmt-library-without-getting-undefined-symbols-for-architecture-x" and tried fmt/format.h instead of fmt/core.h

future thoughts:
6. maybe I should use the optional header-only mode or link to the fmt library?

It seems that all of the above attempts failed.
