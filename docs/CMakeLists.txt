find_package(Doxygen REQUIRED)
find_package(Sphinx REQUIRED)

get_target_property(BRUSH_PUBLIC_SOURCE_DIR _brush INTERFACE_INCLUDE_DIRECTORIES)
file(GLOB_RECURSE BRUSH_PUBLIC_HEADERS ${BRUSH_PUBLIC_SOURCE_DIR}/*.h)

set(DOXYGEN_INPUT_DIR ${PROJECT_SOURCE_DIR}/src)
set(DOXYGEN_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/doxygen)
set(DOXYGEN_INDEX_FILE ${DOXYGEN_OUTPUT_DIR}/xml/index.xml)
set(DOXYFILE_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
set(DOXYFILE_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

configure_file(${DOXYFILE_IN} ${DOXYFILE_OUT} @ONLY)

file(MAKE_DIRECTORY ${DOXYGEN_OUTPUT_DIR})

add_custom_command(OUTPUT ${DOXYGEN_INDEX_FILE}
                   DEPENDS ${BRUSH_PUBLIC_HEADERS}
                   COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYFILE_OUT}
                   MAIN_DEPENDENCY ${DOXYFILE_OUT} ${DOXYFILE_IN}
                   COMMENT "Generating Doxygen docs"
                   VERBATIM)

add_custom_target(Doxygen ALL DEPENDS ${DOXYGEN_INDEX_FILE})

set(SPHINX_SOURCE ${CMAKE_CURRENT_SOURCE_DIR})
set(SPHINX_BUILD ${CMAKE_CURRENT_BINARY_DIR}/sphinx)
set(SPHINX_INDEX_FILE ${SPINX_BUILD}/index.html)

# Only rebuild sphinx documentations when:
#  - Doxygen has been re-run
#  - Doc files have updated
#  - Sphinx config has been updated
add_custom_command(OUTPUT ${SPHINX_INDEX_FILE}
                   COMMAND
                      ${SPHINX_EXECUTABLE} -b html
                      -Dbreathe_projects.brush=${DOXYGEN_OUTPUT_DIR}/xml
                      -D master_doc=index
                      ${SPHINX_SOURCE} ${SPHINX_BUILD}
                   WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
                   DEPENDS
                      ${CMAKE_CURRENT_SOURCE_DIR}/index.md
                      ${DOXYGEN_INDEX_FILE}
                   MAIN_DEPENDENCY ${SPHINX_SOURCE}/conf.py
                   COMMENT "Generating documentation pages with Sphinx")

add_custom_target(Sphinx ALL DEPENDS ${SPHINX_INDEX_FILE})

include(GNUInstallDirs)
install(DIRECTORY ${SPHINX_BUILD}
        DESTINATION ${CMAKE_INSTALL_DOCDIR})