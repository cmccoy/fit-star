set(BOOST_FILES
    src/cmdline.cpp
    src/config_file.cpp
    src/convert.cpp
    src/options_description.cpp
    src/parsers.cpp
    src/positional_options.cpp
    src/split.cpp
    src/utf8_codecvt_facet.cpp
    src/value_semantic.cpp
    src/variables_map.cpp
    src/winmain.cpp)

add_library(boost-programoptions STATIC ${BOOST_FILES})
