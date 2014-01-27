set(BOOST_FILES
  #  src/bzip2.cpp
  src/file_descriptor.cpp
  src/gzip.cpp
  src/mapped_file.cpp
  src/zlib.cpp)

add_library(boost-iostream STATIC ${BOOST_FILES})
