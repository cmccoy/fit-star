#include "fit_star_config.h"
#include "protobuf_util.hpp"

#include <cpplog.hpp>

// STL
#include <cassert>
#include <fstream>
#include <iostream>
#include "mutationio.pb.h"

#include <boost/program_options.hpp>

using namespace fit_star;
namespace po = boost::program_options;

cpplog::StdErrLogger logger;

int main(const int argc, const char** argv)
{
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    std::string inputPath;

    // command-line parsing
    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "Produce help message")
    ("version,v", "Show version")
    ("input-file,i", po::value(&inputPath)->required(),
     "input file(s) - output of build-mutation-matrices [required]");

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
              options(desc).run(), vm);

    if(vm.count("help")) {
        std::cout << desc << '\n';
        return 0;
    }
    if(vm.count("version")) {
        std::cout << star_fit::FIT_STAR_VERSION << '\n';
        return 0;
    }

    po::notify(vm);

    size_t count = 0;
    std::fstream in(inputPath, std::ios::binary | std::ios::in);
    for(DelimitedProtocolBufferIterator<mutationio::MutationCount> it(in, true), end; it != end; it++) {
        const mutationio::MutationCount& m = *it;
        std::cout << m.DebugString() << "\n---------------------------------\n";
        count++;
    }
    std::cout << "Count: " << count << '\n';

    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}
