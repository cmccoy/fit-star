#include <algorithm>
#include <vector>
#include <fstream>
#include <iostream>
#include "mutationlist.pb.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/gzip_stream.h>

#include "gtr.hpp"
#include "sequence.hpp"

using namespace gtr;
using Eigen::Matrix4d;

std::vector<Sequence> load_sequences_from_file(const std::string& path)
{
    std::fstream in(path, std::ios::in | std::ios::binary);
    google::protobuf::io::IstreamInputStream raw_in(&in);
    google::protobuf::io::GzipInputStream zip_in(&raw_in);
    google::protobuf::io::CodedInputStream coded_in(&zip_in);

    std::vector<Sequence> sequences;

    while(true) {
        uint32_t size = 0;
        bool success = false;
        success = coded_in.ReadVarint32(&size);
        if(!success) break;
        mutationio::MutationCount m;
        std::string s;
        coded_in.ReadString(&s, size);
        success = m.ParseFromString(s);
        assert(success && "Failed to parse");
        assert(m.mutations_size() == 16 && "Unexpected mutations count");
        Sequence sequence;
        sequence.distance = m.has_distance() ? m.distance() : 0.1;
        if(m.has_name())
            sequence.name = m.name();
        for(size_t i = 0; i < 4; i++)
            for(size_t j = 0; j < 4; j++)
                sequence.transitions(i, j) = m.mutations(4*i + j);
        sequences.push_back(std::move(sequence));
    }
    return sequences;
}

int main()
{
    GOOGLE_PROTOBUF_VERIFY_VERSION;
    std::vector<Sequence> sequences = load_sequences_from_file("test.muts.pb.gz");

    std::cout << sequences.size() << " sequences." << '\n';

    google::protobuf::ShutdownProtobufLibrary();

    gtr::GTRParameters params;
    empirical_model(sequences, params);

    gtr::GTRModel model = params.buildModel();

    std::cout << "Initial log-like: " << star_likelihood(model, sequences) << '\n';

    optimize(params, sequences);

    auto f = [](double acc, const Sequence& s) { return acc + s.distance; };
    const double mean_branch_length = std::accumulate(sequences.begin(), sequences.end(), 0.0, f) / sequences.size();

    std::cout << "Mean branch length: " << mean_branch_length << '\n';

    std::cout << "Final log-like: " << star_likelihood(params.buildModel(), sequences) << '\n';

    std::cout << "x=" << params.params.transpose() << '\n';
    std::cout << "Q=" << params.buildQMatrix() << '\n';
    std::cout << "P(0.01)=" << params.buildModel().buildPMatrix(0.01) << '\n';

    return 0;
}