#include "sam.h"
#include "faidx.h"

#include <cassert>
#include <cstdio>
#include <iostream>
#include <fstream>
#include <string>
#include <vector>

#include <boost/program_options.hpp>

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/gzip_stream.h>

#include "mutationio.pb.h"
#include "fit_star_config.h"
#include "sam_util.hpp"
#include "protobuf_util.hpp"

namespace po = boost::program_options;
namespace protoio = google::protobuf::io;

bool endsWith(const std::string& s, const std::string& suffix)
{
    if(s.length() >= suffix.length()) {
        return (0 == s.compare(s.length() - suffix.length(), suffix.length(), suffix));
    } else {
        return false;
    }
}

inline int nt16ToIdx(const int b)
{
    switch(b) {
        case 1: return 0; // A
        case 2: return 1; // C
        case 4: return 2; // G
        case 8: return 3; // T
        default: return 4; // N
    }
}

void mutationCountOfSequence(mutationio::MutationCount& count,
                             const bam1_t* b,
                             const std::string& ref,
                             const bool no_ambiguous,
                             const std::string partition_name = "",
                             const bool by_codon = false)
{
    assert(b != nullptr && "null bam record");
    std::vector<std::vector<int>> partitions(by_codon ? 3 : 1);
    for(std::vector<int>& v : partitions)
        v.resize(16);

    const uint32_t* cigar = bam1_cigar(b);
    const uint8_t* seq = bam1_seq(b);
    uint32_t qi = 0, ri = b->core.pos;
    const int8_t* bq = reinterpret_cast<int8_t*>(bam_aux_get(b, "bq"));
    if(no_ambiguous) {
        assert(bq != NULL && "No bq tag");
    }
    for(uint32_t cidx = 0; cidx < b->core.n_cigar; cidx++) {
        const uint32_t clen = bam_cigar_oplen(cigar[cidx]);
        const uint32_t consumes = bam_cigar_type(cigar[cidx]); // bit 1: consume query; bit 2: consume reference
        if((consumes & 0x3) == 0x3) {  // Reference and query
            for(uint32_t i = 0; i < clen; i++) {
                const int qb = nt16ToIdx(bam1_seqi(seq, qi + i)),
                          rb = nt16ToIdx(bam_nt16_table[static_cast<int>(ref[ri + i])]);
                if(qb < 4 && rb < 4 && (!no_ambiguous || bq[qi + i] % 100 == 0)) {
                    partitions[by_codon ? (ri + i) % 3 : 0][(rb * 4) + qb] += 1;
                }
            }
        }
        if(consumes & 0x1) // Consumes query
            qi += clen;
        else if(consumes & 0x2) // Consumes reference
            ri += clen;
    }

    int p = 0;
    for(const std::vector<int>& v : partitions) {
        std::string name = partition_name;
        if(partitions.size() > 1)
            name += "p" + std::to_string(p++);

        mutationio::Partition* partition = count.add_partition();
        partition->set_name(name);
        for(const int i : v)
            partition->add_substitution(i);
    }
}

int usage(po::options_description& desc)
{
    std::cerr << "Usage: build_mutation_matrices [options] <ref.fasta> <in.bam> <out.bin>\n";
    std::cerr << desc << '\n';
    return 1;
}


int main(int argc, char* argv[])
{
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    std::string fastaPath, outputPath;
    std::vector<std::string> bamPaths;

    bool no_ambiguous = false;
    bool by_codon = false;
    bool no_group_by_qname = false;
    size_t maxRecords = 0;
    int prefix = 4;

    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "Produce help message")
    ("version,v", "Print version")
    ("no-ambiguous", po::bool_switch(&no_ambiguous), "Do not include ambiguous sites")
    ("by-codon", po::bool_switch(&by_codon), "Partition by codon")
    ("max-records,n", po::value(&maxRecords), "Maximum number of records to parse")
    ("prefix", po::value(&prefix), "Prefix of reference sequence to use as group (default: 4; use -1 for full string)")
    ("input-fasta,f", po::value<std::string>(&fastaPath)->required(), "Path to (indexed) FASTA file")
    ("input-bam,i", po::value(&bamPaths)->composing()->required(), "Path to BAM(s)")
    ("no-group", po::bool_switch(&no_group_by_qname), "Do *not* group records by name")
    ("output-file,o", po::value<std::string>(&outputPath)->required(), "Path to output file");

    po::variables_map vm;
    po::store(po::parse_command_line(argc, argv, desc), vm);

    if(vm.count("help")) {
        std::cout << desc << '\n';
        return 0;
    }
    if(vm.count("version")) {
        std::cout << star_fit::FIT_STAR_VERSION << '\n';
        return 0;
    }

    po::notify(vm);

    faidx_t* fidx = fai_load(fastaPath.c_str());
    assert(fidx != NULL && "Failed to load FASTA index");

    size_t processed = 0;

    std::fstream out(outputPath, std::ios::out | std::ios::trunc | std::ios::binary);
    protoio::OstreamOutputStream rawOut(&out);
    protoio::GzipOutputStream zipOut(&rawOut);
    protoio::ZeroCopyOutputStream* outptr = &rawOut;
    if(endsWith(outputPath, ".gz"))
        outptr = &zipOut;

    for(const std::string& bamPath : bamPaths) {
        SamFile in(bamPath);
        SamRecord record;
        mutationio::MutationCount count;

        std::vector<std::string> targetBases(in.fp->header->n_targets);
        std::vector<int> targetLen(in.fp->header->n_targets);
        for(SamIterator it(in.fp, record.record), end; it != end; it++) {
            if(maxRecords > 0 && processed++ > maxRecords)
                break;

            const std::string qname = bam1_qname(*it);
            if((count.has_name() && count.name() != qname) || no_group_by_qname) {
                if(count.has_name() && count.partition_size())
                    writeDelimitedItem(*outptr, count);
                count.Clear();
            }

            if(!count.has_name()) {
                count.set_name(qname);
                count.set_distance(0.1);
            }
            std::string target_name = in.fp->header->target_name[(*it)->core.tid];
            if(targetBases[(*it)->core.tid].empty()) {
                char* ref = fai_fetch(fidx, target_name.c_str(), &targetLen[(*it)->core.tid]);
                assert(ref != nullptr && "Missing reference");
                targetBases[(*it)->core.tid] = ref;
                free(ref);
            }

            const std::string& ref = targetBases[(*it)->core.tid];

            // Assign a group based on germline prefix
            if(prefix >= 0)
                target_name.resize(prefix);

            mutationCountOfSequence(count, *it, ref, no_ambiguous, target_name, by_codon);
        }
        assert(count.has_name() && "Name not set");
        assert(count.partition_size() > 0 && "No partitions");
        writeDelimitedItem(*outptr, count);
    }

    fai_destroy(fidx);

    google::protobuf::ShutdownProtobufLibrary();

    return 0;
}
