#include "sam.h"
#include "faidx.h"

#include <cassert>
#include <cstdio>
#include <iostream>

#include <string>
#include <vector>

#include <boost/program_options.hpp>

#include <Bpp/Seq/Alphabet/AlphabetTools.h>
#include <Bpp/Seq/Alphabet/WordAlphabet.h>

#include <Eigen/Dense>

#include "fit_star_config.h"
#include "sam_util.hpp"

namespace po = boost::program_options;
using namespace fit_star;

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

size_t offset(const size_t pos, const size_t sequenceFrame, const size_t wordSize) {
    if(pos == 0)
        return sequenceFrame;
    const size_t rem = pos % wordSize;
    if(rem == sequenceFrame)
        return 0;
    else if(rem < sequenceFrame)
        return sequenceFrame - rem;
    else
        return wordSize + sequenceFrame - rem;
}

Eigen::MatrixXd mutationMatrixOfSequence(const bam1_t* b,
                                         const std::string& ref,
                                         const size_t wordSize,
                                         const size_t sequenceFrame,
                                         const bpp::WordAlphabet& alphabet)
{
    assert(b != nullptr && "null bam record");

    Eigen::MatrixXd mutations = Eigen::MatrixXd::Zero(alphabet.getSize(), alphabet.getSize());

    const uint32_t* cigar = bam1_cigar(b);
    const uint8_t* seq = bam1_seq(b);

    // Query index, reference index
    int32_t qi = 0, ri = b->core.pos;
    const std::vector<char> bases{'A', 'C', 'G', 'T', 'N'};

    // Iterate over cigar
    for(uint32_t cidx = 0; cidx < b->core.n_cigar; cidx++) {
        const uint32_t clen = bam_cigar_oplen(cigar[cidx]);
        const uint32_t consumes = bam_cigar_type(cigar[cidx]); // bit 1: consume query; bit 2: consume reference
        if((consumes & 0x3) == 0x3 && clen >= wordSize) { // Reference and query
            for(size_t i = offset(qi, sequenceFrame, wordSize); i + wordSize < clen; i += wordSize) {
                std::string qword, rword;
                for(size_t j = 0; j < wordSize; j++) {
                    qword += bases[nt16ToIdx(bam1_seqi(seq, qi + i + j))];
                    rword += ref[ri + i + j];
                }

                mutations(alphabet.charToInt(rword), alphabet.charToInt(qword)) += 1;
            }
        }
        if(consumes & 0x1) // Consumes query
            qi += clen;
        else if(consumes & 0x2) // Consumes reference
            ri += clen;
    }

    return mutations;
}


int usage(po::options_description& desc)
{
    std::cerr << "Usage: fit-kgt1 [options] <ref.fasta> <in.bam> <out.bin>\n";
    std::cerr << desc << '\n';
    return 1;
}


int main(int argc, char* argv[])
{

    std::string fastaPath, outputPath;
    std::vector<std::string> bamPaths;

    size_t maxRecords = 0;
    int prefix = 4;
    size_t wordSize = 2, mutatedPosition = 0;

    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "Produce help message")
    ("version,v", "Print version")
    ("word-size,k", po::value(&wordSize), "Word size")
    ("position,p", po::value(&mutatedPosition), "Mutated position (0-based)")
    ("prefix", po::value(&prefix), "Prefix of reference sequence to use as group (default: 4; use -1 for full string)")
    ("input-fasta,f", po::value<std::string>(&fastaPath)->required(), "Path to (indexed) FASTA file")
    ("input-bam,i", po::value(&bamPaths)->composing()->required(), "Path to BAM(s)")
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

    FastaIndex fidx(fastaPath);
    assert(fidx.index != nullptr && "Failed to load FASTA index");

    std::vector<const bpp::Alphabet*> alphabets(wordSize);
    for(size_t i = 0; i < wordSize; i++)
        alphabets[i] = &bpp::AlphabetTools::DNA_ALPHABET;
    const bpp::WordAlphabet alphabet(alphabets);

    for(const std::string& bamPath : bamPaths) {
        SamFile in(bamPath);
        SamRecord record;

        std::vector<std::string> targetBases(in.fp->header->n_targets);
        std::vector<int> targetLen(in.fp->header->n_targets);
        for(SamIterator it(in.fp, record.record), end; it != end; it++) {
            const std::string qname = bam1_qname(*it);
            std::string targetName = in.fp->header->target_name[(*it)->core.tid];

            if(targetBases[(*it)->core.tid].empty()) {
                char* ref = fai_fetch(fidx.index, targetName.c_str(), &targetLen[(*it)->core.tid]);
                assert(ref != nullptr && "Missing reference");
                targetBases[(*it)->core.tid] = ref;
                free(ref);
            }

            const std::string& ref = targetBases[(*it)->core.tid];
            const size_t sequenceFrame = (*it)->core.pos % wordSize;

            const Eigen::MatrixXd mutations = mutationMatrixOfSequence(*it, ref, wordSize, sequenceFrame, alphabet);
            std::cerr << mutations << '\n';
            break;
        }

    }

    return 0;
}

