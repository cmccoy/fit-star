#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <vector>
#include "mutationio.pb.h"

#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/gzip_stream.h>

#include <boost/program_options.hpp>
#include <json/json.h>
#include <json/value.h>

#include "fit_star_config.h"
#include "star_optim.hpp"
#include "sequence.hpp"

// Beagle
#include "libhmsbeagle/beagle.h"

// Bio++
#include <Bpp/Numeric/Prob/ConstantDistribution.h>
#include <Bpp/Numeric/Prob/GammaDiscreteDistribution.h>
#include <Bpp/Phyl/Model/GTR.h>
#include <Bpp/Phyl/Model/HKY85.h>
#include <Bpp/Phyl/Model/TN93.h>
#include <Bpp/Phyl/Model/JCnuc.h>
#include <Bpp/Seq/Alphabet/DNA.h>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace po = boost::program_options;

const bpp::DNA DNA;

void loadSequencesFromFile(const std::string& file_path, std::vector<Sequence>& dest)
{
    std::fstream in(file_path, std::ios::in | std::ios::binary);
    std::clog << "Loading from " << file_path << '\n';
    assert(in.good() && "Input stream is not good.");
    google::protobuf::io::IstreamInputStream raw_in(&in);
    google::protobuf::io::GzipInputStream zip_in(&raw_in);

    while(true) {
        google::protobuf::io::CodedInputStream coded_in(&zip_in);
        uint32_t size = 0;
        bool success = false;
        success = coded_in.ReadVarint32(&size);
        if(!success) break;
        mutationio::MutationCount m;
        std::string s;
        coded_in.ReadString(&s, size);
        success = m.ParseFromString(s);
        assert(success && "Failed to parse");
        Sequence sequence;
        if(m.has_name())
            sequence.name = m.name();
        sequence.substitutions.resize(m.partition_size());
        for(size_t p = 0; p < sequence.substitutions.size(); p++) {
            sequence.substitutions[p].fill(0);
            const mutationio::Partition& partition = m.partition(p);
            assert(partition.substitution_size() == 16 && "Unexpected substitution count");
            for(size_t i = 0; i < 4; i++)
                for(size_t j = 0; j < 4; j++)
                    sequence.substitutions[p](i, j) = partition.substitution(4 * i + j);
        }

        // TODO: use mutation count
        //double d = 1 - sequence.substitutions.diagonal().sum() / sequence.substitutions.sum();
        //if(d == 0)
        //d = 1e-3;
        sequence.distance = 0.1;

        dest.push_back(std::move(sequence));
    }
}

std::unique_ptr<bpp::SubstitutionModel> substitutionModelForName(const std::string& name)
{
    using p = std::unique_ptr<bpp::SubstitutionModel>;
    std::string upper = name;
    std::transform(name.begin(), name.end(), upper.begin(), ::toupper);
    if(upper == "GTR")
        return p(new bpp::GTR(&DNA));
    else if(upper == "HKY85")
        return p(new bpp::HKY85(&DNA));
    else if(upper == "TN93")
        return p(new bpp::TN93(&DNA));
    else if(upper == "JC")
        return p(new bpp::JCnuc(&DNA));
    throw std::runtime_error("Unknown model: " + name);
}

std::unique_ptr<bpp::DiscreteDistribution> rateDistributionForName(const std::string& name)
{
    using p = std::unique_ptr<bpp::DiscreteDistribution>;
    std::string lower = name;
    std::transform(name.begin(), name.end(), lower.begin(), ::tolower);

    if(lower == "gamma")
        return p(new bpp::GammaDiscreteDistribution(4));
    else if(lower == "constant")
        return p(new bpp::ConstantDistribution(1.0));
    throw std::runtime_error("Unknown model: " + name);
}

void writeResults(std::ostream& out,
                  const std::vector<std::unique_ptr<bpp::SubstitutionModel>>& models,
                  const std::vector<std::unique_ptr<bpp::DiscreteDistribution>>& rates,
                  const std::vector<Sequence>& sequences,
                  const double logLikelihood,
                  const bool include_branch_lengths = true)
{
    Json::Value root;
    Json::Value partitionsNode(Json::arrayValue);

    assert(models.size() == rates.size() && "Different number of rates / models");

    auto f = [](double acc, const Sequence & s) { return acc + s.distance; };
    const double meanBranchLength = std::accumulate(sequences.begin(), sequences.end(), 0.0, f) / sequences.size();
    root["meanBranchLength"] = meanBranchLength;

    for(size_t i = 0; i < models.size(); i++) {
        Json::Value modelNode(Json::objectValue);
        modelNode["partitionIndex"] = static_cast<int>(i);
        Json::Value rateNode(Json::objectValue);
        Json::Value parameterNode(Json::objectValue);
        Json::Value piNode(Json::arrayValue);
        const bpp::SubstitutionModel& model = *models[i];
        const bpp::DiscreteDistribution& r = *rates[i];
        bpp::ParameterList p = model.getParameters();
        for(size_t i = 0; i < p.size(); i++) {
            parameterNode[p[i].getName()] = p[i].getValue();
        }

        modelNode["name"] = model.getName();
        modelNode["parameters"] = parameterNode;

        //rateNode["name"] = rates.getName();
        p = r.getParameters();
        //rateNode["name"] = rates.getName();
        for(size_t i = 0; i < p.size(); i++) {
            rateNode[p[i].getName()] = p[i].getValue();
        }
        Json::Value rateRates(Json::arrayValue);
        Json::Value rateProbs(Json::arrayValue);
        for(unsigned int i = 0; i < r.getNumberOfCategories(); i++) {
            rateRates.append(r.getCategory(i));
            rateProbs.append(r.getProbability(i));
        }
        rateNode["rates"] = rateRates;
        rateNode["probabilities"] = rateProbs;
        modelNode["rate"] = rateNode;

        for(size_t i = 0; i < model.getNumberOfStates(); i++) {
            piNode.append(model.freq(i));
        }
        modelNode["pi"] = piNode;
        // Matrices
        Json::Value qNode(Json::arrayValue);
        //Json::Value sNode(Json::arrayValue);
        Json::Value pNode(Json::arrayValue);

        const auto& pt = model.getPij_t(meanBranchLength);
        //const auto& sij = model.getExchangeabilityMatrix();
        for(size_t i = 0; i < model.getNumberOfStates(); i++) {
            for(size_t j = 0; j < model.getNumberOfStates(); j++) {
                qNode.append(model.Qij(i, j));
                //sNode.append(sij(i, j));
                pNode.append(pt(static_cast<unsigned int>(i), static_cast<unsigned int>(j)));
            }
        }
        modelNode["Q"] = qNode;
        //modelNode["S"] = sNode;
        modelNode["Pmean"] = pNode;
        partitionsNode.append(modelNode);
    }

    root["partitions"] = partitionsNode;

    root["logLikelihood"] = logLikelihood;
    if(include_branch_lengths) {
        Json::Value blNode(Json::arrayValue);
        for(const Sequence& sequence : sequences)
            blNode.append(sequence.distance);
        root["branchLengths"] = blNode;
    }

    out << root << '\n';
}

int main(const int argc, const char** argv)
{
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    std::string output_path, model_name = "GTR", rate_dist_name = "Constant";
    std::vector<std::string> input_paths;
    bool no_branch_lengths = false;
    double hky85KappaPrior = -1;

    // command-line parsing
    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "Produce help message")
    ("version,v", "Show version")
    ("input-file,i", po::value(&input_paths)->composing()->required(),
     "input file(s) - output of build-mutation-matrices [required]")
    ("output-file,o", po::value(&output_path)->required(), "output file [required]")
    ("model,m", po::value(&model_name), "model [default: GTR]")
    ("rate-dist,r", po::value(&rate_dist_name), "rate distribution [default: constant]")
    ("kappa-prior,k", po::value(&hky85KappaPrior), "Prior on HKY85 kappa [default: None]")
    ("no-branch-lengths", po::bool_switch(&no_branch_lengths), "*do not* include fit branch lengths in output");

    po::variables_map vm;
    po::store(po::command_line_parser(argc, argv).
              options(desc).run(), vm);

    if(vm.count("help")) {
        std::cout << desc << '\n';
        return 0;
    }
    if(vm.count("version")) {
        std::cout << star_fit::GTR_FIT_VERSION << '\n';
        return 0;
    }

    if(vm.count("kappa-prior") && model_name != "HKY85") {
        std::clog << "kappa prior is not compatible with model " << model_name << '\n';
        return 1;
    }

    po::notify(vm);

    std::vector<Sequence> sequences;
    for(const std::string& path : input_paths) {
        loadSequencesFromFile(path, sequences);
    }

    const size_t nPartitions = sequences[0].substitutions.size();
    for(const Sequence& sequence : sequences) {
        assert(nPartitions == sequence.substitutions.size() && "Varying number of partitions");
    }

    std::clog << sequences.size() << " sequences." << '\n';

    std::vector<std::unique_ptr<bpp::SubstitutionModel>> models;
    std::vector<std::unique_ptr<bpp::DiscreteDistribution>> rates;
    for(size_t i = 0; i < nPartitions; i++) {
        models.emplace_back(substitutionModelForName(model_name));
        rates.emplace_back(rateDistributionForName(rate_dist_name));
    }
    std::vector<std::vector<int>> beagleInstances(1);

#ifdef _OPENMP
    beagleInstances.resize(omp_get_max_threads());
#endif
    for(std::vector<int>& v : beagleInstances) {
        for(size_t i = 0; i < models.size(); i++) {
            v.push_back(star_optim::createBeagleInstance(*models[i], *rates[i]));
        }
    }

    star_optim::optimize(beagleInstances, models, rates, sequences, hky85KappaPrior);

    const double finalLike = star_optim::starLikelihood(beagleInstances, models, rates, sequences, hky85KappaPrior);
    std::clog << "final log-like: " << finalLike << '\n';

    std::ofstream out(output_path);
    writeResults(out, models, rates, sequences, finalLike, !no_branch_lengths);

    for(const std::vector<int>& v : beagleInstances) {
        for(const int i : v)
            beagleFinalizeInstance(i);
    }

    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}
