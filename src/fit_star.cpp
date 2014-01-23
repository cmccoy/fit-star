#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <unordered_set>
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
#include "star_tree_optimizer.hpp"
#include "aligned_pair.hpp"
#include "protobuf_util.hpp"

// Beagle
#include "libhmsbeagle/beagle.h"

// Bio++
#include <Bpp/Phyl/Model/RateDistributionFactory.h>
#include <Bpp/Phyl/Model/Nucleotide/GTR.h>
#include <Bpp/Phyl/Model/Nucleotide/HKY85.h>
#include <Bpp/Phyl/Model/Nucleotide/TN93.h>
#include <Bpp/Phyl/Model/Nucleotide/JCnuc.h>
#include <Bpp/Seq/Alphabet/AlphabetTools.h>

#include <cpplog.hpp>

#ifdef _OPENMP
#include <omp.h>
#endif

namespace po = boost::program_options;

cpplog::StdErrLogger logger;

void loadAlignedPairsFromFile(const std::string& file_path, std::vector<star_optim::AlignedPair>& dest)
{
    std::fstream in(file_path, std::ios::binary | std::ios::in);
    for(DelimitedProtocolBufferIterator<mutationio::MutationCount> it(in, true), end; it != end; it++) {
        star_optim::AlignedPair sequence;
        const mutationio::MutationCount& m = *it;
        if(m.has_name())
            sequence.name = m.name();
        sequence.partitions.resize(m.partition_size());
        for(size_t p = 0; p < sequence.partitions.size(); p++) {
            const mutationio::Partition& partition = m.partition(p);
            sequence.partitions[p].name = partition.name();
            sequence.partitions[p].substitutions.fill(0);
            assert(partition.substitution_size() == 16 && "Unexpected substitution count");
            for(size_t i = 0; i < 4; i++)
                for(size_t j = 0; j < 4; j++)
                    sequence.partitions[p].substitutions(i, j) = partition.substitution(4 * i + j);
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
        return p(new bpp::GTR(&bpp::AlphabetTools::DNA_ALPHABET));
    else if(upper == "HKY85")
        return p(new bpp::HKY85(&bpp::AlphabetTools::DNA_ALPHABET));
    else if(upper == "TN93")
        return p(new bpp::TN93(&bpp::AlphabetTools::DNA_ALPHABET));
    else if(upper == "JC")
        return p(new bpp::JCnuc(&bpp::AlphabetTools::DNA_ALPHABET));
    throw std::runtime_error("Unknown model: " + name);
}

std::unique_ptr<bpp::DiscreteDistribution> rateDistributionForName(const std::string& name)
{
    using p = std::unique_ptr<bpp::DiscreteDistribution>;
    std::string lower = name;
    std::transform(name.begin(), name.end(), lower.begin(), ::tolower);

    if(lower == "gamma") {
        LOG_INFO(logger) << "Gamma rate distribution with 4 categories.\n";
        bpp::RateDistributionFactory factory(4);
        p result(factory.createDiscreteDistribution("Gamma"));
        // Check
        std::vector<std::string> indNames = result->getIndependentParameters().getParameterNames();
        CHECK_EQUAL(logger, std::count(indNames.begin(), indNames.end(), "Gamma.beta"), 0) << "Beta parameter should be aliased.\n";
        CHECK_EQUAL(logger, std::count(indNames.begin(), indNames.end(), "Gamma.alpha"), 1) << "Alpha parameter should be estimated.\n";
        return result;
    } else if(lower == "constant") {
        LOG_INFO(logger) << "Constant rate distribution.\n";
        bpp::RateDistributionFactory factory(1);
        return p(factory.createDiscreteDistribution("Constant"));
    }
    throw std::runtime_error("Unknown model: " + name);
}

void writeResults(std::ostream& out,
                  const std::vector<std::unique_ptr<bpp::SubstitutionModel>>& models,
                  const std::vector<std::unique_ptr<bpp::DiscreteDistribution>>& rates,
                  const std::vector<star_optim::AlignedPair>& sequences,
                  const double logLikelihood,
                  const bool include_branch_lengths = true)
{
    Json::Value root;
    Json::Value partitionsNode(Json::arrayValue);

    assert(models.size() == rates.size() && "Different number of rates / models");

    auto f = [](double acc, const star_optim::AlignedPair & s) { return acc + s.distance; };
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
        for(size_t i = 0; i < r.getNumberOfCategories(); i++) {
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
        for(const star_optim::AlignedPair& sequence : sequences)
            blNode.append(sequence.distance);
        root["branchLengths"] = blNode;
    }

    out << root << '\n';
}

int main(const int argc, const char** argv)
{
    GOOGLE_PROTOBUF_VERIFY_VERSION;

    std::string outputPath, modelName = "GTR", rateDistName = "constant";
    std::vector<std::string> inputPaths;
    bool no_branch_lengths = false;
    double hky85KappaPrior = -1;
    double gammaAlpha = -1, threshold = 0.1;
    size_t maxRounds = 30;

    // command-line parsing
    po::options_description desc("Allowed options");
    desc.add_options()
    ("help,h", "Produce help message")
    ("version,v", "Show version")
    ("input-file,i", po::value(&inputPaths)->composing()->required(),
     "input file(s) - output of build-mutation-matrices [required]")
    ("output-file,o", po::value(&outputPath)->required(), "output file [required]")
    ("model,m", po::value(&modelName), "model [default: GTR]")
    ("rate-dist,r", po::value(&rateDistName), "rate distribution [default: constant]")
    ("kappa-prior,k", po::value(&hky85KappaPrior), "Prior on HKY85 kappa [default: None]")
    ("gamma-alpha,g", po::value(&gammaAlpha), "Fix gamma rate distribtion to value")
    ("threshold,t", po::value(&threshold), "Minimum improvement in an iteration to continue fitting")
    ("max-rounds,r", po::value(&maxRounds), "Maximum number of fitting rounds")
    ("no-branch-lengths", po::bool_switch(&no_branch_lengths), "*do not* include fit branch lengths in output");

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

    if(vm.count("kappa-prior") && modelName != "HKY85") {
        LOG_FATAL(logger) << "kappa prior is not compatible with model " << modelName << '\n';
        return 1;
    }
    if(vm.count("gamma-alpha") && rateDistName != "gamma") {
        LOG_FATAL(logger) << "Specifying gamma alpha requires `--rate-dist gamma` (got " << rateDistName << ")\n";
        return 1;
    }

    std::vector<star_optim::AlignedPair> sequences;
    for(const std::string& path : inputPaths) {
        LOG_INFO(logger) << "Loading from " << path << '\n';
        loadAlignedPairsFromFile(path, sequences);
    }

    LOG_INFO(logger) << sequences.size() << " sequences." << '\n';
    std::vector<std::unique_ptr<bpp::SubstitutionModel>> models;
    std::vector<std::unique_ptr<bpp::DiscreteDistribution>> rates;

    std::unordered_map<std::string, star_optim::PartitionModel> partitionModels;
    for(const star_optim::AlignedPair& sequence : sequences) {
        for(const star_optim::Partition& p : sequence.partitions) {
            if(partitionModels.count(p.name) == 0) {
                models.emplace_back(substitutionModelForName(modelName));
                rates.emplace_back(rateDistributionForName(rateDistName));
                partitionModels[p.name] = star_optim::PartitionModel { models.back().get(), rates.back().get() };
            }
        }
    }

    star_optim::StarTreeOptimizer optimizer(partitionModels, sequences);
    if(vm.count("kappa-prior"))
        optimizer.hky85KappaPrior(hky85KappaPrior);
    if(vm.count("gamma-alpha")) {
        for(auto &r : rates)
            r->setParameterValue("alpha", gammaAlpha);
        for(auto &p : optimizer.fitRates())
            p.second = false;
    } else if(rateDistName == "constant") {
        int i = 0;
        for(auto &p : optimizer.fitRates())
            if(i++ == 0) p.second = false;
    }
    if(vm.count("threshold"))
        optimizer.threshold(threshold);
    if(vm.count("max-rounds"))
        optimizer.maxRounds(maxRounds);

    size_t rounds = optimizer.optimize();
    LOG_INFO(logger) << "finished in " << rounds + 1 << " fitting rounds.\n";

    const double finalLike = optimizer.starLikelihood();
    LOG_INFO(logger) << "final log-like: " << finalLike << '\n';

    std::ofstream out(outputPath);
    writeResults(out, models, rates, sequences, finalLike, !no_branch_lengths);

    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}
