#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>
#include <google/protobuf/io/gzip_stream.h>

#include <boost/algorithm/string/predicate.hpp>
#include <boost/iostreams/filtering_streambuf.hpp>
#include <boost/iostreams/filter/gzip.hpp>
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
#include <Bpp/Numeric/Prob/InvariantMixedDiscreteDistribution.h>
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

// STL
#include <algorithm>
#include <cassert>
#include <fstream>
#include <iostream>
#include <memory>
#include <unordered_map>
#include <vector>
#include "mutationio.pb.h"

using namespace fit_star;
namespace po = boost::program_options;

cpplog::StdErrLogger logger;

void loadAlignedPairsFromFile(const std::string& file_path, std::vector<fit_star::AlignedPair>& dest)
{
    std::fstream in(file_path, std::ios::binary | std::ios::in);
    for(DelimitedProtocolBufferIterator<mutationio::MutationCount> it(in, true), end; it != end; it++) {
        fit_star::AlignedPair sequence;
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

std::unique_ptr<bpp::DiscreteDistribution> rateDistributionForName(const std::string& name, bool includeInvariant = false)
{
    using p = std::unique_ptr<bpp::DiscreteDistribution>;
    std::string lower = name;
    std::transform(name.begin(), name.end(), lower.begin(), ::tolower);

    p result;
    if(lower == "gamma") {
        LOG_INFO(logger) << "Gamma rate distribution with 4 categories.\n";
        bpp::RateDistributionFactory factory(4);
        result.reset(factory.createDiscreteDistribution("Gamma"));
        // Check
        std::vector<std::string> indNames = result->getIndependentParameters().getParameterNames();
        if(std::count(indNames.begin(), indNames.end(), "Gamma.beta"))
            result->aliasParameters("Gamma.alpha", "Gamma.beta");
        indNames = result->getIndependentParameters().getParameterNames();
        CHECK_EQUAL(logger, std::count(indNames.begin(), indNames.end(), "Gamma.beta"), 0) << "Beta parameter should be aliased.\n";
        CHECK_EQUAL(logger, std::count(indNames.begin(), indNames.end(), "Gamma.alpha"), 1) << "Alpha parameter should be estimated.\n";
    } else if(lower == "constant") {
        LOG_INFO(logger) << "Constant rate distribution.\n";
        bpp::RateDistributionFactory factory(1);
        result.reset(factory.createDiscreteDistribution("Constant"));
    } else {
        throw std::runtime_error("Unknown model: " + name);
    }

    if(includeInvariant) {
        result.reset(new bpp::InvariantMixedDiscreteDistribution(result.release(), 0.5));
    }
    return result;
}

void writeResults(std::ostream& out,
                  const std::unordered_map<std::string, fit_star::PartitionModel>& models,
                  const std::vector<fit_star::AlignedPair>& sequences,
                  const double logLikelihood,
                  const size_t nRounds = -1,
                  const bool include_branch_lengths = true)
{
    Json::Value root;
    Json::Value partitionsNode(Json::arrayValue);

    auto f = [](double acc, const fit_star::AlignedPair & s) { return acc + s.distance; };
    const double meanBranchLength = std::accumulate(sequences.begin(), sequences.end(), 0.0, f) / sequences.size();
    root["meanBranchLength"] = meanBranchLength;
    root["nRounds"] = static_cast<int>(nRounds);

    Json::Value paramNode(Json::objectValue);
    bpp::ParameterList pl;
    for(auto it = models.begin(), end = models.end(); it != end; it++) {
        pl.includeParameters(it->second.model->getIndependentParameters());
        pl.includeParameters(it->second.rateDist->getIndependentParameters());
    }
    for(size_t i = 0; i < pl.size(); i++) {
        paramNode[pl[i].getName()] = pl[i].getValue();
    }
    root["independentParameters"] = paramNode;
    root["degreesOfFreedom"] = static_cast<int>(pl.size() + sequences.size());
    paramNode = Json::Value(Json::objectValue);
    for(auto it = models.begin(), end = models.end(); it != end; it++) {
        pl.includeParameters(it->second.model->getParameters());
        pl.includeParameters(it->second.rateDist->getParameters());
    }
    for(size_t i = 0; i < pl.size(); i++) {
        paramNode[pl[i].getName()] = pl[i].getValue();
    }
    root["parameters"] = paramNode;

    using p = std::pair<std::string, fit_star::PartitionModel>;
    std::vector<p> parts(models.begin(), models.end());
    std::sort(parts.begin(), parts.end(), [](const p& x, const p& y) {
        return x.first < y.first;
    });

    for(const p& part : parts) {
        Json::Value modelNode(Json::objectValue);
        modelNode["partition"] = part.first;
        Json::Value rateNode(Json::objectValue);
        Json::Value parameterNode(Json::objectValue);
        Json::Value piNode(Json::arrayValue);
        std::unique_ptr<bpp::SubstitutionModel> model(part.second.model->clone());
        std::unique_ptr<bpp::DiscreteDistribution> r(part.second.rateDist->clone());

        model->setNamespace(model->getName() + ".");
        bpp::ParameterList p = model->getParameters();
        for(size_t i = 0; i < p.size(); i++) {
            parameterNode[p[i].getName()] = p[i].getValue();
        }

        modelNode["modelName"] = model->getName();
        modelNode["parameters"] = parameterNode;

        rateNode["distribution"] = r->getName();
        r->setNamespace(r->getName() + ".");
        p = r->getParameters();
        //rateNode["name"] = rates.getName();
        for(size_t i = 0; i < p.size(); i++) {
            rateNode[p[i].getName()] = p[i].getValue();
        }
        Json::Value rateRates(Json::arrayValue);
        Json::Value rateProbs(Json::arrayValue);
        for(size_t i = 0; i < r->getNumberOfCategories(); i++) {
            rateRates.append(r->getCategory(i));
            rateProbs.append(r->getProbability(i));
        }
        rateNode["rates"] = rateRates;
        rateNode["probabilities"] = rateProbs;
        modelNode["rate"] = rateNode;

        for(size_t i = 0; i < model->getNumberOfStates(); i++) {
            piNode.append(model->freq(i));
        }
        modelNode["pi"] = piNode;
        // Matrices
        Json::Value qNode(Json::arrayValue);
        Json::Value sNode(Json::arrayValue);
        Json::Value pNode(Json::arrayValue);

        const auto& pt = model->getPij_t(meanBranchLength);
        const auto& sij = model->getExchangeabilityMatrix();
        for(size_t i = 0; i < model->getNumberOfStates(); i++) {
            for(size_t j = 0; j < model->getNumberOfStates(); j++) {
                qNode.append(model->Qij(i, j));
                sNode.append(sij(i, j));
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
        for(const fit_star::AlignedPair& sequence : sequences)
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
    bool noBranchLengths = false, shareRates = false, shareModels = false, addRate = false, invariant = false, noFixRootFreqs = false;
    double hky85KappaPrior = -1;
    double gammaAlpha = -1, threshold = 0.1, maxTimePerRound = 30 * 60;
    size_t maxRounds = 100, maxIterations = 1000;

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
    ("invariant", po::bool_switch(&invariant), "Include an invariant category")
    ("kappa-prior,k", po::value(&hky85KappaPrior), "Prior on HKY85 kappa [default: None]")
    ("gamma-alpha,g", po::value(&gammaAlpha), "Fix gamma rate distribtion to value")
    ("threshold,t", po::value(&threshold), "Minimum improvement in an iteration to continue fitting")
    ("max-rounds,r", po::value(&maxRounds), "Maximum number of fitting rounds")
    ("max-iterations", po::value(&maxIterations), "Maximum number of iterations per round")
    ("max-time", po::value(&maxTimePerRound), "Maximum time (s) per fitting round (default: 30 minutes)")
    ("share-rates", po::bool_switch(&shareRates), "Share rate distribution")
    ("no-fix-root-frequencies", po::bool_switch(&noFixRootFreqs), "Do *not* fix root frequencies")
    ("add-rates", po::bool_switch(&addRate), "Add relative rate to secondary mutation models")
    ("share-models", po::bool_switch(&shareModels), "Share substitution model")
    ("no-branch-lengths", po::bool_switch(&noBranchLengths), "*do not* include fit branch lengths in output");

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

    std::vector<fit_star::AlignedPair> sequences;
    for(const std::string& path : inputPaths) {
        LOG_INFO(logger) << "Loading from " << path << '\n';
        loadAlignedPairsFromFile(path, sequences);
    }

    LOG_INFO(logger) << sequences.size() << " sequences." << '\n';
    std::vector<std::unique_ptr<bpp::SubstitutionModel>> models;
    std::vector<std::unique_ptr<bpp::DiscreteDistribution>> rates;

    std::unordered_map<std::string, fit_star::PartitionModel> partitionModels;
    for(const fit_star::AlignedPair& sequence : sequences) {
        for(const fit_star::Partition& p : sequence.partitions) {
            if(partitionModels.count(p.name) == 0) {
                if(!shareModels || models.size() == 0) {
                    models.emplace_back(substitutionModelForName(modelName));
                    models.back()->setNamespace(p.name + '.' +  models.back()->getNamespace());
                }
                if(!shareRates || rates.size() == 0) {
                    rates.emplace_back(rateDistributionForName(rateDistName, invariant));
                    rates.back()->setNamespace(p.name + '.' + rates.back()->getNamespace());
                }
                partitionModels[p.name] = fit_star::PartitionModel { models.back().get(), rates.back().get() };
            }
        }
    }

    fit_star::StarTreeOptimizer optimizer(partitionModels, sequences);
    optimizer.fixRootFrequencies(!noFixRootFreqs);
    if(vm.count("kappa-prior"))
        optimizer.hky85KappaPrior(hky85KappaPrior);
    if(vm.count("gamma-alpha")) {
        for(auto& r : rates)
            r->setParameterValue("alpha", gammaAlpha);
        for(auto& p : optimizer.fitRates())
            p.second = false;
    } else if(rateDistName == "constant") {
        int i = 0;
        for(auto& p : optimizer.fitRates())
            if(i++ == 0) p.second = false;
    }
    if(vm.count("threshold"))
        optimizer.threshold(threshold);
    optimizer.maxRounds(maxRounds);
    optimizer.maxIterations(maxIterations);
    optimizer.maxTime(maxTimePerRound);

    if(addRate) {
        for(size_t i = 1; i < models.size(); i++)
            models[i]->addRateParameter();
    }

    size_t rounds = optimizer.optimize();
    LOG_INFO(logger) << "finished in " << rounds + 1 << " fitting rounds.\n";

    const double finalLike = optimizer.starLikelihood();
    LOG_INFO(logger) << "final log-like: " << finalLike << '\n';

    std::ofstream file(outputPath, std::ios_base::out | std::ios_base::binary);
    boost::iostreams::filtering_streambuf<boost::iostreams::output> outBuf;

    if(boost::algorithm::ends_with(outputPath, ".gz"))
        outBuf.push(boost::iostreams::gzip_compressor());
    outBuf.push(file);
    std::ostream outStream(&outBuf);
    writeResults(outStream, partitionModels, sequences, finalLike, rounds, !noBranchLengths);

    google::protobuf::ShutdownProtobufLibrary();
    return 0;
}
