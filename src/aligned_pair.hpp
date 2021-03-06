#ifndef SEQUENCE_H
#define SEQUENCE_H

#include <string>
#include <Eigen/Dense>

namespace fit_star
{

struct Partition {
    std::string name;
    Eigen::MatrixXd substitutions;
};

struct AlignedPair {
    std::string name;
    std::vector<Partition> partitions;
    double distance;
};

}


#endif
