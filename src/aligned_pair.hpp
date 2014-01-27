#ifndef SEQUENCE_H
#define SEQUENCE_H

#include <string>
#include <Eigen/Dense>

namespace star_optim
{

struct Partition {
    std::string name;
    Eigen::Matrix4d substitutions;
};

struct AlignedPair {
    std::string name;
    std::vector<Partition> partitions;
    double distance;
};

}


#endif
