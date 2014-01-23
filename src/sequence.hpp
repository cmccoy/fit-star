#ifndef SEQUENCE_H
#define SEQUENCE_H

#include <string>
#include <Eigen/Dense>

struct Sequence {
    std::string name;
    std::vector<std::string> partitionNames;
    std::vector<Eigen::Matrix4d> substitutions;
    double distance;
};


#endif
