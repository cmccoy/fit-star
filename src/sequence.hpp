#ifndef SEQUENCE_H
#define SEQUENCE_H

#include <string>
#include <Eigen/Dense>

struct Sequence
{
    std::string name;
    Eigen::Matrix4d transitions;
    double distance;
};


#endif