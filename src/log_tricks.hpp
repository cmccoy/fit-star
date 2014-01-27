#ifndef LOG_TRICKS_H
#define LOG_TRICKS_H

#include <cmath>
#include <limits>


template<typename T>
T logSum(const T x, const T y)
{
    const static T maxValue = std::numeric_limits<T>::max();
    const static T logLimit = -maxValue / 100;
    const static T NATS = 400;

    const T temp = y - x;
    if(temp > NATS || x < logLimit)
        return y;
    if(temp < -NATS || y < logLimit)
        return x;
    if(temp < 0)
        return x + std::log1p(std::exp(temp));
    return y + std::log1p(std::exp(-temp));
}

#endif
