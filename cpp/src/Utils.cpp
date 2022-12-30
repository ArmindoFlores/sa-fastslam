#include "Utils.hpp"

cv::Vec2d polar_to_cartesian(const cv::Vec2d& vec)
{
    return polar_to_cartesian(vec(0), vec(1));
}

cv::Vec2d polar_to_cartesian(double theta, double r)
{
    cv::Vec2d result;
    result(0) = std::cos(theta);
    result(1) = std::sin(theta);
    if (r > 0.001)
        return r * result;
    return 5000 * result;
}