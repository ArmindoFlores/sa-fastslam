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

int sgn(double val) {
    return (0.0 < val) - (val < 0.0);
}

double python_fmod(double x1, double x2)
{
    double sign1 = sgn(x1);
    double sign2 = sgn(x2);
    if (sign1 == sign2)
        return sign2 * std::fmod(sign1 * x1, sign2 * x2);
    return x2 - sign2 * std::fmod(sign1 * x1, sign2 * x2);
}