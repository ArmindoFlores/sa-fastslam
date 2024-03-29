#include "Landmark.hpp"

#include <cmath>
#include "Utils.hpp"

Landmark::Landmark(double r, double theta) : r(python_fmod(r, 2*M_PI)), theta(theta) {}
Landmark::Landmark(double a, double b, double c)
{
    double a_squared = a * a, b_squared = b * b;

    if (c != 0)
        theta = std::atan2(-b * c / (a_squared + b_squared), -a * c / (a_squared + b_squared));
    else
        theta = std::atan2(-b / (a_squared + b_squared), -a / (a_squared + b_squared));
    
    r = std::abs(c) / std::sqrt(a_squared + b_squared);
    theta = python_fmod(theta, 2*M_PI);
}
Landmark::Landmark(Landmark&& other)
{
    r = other.r;
    theta = other.theta;
}
Landmark& Landmark::operator = (const Landmark& l)
{
    r = l.r;
    theta = l.theta;
    return *this;
}
Landmark& Landmark::operator = (Landmark&& l)
{
    r = l.r;
    theta = l.theta;
    return *this;
}

void Landmark::update_parameters(double r, double theta)
{
    this->r = r;
    if (this->r < 0) {
        this->r *= -1;
        theta += M_PI;
    }
    this->theta = python_fmod(theta, 2*M_PI);
}

cv::Vec2d Landmark::get_parameters() const
{
    return cv::Vec2d {r, theta};
}

cv::Vec2d Landmark::get_parameters(const cv::Vec3d& pose) const
{
    cv::Vec2d computed_params {
        r - pose[0] * std::cos(theta) - pose[1] * std::sin(theta), 
        theta - pose[2]
    };
    if (computed_params[0] < 0) {
        computed_params[0] *= -1;
        computed_params[1] += M_PI;
    }
    computed_params[1] = python_fmod(computed_params[1], 2 * M_PI);
    return computed_params;
}

cv::Vec2d Landmark::closest_point(const cv::Vec2d& position) const
{
    return closest_point(position(0), position(1));
}

cv::Vec2d Landmark::closest_point(double x, double y) const
{
    cv::Vec2d result;
    cv::Vec3d params = equation();

    double a = params(0), b = params(1), c = params(2);
    double a_squared = a * a, b_squared = b * b;
    result(0) = (b_squared * x - a * b * y - a * c) / (a_squared + b_squared);
    result(1) = (-b * a * x + a_squared * y - b * c) / (a_squared + b_squared);
    return result;
}

cv::Vec3d Landmark::equation() const
{
    cv::Vec3d result;
    int sign = (theta >= 0 && theta < M_PI) ? 1 : -1;
    if (std::abs(std::abs(theta) - M_PI) >= 0.0001) {
        result(1) = -1;
        result(0) = result(1) / std::tan(theta);
        result(2) = sign * r * std::sqrt(1 + result(0)*result(0));
    }
    else {
        result(0) = -1;
        result(1) = result(0) * std::tan(theta);
        result(2) = sign * r * std::sqrt(1 + result(1)*result(1));
    }
    return result;
}

std::ostream& operator << (std::ostream& os, const Landmark& l)
{
    std::cout << "<Landmark r=" << l.r << " \u03B8=" << l.theta << ">";
    return os;
}
