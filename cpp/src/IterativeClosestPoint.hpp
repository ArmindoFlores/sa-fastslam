#ifndef _H_ITERATIVE_CLOSEST_POINT_
#define _H_ITERATIVE_CLOSEST_POINT_

#include <limits>
#include <numeric>
#include <vector>
#include <opencv2/core.hpp>

struct ICPSolution {
    cv::Mat A;
    cv::Vec3d b;
};

ICPSolution iterative_closest_point(const std::vector<cv::Vec3d>& baseline, std::vector<cv::Vec3d> new_scan, double error=0.1, std::size_t max_iterations=100);

#endif