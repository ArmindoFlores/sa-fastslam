#ifndef _H_LANDMARK_EXTRACTOR_
#define _H_LANDMARK_EXTRACTOR_

#include "Landmark.hpp"
#include <opencv2/core.hpp>
#include <vector>

enum class ExtractionAlgorithm {
    RANSAC,
    HOUGH
};

/*
    Convert coordinates in a polar system to a cartesian one

    \param pos the position in polar coordinates
    \return The same position in a cartesian coordinate system
*/
cv::Vec2d polar_to_cartesian(const cv::Vec2d& pos);
/*
    Convert coordinates in a polar system to a cartesian one

    \param theta the angle of the point
    \param r the distance of the point to the origin
    \return The same position in a cartesian coordinate system
*/
cv::Vec2d polar_to_cartesian(double theta,  double r);

/*
    A structure representing a line obtained from the RANSAC algorithm
*/
struct RANSACResult {
    // Parameter from a line equation: ax + by + c = 0
    double a, b, c;
    // r^2 parameter from the linear regression the yielded this line
    double r2;
};

/*
    A function that detects lines in a given set of points using the RANSAC
    outlier removing algorithm.

    \param ranges a vector of distances to the origin, equally spaced by an
    angle of 360 / ranges.size() degrees
    \param N maximum number of iteratiosn
    \param C minimum number of points in each valid line
    \param X maximum distance from a point to the line to be valid
    \param D angle (degrees) from which to choose the first sample
    \param S number of points in the initial sample 
*/
std::vector<RANSACResult> RANSAC(
    const std::vector<double> ranges,
    std::size_t N, 
    std::size_t C, 
    double X, 
    double D, 
    std::size_t S
);

/*
    This functions returns a list of found landmarks from a laser scan

    \param points the vector of ranges from a laser scan
    \param algo the algorithm to use
    \return The vector of found landmarks
*/
std::vector<Landmark> extract_landmarks(const std::vector<double>& points, ExtractionAlgorithm algo=ExtractionAlgorithm::RANSAC);

#endif