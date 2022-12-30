#ifndef _H_UTILS_
#define _H_UTILS_

#include <opencv2/core.hpp>

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

#endif