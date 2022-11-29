#ifndef _H_LANDMARK_MATCHER_
#define _H_LANDMARK_MATCHER_

#include "KalmanFilter.hpp"
#include <memory>
#include <optional>
#include <vector>
#include <opencv2/core.hpp>

/*
    This class is used to handle the matching of landmarks observed at different
    times from different poses.
*/
class LandmarkMatcher {
public:
    /*
        Create a new LandmarkMatcher object

        \param Qt a 2x2 matrix representing sensor covariance
        \param minimum_observations the minimum number of times a landmark must
        be seen before it is confirmed as a real landmark
        \param distance_threshold how close two landmarks have to be to be
        considered the same one
        \param max_invalid_landmarks the maximum number of invalid landmarks to
        keep track of [NOT IMPLEMENTED!]
    */
    LandmarkMatcher(const cv::Mat& Qt, int minimum_observations=6, double distance_threshold=0.3, int max_invalid_landmarks=-1);
    LandmarkMatcher(const LandmarkMatcher&);
    LandmarkMatcher(LandmarkMatcher&&);
    LandmarkMatcher& operator = (const LandmarkMatcher&);
    LandmarkMatcher& operator = (LandmarkMatcher&&);
    
    /*
        To be called when a new landmark is observed to check if it matches with
        previously seen ones and to commit it to memory

        \param landmark the newly seen landmark
        \param H a function that returns the 2x2 Jacobian matrix of the measurement
        model matrix
        \param pose the robot's pose
        \return The matching landmark, if found 
    */
    std::optional<Landmark> observe(const Landmark& landmark, std::function<cv::Mat(double, double, double)> H, const cv::Vec3d& pose);
    /*
        \return A vector of all valid landmarks
    */
    std::vector<Landmark> get_map() const;
    /*
        \return A vector of all previously seen landmarks
    */
    std::vector<Landmark> get_full_map() const;

private:
    // Sensor covariance matrix
    cv::Mat Qt;
    // Minimum number of observations before a landmark is considered valid
    int minimum_observations;
    // Maximum distance for two landmarks to be considered the same
    double distance_threshold;
    // Maximum number of invalid landmarks to keep track of (NOT IMPLEMENTED!)
    int max_invalid_landmarks;
    // Extended Kalman Filters pertaining to all previously seen landmarks
    std::vector<std::pair<std::shared_ptr<KalmanFilter>, std::size_t>> filters;
};

#endif