#ifndef _H_LANDMARK_MATCHER_
#define _H_LANDMARK_MATCHER_

#include "KalmanFilter.hpp"
#include <memory>
#include <optional>
#include <vector>
#include <opencv2/core.hpp>

class LandmarkMatcher {
public:
    LandmarkMatcher(cv::Mat Qt, int minimum_observations=6, double distance_threshold=0.3, int max_invalid_landmarks=-1);
    LandmarkMatcher(const LandmarkMatcher&);
    LandmarkMatcher(LandmarkMatcher&&);
    LandmarkMatcher& operator = (const LandmarkMatcher&);
    LandmarkMatcher& operator = (LandmarkMatcher&&);
    
    std::optional<Landmark> observe(const Landmark& landmark, std::function<cv::Mat(double, double, double)> H, cv::Vec3d pose);
    std::vector<Landmark> get_map() const;
    std::vector<Landmark> get_full_map() const;

private:
    cv::Mat Qt;
    int minimum_observations;
    double distance_threshold;
    int max_invalid_landmarks;
    std::vector<std::pair<std::shared_ptr<KalmanFilter>, std::size_t>> filters;
};

#endif