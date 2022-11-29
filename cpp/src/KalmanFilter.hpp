#ifndef _H_KALMAN_FILTER_
#define _H_KALMAN_FILTER_

#include "Landmark.hpp"
#include <iostream>
#include <opencv2/core.hpp>

/*
    This class implements an Extended Kalman Filter associated with a Landmark
*/
class KalmanFilter {
public:
    KalmanFilter(const Landmark& landmark, cv::Mat covariance, cv::Mat Qt);
    KalmanFilter(const KalmanFilter&);
    KalmanFilter(KalmanFilter&&);

    const cv::Mat& get_covariance() const;
    const cv::Mat& get_Qt() const;
    Landmark& get_landmark();
    Landmark view_landmark() const;
    void update(const cv::Vec2d& measurement, const cv::Mat& H);

    friend std::ostream& operator << (std::ostream&, const KalmanFilter&);

private:
    Landmark landmark;
    cv::Mat covariance, Qt;
};

#endif