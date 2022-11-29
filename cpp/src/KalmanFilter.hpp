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
    /*
        Construct a new Kalman Filter

        \param landmark the landmark to keep track of
        \param covariance the initial 2x2 covariance matrix
        \param Qt the 2x2 sensor covariance matrix
    */
    KalmanFilter(const Landmark& landmark, const cv::Mat& covariance, const cv::Mat& Qt);
    KalmanFilter(const KalmanFilter&);
    KalmanFilter(KalmanFilter&&);

    /*
        \return The 2x2 covariance matrix associated with this landmark
    */
    const cv::Mat& get_covariance() const;
    /*
        \return The 2x2 sensor covariance matrix
    */
    const cv::Mat& get_Qt() const;
    /*
        \return A reference to the underlying landmark
    */
    Landmark& get_landmark();
    /*
        \return A copy of the underlying landmark
    */
    Landmark view_landmark() const;
    /*
        Update the covariance matrix and parameter estimations

        \param measurement the observed parameters of the landmark
        \param H the 2x2 Jacobian matrix of the measurement model matrix 
    */
    void update(const cv::Vec2d& measurement, const cv::Mat& H);

    friend std::ostream& operator << (std::ostream&, const KalmanFilter&);

private:
    Landmark landmark;
    cv::Mat covariance, Qt;
};

#endif