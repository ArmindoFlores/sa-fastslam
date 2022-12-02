#include "KalmanFilter.hpp"

KalmanFilter::KalmanFilter(const Landmark& landmark, const cv::Mat& covariance, const cv::Mat& Qt)
    : landmark(landmark)
    , covariance(covariance)
    , Qt(Qt)
{}
KalmanFilter::KalmanFilter(const KalmanFilter& kf)
    : landmark(kf.landmark)
    , covariance(kf.covariance)
    , Qt(kf.Qt)
{}
KalmanFilter::KalmanFilter(KalmanFilter&& kf)
    : landmark(std::move(kf.landmark))
    , covariance(std::move(kf.covariance))
    , Qt(std::move(kf.Qt))
{}
KalmanFilter& KalmanFilter::operator = (const KalmanFilter& ekf)
{
    landmark = ekf.landmark;
    covariance = ekf.covariance;
    Qt = ekf.Qt;
    return *this;
}
KalmanFilter& KalmanFilter::operator = (KalmanFilter&& ekf)
{
    landmark = std::move(ekf.landmark);
    covariance = std::move(ekf.covariance);
    Qt = std::move(ekf.Qt);
    return *this;
}

const cv::Mat& KalmanFilter::get_covariance() const
{
    return covariance;
}

const cv::Mat& KalmanFilter::get_Qt() const
{
    return Qt;
}

Landmark& KalmanFilter::get_landmark()
{
    return landmark;
}

Landmark KalmanFilter::view_landmark() const
{
    return landmark;
}

void KalmanFilter::update(const cv::Vec2d& measurement, const cv::Mat& H)
{
    cv::Vec2d prediction = landmark.get_parameters();
    cv::Mat H_transposed;
    cv::transpose(H, H_transposed);

    auto Q = H * covariance * H_transposed + Qt;
    auto K = covariance * H_transposed * Q.inv();
    cv::Vec2d diff = measurement - prediction;
    diff[1] = std::fmod(diff[1] + M_PI, M_PI * 2) - M_PI;

    cv::Mat new_parameters = K * diff;
    landmark.update_parameters(new_parameters.at<double>(0, 0), new_parameters.at<double>(1, 0));
    covariance -= K * H * covariance;
}

std::ostream& operator << (std::ostream& os, const KalmanFilter& kf)
{
    return os << "<EKF " << kf.landmark << ">";
}