#ifndef _H_PARTICLE_FILTER_
#define _H_PARTICLE_FILTER

#include "LandmarkMatcher.hpp"
#include <iostream>
#include <optional>
#include <vector>
#include <opencv2/core.hpp>

class Particle {
public:
    Particle(const cv::Mat& Qt, std::function<cv::Mat(double, double, double)> H, cv::Vec3d pose);
    Particle(const Particle&);
    Particle(Particle&&);
    Particle& operator = (const Particle&);
    Particle& operator = (Particle&&);

    std::optional<Landmark> observe_landmark(const Landmark& landmark);
    void update_pose(double r, double theta, const cv::Vec2d& variance);

    void set_weight(double weight);

    friend std::ostream& operator << (std::ostream&, const Particle&);

private:
    void weigh(const cv::Vec2d& measurement, const Landmark& matched_landmark);

    double weight;
    cv::Mat Qt;
    std::function<cv::Mat(double, double, double)> h_func;
    LandmarkMatcher landmark_matcher;
};

class ParticleFilter {
public:
    ParticleFilter(std::size_t N, const cv::Mat& Qt, std::function<cv::Mat(double, double, double)> H);
    ParticleFilter(std::size_t N, const cv::Mat& Qt, std::function<cv::Mat(double, double, double)> H, const cv::Vec3d& initial_pose);
    ParticleFilter(const Particle&) = delete;
    ParticleFilter(ParticleFilter&&) = delete;

    void sample_pose(const cv::Vec2d& odometry_reading, const cv::Vec2d& variance);
    // std::vector<Landmark> observe_landmarks(const std::vector<Landmark>&);
    void observe_landmarks(const std::vector<Landmark>& landmarks);
    void resample(double frac=1.0);
    void resample(std::size_t N, double frac);

private:
    std::vector<Particle> particles;
};

#endif