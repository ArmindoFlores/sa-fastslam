#ifndef _H_PARTICLE_FILTER_
#define _H_PARTICLE_FILTER

#include "LandmarkMatcher.hpp"
#include <iostream>
#include <optional>
#include <random>
#include <vector>
#include <opencv2/core.hpp>

static std::default_random_engine generator;

/*
    This class describes a particle from a particle filter
*/
class Particle {
public:
    /*
        Construct a new particle

        \param Qt the 2x2 sensor covariance matrix
        \param H a function that returns the 2x2 Jacobian matrix of the measurement
        model matrix
        \param pose the starting pose for the particle
    */
    Particle(const cv::Mat& Qt, std::function<cv::Mat(double, double, double)> H, const cv::Vec3d& pose);
    /*
        Construct a new particle

        \param Qt the 2x2 sensor covariance matrix
        \param H a function that returns the 2x2 Jacobian matrix of the measurement
        model matrix
    */
    Particle(const cv::Mat& Qt, std::function<cv::Mat(double, double, double)> H);
    Particle(const Particle&);
    Particle(Particle&&);
    Particle& operator = (const Particle&);
    Particle& operator = (Particle&&);

    /*
        Update this particle upon a landmark sighting

        \param landmark the observed landmark
        \return A Kalman Filter responsible for the corresponding landmark if
        a match is found
    */
    std::optional<KalmanFilter> observe_landmark(const Landmark& landmark);
    /*
        Update this particle's pose according to a normal distribution

        \param r the mean distance of the distribution
        \param theta the mean angle of the distribution
        \param variance the variance of the distance and the angle
    */
    void update_pose(double r, double theta, const cv::Vec2d& variance);

    /*
        Manually set the particle's current weight

        \param weight the new particle weight
    */
    void set_weight(double weight);

    /*
        \return The particle's weight
    */
    double get_weight() const;

    friend std::ostream& operator << (std::ostream&, const Particle&);

private:
    void weigh(const cv::Vec2d& measurement, const KalmanFilter& matched_landmark);

    double weight;
    cv::Mat Qt;
    std::function<cv::Mat(double, double, double)> h_func;
    cv::Vec3d pose;
    LandmarkMatcher landmark_matcher;
};

class ParticleFilter {
public:
    /*
        Construct a new particle filter

        \param N the number of particles
        \param Qt the 2x2 sensor covariance matrix
        \param H a function that returns the 2x2 Jacobian matrix of the measurement
        model matrix
        \param initial_pose a vector with the particles' initial pose
    */
    ParticleFilter(std::size_t N, const cv::Mat& Qt, std::function<cv::Mat(double, double, double)> H, const cv::Vec3d& initial_pose);
    
    /*
        Construct a new particle filter

        \param N the number of particles
        \param Qt the 2x2 sensor covariance matrix
        \param H a function that returns the 2x2 Jacobian matrix of the measurement
        model matrix
    */
    ParticleFilter(std::size_t N, const cv::Mat& Qt, std::function<cv::Mat(double, double, double)> H);
    // ParticleFilter(const Particle&) = delete;
    // ParticleFilter(ParticleFilter&&) = delete;

    /*
        Change the pose of all the particles based on a reading

        \param odometry_reading the distance and angle reading from odometry
        \param variance the variance of this reading
    */
    void sample_pose(const cv::Vec2d& odometry_reading, const cv::Vec2d& variance);
    // std::vector<Landmark> observe_landmarks(const std::vector<Landmark>&);
    /*
        Report a landmark sighting to all particles and update them

        \param landmarks the vector of observed landmarks
    */
    void observe_landmarks(const std::vector<Landmark>& landmarks);
    /*
        Perform resampling on the set of particles based on their weight

        \param N the number of new particles
        \param frac the fraction of particles to sample based on weight
    */
    void resample(std::size_t N, double frac);
    /*
        Perform resampling on the set of particles based on their weight

        \param frac the fraction of particles to sample based on weight
    */
    void resample(double frac=1.0);

    const std::vector<Particle>& get_particles() const;

private:
    std::vector<Particle> particles;
};

#endif