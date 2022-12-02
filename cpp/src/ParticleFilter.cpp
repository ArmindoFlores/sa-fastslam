#include "ParticleFilter.hpp"

Particle::Particle(const cv::Mat& Qt, std::function<cv::Mat(double, double, double)> H, const cv::Vec3d& pose)
    : weight(0)
    , Qt(Qt)
    , h_func(H)
    , pose(pose)
    , landmark_matcher(Qt)
{}
Particle::Particle(const cv::Mat& Qt, std::function<cv::Mat(double, double, double)> H)
    : Particle(Qt, H, {0.0, 0.0, 0.0})
{}
Particle::Particle(const Particle& p)
    : Particle::Particle(p.Qt, p.h_func, p.pose)
{}
Particle::Particle(Particle&& p)
    : weight(p.weight)
    , landmark_matcher(std::move(p.landmark_matcher))
{
    Qt = std::move(p.Qt);
    h_func = p.h_func;
    pose = std::move(p.pose);
}
Particle& Particle::operator = (const Particle& p)
{
    weight = p.weight;
    Qt = p.Qt;
    pose = p.pose;
    landmark_matcher = p.landmark_matcher;
    return *this;
}
Particle& Particle::operator = (Particle&& p)
{
    weight = p.weight;
    Qt = std::move(p.Qt);
    h_func = p.h_func;
    pose = std::move(p.pose);
    landmark_matcher = std::move(p.landmark_matcher);
    return *this;
}

std::optional<KalmanFilter> Particle::observe_landmark(const Landmark& landmark)
{
    auto match = landmark_matcher.observe(landmark, h_func, pose);
    if (match.has_value())
        weigh(landmark.get_parameters(), match.value());
    return match;
}

void Particle::update_pose(double r, double theta, const cv::Vec2d& variance)
{
    std::normal_distribution<double> r_dist(r, std::sqrt(variance[0])), theta_dist(theta, std::sqrt(variance[1]));
    double r_estimate = r_dist(generator);
    auto theta_estimate = theta_dist(generator);
    pose[0] += r_estimate * std::cos(pose[2]);
    pose[1] += r_estimate * std::sin(pose[2]);
    pose[2] += theta_estimate;
}

void Particle::set_weight(double weight)
{
    this->weight = weight;
}

void Particle::weigh(const cv::Vec2d& measurement, const KalmanFilter& matched_landmark)
{
    Landmark landmark {matched_landmark.view_landmark()};
    cv::Vec2d z_predicted = landmark.get_parameters(pose);
    cv::Mat H = h_func(pose[0], pose[1], landmark.get_parameters()[1]), H_transposed;
    cv::transpose(H, H_transposed);
    cv::Mat Q = H * matched_landmark.get_covariance() * H_transposed + Qt;
    cv::Vec2d Z = measurement - z_predicted;
    weight *= std::exp(((cv::Mat) (-0.5 * Z * Q.inv() * Z)).at<double>(0, 0)) / std::sqrt(cv::determinant(2 * M_PI * Q));
}

std::ostream& operator << (std::ostream& os, const Particle& p)
{
    os << "<Particle pose=[" << p.pose[0] << ", " << p.pose[1] << ", " << p.pose[2] << "]>";
    return os;
}

ParticleFilter::ParticleFilter(std::size_t N, const cv::Mat& Qt, std::function<cv::Mat(double, double, double)> H, const cv::Vec3d& initial_pose)
{
    particles.reserve(N);
    for (std::size_t i = 0; i < N; i++)
        particles.emplace_back(Qt, H, initial_pose);
}
ParticleFilter::ParticleFilter(std::size_t N, const cv::Mat& Qt, std::function<cv::Mat(double, double, double)> H)
{
    particles.reserve(N);
    for (std::size_t i = 0; i < N; i++)
        particles.emplace_back(Qt, H);
}

void ParticleFilter::sample_pose(const cv::Vec2d& odometry_reading, const cv::Vec2d& variance)
{
    double r = cv::norm(odometry_reading);
    double theta = odometry_reading[1];
    for (auto& particle : particles)
        particle.update_pose(r, theta, variance);
}

void ParticleFilter::observe_landmarks(const std::vector<Landmark>& landmarks)
{
    for (auto& particle : particles) {
        for (const auto& landmark : landmarks) {
            particle.observe_landmark(landmark);
        }
    }
}

void ParticleFilter::resample(std::size_t N, double frac)
{

}

void ParticleFilter::resample(double frac)
{
    return resample(particles.size(), frac);
}

const std::vector<Particle>& ParticleFilter::get_particles() const
{
    return particles;
}
