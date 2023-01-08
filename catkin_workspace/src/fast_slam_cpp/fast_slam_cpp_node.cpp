#include <array>
#include <memory>
#include <mutex>
#include <sstream>
#include <string>
#include <vector>

#include <geometry_msgs/PoseArray.h>
#include <geometry_msgs/PoseStamped.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Odometry.h>
#include <nav_msgs/MapMetaData.h>
#include <ros/ros.h>
#include <sensor_msgs/LaserScan.h>

#include "LandmarkExtractor.hpp"
#include "ParticleFilter.hpp"
#include "Utils.hpp"

static constexpr std::size_t MAP_HEIGHT = 832;
static constexpr std::size_t MAP_WIDTH = 832;
static constexpr double MAP_RESOLUTION = 0.05;
static constexpr std::size_t MAX_SUBSCRIBER_QUEUE_SIZE = 1000;
static constexpr double RESAMPLE_PROPORTIONAL_FRACTION = 0.9;

// Smart pointer to the particle filter instance
static std::unique_ptr<ParticleFilter> particle_filter = nullptr;
// Mutex protecting `particle_filter`
static std::mutex particle_filter_mtx;
// The last pose the robot estimate through odometry
static cv::Vec3d last_pose {0.0, 0.0, 0.0};
// The empirical variance of odometry measurements
static cv::Vec2d odom_variance {0.00001, 0.0003};
// Inner map representation
static std::vector<int8_t> best_map_estimate(MAP_HEIGHT * MAP_WIDTH);

static int resample_count = 0;

static ros::Publisher map_publisher;
static ros::Publisher map_metadata_publisher;
static ros::Publisher pose_publisher;
static ros::Publisher pose_array_publisher;

/*
  A structure representing an Euler angle
*/
struct EulerAngle {
    double x, y, z;
};

/*
  This function converts a quaternion to an Euler angle

  \param q the quaternion
  \returns The Euler angle
*/
EulerAngle quaternion_to_euler_angle(geometry_msgs::Quaternion q) {
    return {
        std::atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y)),
        std::asin(2 * (q.w * q.y - q.z * q.x)), 
        std::atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    };
}

/*
  This function converts an Euler angle to a quaternion

  \params an Euler angle
  \returns The quaternion
*/
geometry_msgs::Quaternion euler_angle_to_quaternion(double x, double y, double z) {
  geometry_msgs::Quaternion result;
  result.x = std::sin(x/2) * std::cos(y/2) * std::cos(z/2) - std::cos(x/2) * std::sin(y/2) * std::sin(z/2);
  result.y = std::cos(x/2) * std::sin(y/2) * std::cos(z/2) + std::sin(x/2) * std::cos(y/2) * std::sin(z/2);
  result.z = std::cos(x/2) * std::cos(y/2) * std::sin(z/2) - std::sin(x/2) * std::sin(y/2) * std::cos(z/2);
  result.w = std::cos(x/2) * std::cos(y/2) * std::cos(z/2) + std::sin(x/2) * std::sin(y/2) * std::sin(z/2);
  return result;
}

double stddev(const std::vector<double>& v)
{
  double sum = std::accumulate(
    v.begin(), 
    v.end(), 
    0.0, 
    [](double acc, double element) { return element; }
  );
  double mean = sum / v.size();

  std::vector<double> diff(v.size());
  std::transform(
    v.begin(), 
    v.end(), 
    diff.begin(),
    [mean](double element) { return element - mean; }
  );
  double sq_sum = std::inner_product(diff.begin(), diff.end(), diff.begin(), 0.0);
  double stddev = std::sqrt(sq_sum / v.size());
  return stddev;
}

/*
  This function is called when an odometry reading is published to the "odom"
  topic. It updates the particle filter with this new information.
*/
void odom_callback(const nav_msgs::Odometry::ConstPtr& msg)
{
  // Convert the rotation to an euler angle
  EulerAngle rot = quaternion_to_euler_angle(msg->pose.pose.orientation);
  // Get robot's position estimate
  auto pos = msg->pose.pose.position;

  cv::Vec3d pose {pos.x, pos.y, pos.z};
  // Create a new estimate based on our model
  cv::Vec3d pose_estimate = pose - last_pose;
  cv::Vec2d transformed_pose_estimate {std::sqrt(pose_estimate[0]*pose_estimate[0] + pose_estimate[1]*pose_estimate[1]), pose_estimate[2]};
  last_pose = pose;

  std::unique_lock<std::mutex> lck(particle_filter_mtx);
  // Update the positions of all particles
  particle_filter->sample_pose(transformed_pose_estimate, odom_variance);
}

/*
  This function is called when laser measurements are published to the "laser"
  topic. It updates the particles with new landmark observations when needed 
  and performs resampling.
*/
void scan_callback(const sensor_msgs::LaserScan::ConstPtr& msg)
{
  resample_count++;

  // float -> double conversion
  std::vector<double> points;
  points.resize(msg->ranges.size());
  std::copy(msg->ranges.begin(), msg->ranges.end(), points.begin());

  // Extract landmarks from the laser scan
  ros::Time start, end;
  start = ros::Time::now();
  auto extracted = extract_landmarks(points, ExtractionAlgorithm::RANSAC);

  std::cout << "Extracted:" << std::endl;
  for (const auto& landmark : extracted) {
    std::cout << landmark << std::endl;
  }

  std::vector<Particle> particles;
  {
    std::unique_lock<std::mutex> lck(particle_filter_mtx);
    // Update particle weights using extracted landmarks
    std::size_t matches = particle_filter->observe_landmarks(extracted);
    particles = particle_filter->get_particles();

    std::vector<double> weights(particles.size());
    std::transform(particles.begin(), particles.end(), weights.begin(), [](const Particle& p){return p.get_weight();});
    double stddeviation = stddev(weights);
    ROS_INFO("Standard deviation: %.3f (%lu matches)", stddeviation, matches);
    
    // Perform resampling
    if (resample_count >= 5) {
      particle_filter->resample(RESAMPLE_PROPORTIONAL_FRACTION);
      resample_count = 0;
    }
    end = ros::Time::now();
    particles = particle_filter->get_particles();
  }
  auto processing_time = end - start;
  double elapsed_milliseconds = processing_time.sec * 1e3 + processing_time.nsec * 1e-6;

  ROS_INFO("Elapsed: %.3lf ms (%lu landmarks)", elapsed_milliseconds, extracted.size());

  Particle best_particle = *std::max_element(particles.begin(), particles.end(), [](const Particle& p1, const Particle& p2){
    return p1.get_weight() < p2.get_weight();
  });

  if (resample_count == 0) {
    for (const auto& landmark : best_particle.get_all_landmarks())
      std::cout << landmark << std::endl;
  }

  for (std::size_t i = 0; i < points.size(); i++) {
    if (points[i] <= 0.001)
      continue;
    cv::Vec2d point = polar_to_cartesian(2 * M_PI / points.size() * i + best_particle.get_pose()(2), points[i]);
    point(0) += best_particle.get_pose()(0);
    point(1) += best_particle.get_pose()(1);
    point *= 1.0 / MAP_RESOLUTION;

    std::size_t row = (std::size_t) (MAP_HEIGHT / 2.0 + point(1));
    std::size_t column = (std::size_t) (MAP_WIDTH / 2.0 + point(0));
    std::size_t index = row * MAP_WIDTH + column;

    if (index > best_map_estimate.size())
      continue;

    if (best_map_estimate[index] < 100) {
      if (best_map_estimate[index] == -1)
        best_map_estimate[index] = 1;
      best_map_estimate[index] += 2;
    }
  }

  nav_msgs::MapMetaData md;
  md.map_load_time.sec = processing_time.sec;
  md.map_load_time.nsec = processing_time.nsec;
  md.height = MAP_HEIGHT;
  md.width = MAP_WIDTH;
  md.resolution = MAP_RESOLUTION;
  md.origin.position.x = -(MAP_WIDTH / 2.0) * MAP_RESOLUTION;
  md.origin.position.y = -(MAP_HEIGHT / 2.0) * MAP_RESOLUTION;
  md.origin.position.z = 0;
  md.origin.orientation.w = 0;
  md.origin.orientation.x = 0;
  md.origin.orientation.y = 0;
  md.origin.orientation.z = 0;

  nav_msgs::OccupancyGrid og;
  og.header.stamp = ros::Time::now();
  og.header.frame_id = "map";
  og.data = best_map_estimate;
  og.info = md;

  geometry_msgs::PoseStamped ps;
  ps.header.stamp = og.header.stamp;
  ps.header.frame_id = "map";
  ps.pose.orientation = euler_angle_to_quaternion(0, 0, best_particle.get_pose()(2));
  ps.pose.position.x = best_particle.get_pose()(0);
  ps.pose.position.y = best_particle.get_pose()(1);
  ps.pose.position.z = 0;

  geometry_msgs::PoseArray pa;
  pa.header.stamp = og.header.stamp;
  pa.header.frame_id = "map";
  for (const Particle& particle : particles) {
    geometry_msgs::Pose pose_for_array;
    pose_for_array.orientation = euler_angle_to_quaternion(0, 0, particle.get_pose()(2));
    pose_for_array.position.x = particle.get_pose()(0);
    pose_for_array.position.y = particle.get_pose()(1);
    pose_for_array.position.z = 0;
    pa.poses.push_back(pose_for_array);
  }

  map_metadata_publisher.publish(md);
  map_publisher.publish(og);
  pose_publisher.publish(ps);
  pose_array_publisher.publish(pa);
}

cv::Mat function_h(double x, double y, double theta)
{
    static cv::Mat result {cv::Mat::ones({2, 2}, CV_64F)};
    result.at<double>(1, 0) = 0;
    result.at<double>(0, 1) = x * std::sin(theta) - y * std::cos(theta);
    return result;
}

int main(int argc, char* argv[])
{
  for (auto& x : best_map_estimate)
    x = -1;

  // This must be called before anything else ROS-related
  ros::init(argc, argv, "fast_slam_cpp");

  // Create a ROS node handle
  ros::NodeHandle nh;

  // Subscribe and create publishers for all relevant topics
  ros::Subscriber sub1 = nh.subscribe("odom", MAX_SUBSCRIBER_QUEUE_SIZE, odom_callback);
  ros::Subscriber sub2 = nh.subscribe("scan", MAX_SUBSCRIBER_QUEUE_SIZE, scan_callback);
  map_publisher = nh.advertise<nav_msgs::OccupancyGrid>("map", MAX_SUBSCRIBER_QUEUE_SIZE);
  map_metadata_publisher = nh.advertise<nav_msgs::MapMetaData>("map_metadata", MAX_SUBSCRIBER_QUEUE_SIZE);
  pose_publisher = nh.advertise<geometry_msgs::PoseStamped>("pose", MAX_SUBSCRIBER_QUEUE_SIZE);
  pose_array_publisher = nh.advertise<geometry_msgs::PoseArray>("pose_array", MAX_SUBSCRIBER_QUEUE_SIZE);

  // This is the data for the sensor covariance matrix
  double Qtdata[] = {0.01, 0, 0, 0.003};

  {
    std::unique_lock<std::mutex> lck(particle_filter_mtx);
    // Instantiate a new particle filter
    particle_filter = std::make_unique<ParticleFilter>(200, cv::Mat {2, 2, CV_64F, Qtdata}, function_h);
  }

  ROS_INFO("Initialized particle filter, waiting for data...");

  // Run until node is killed
  ros::spin();

}