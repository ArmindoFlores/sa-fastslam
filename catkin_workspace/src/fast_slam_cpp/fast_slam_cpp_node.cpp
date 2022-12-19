#include <memory>
#include <mutex>

#include <ros/ros.h>
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/LaserScan.h>

#include "ParticleFilter.hpp"
#include "LandmarkExtractor.hpp"

static std::shared_ptr<ParticleFilter> particle_filter = nullptr;
static std::mutex particle_filter_mtx;
static cv::Vec3d last_pose {0.0, 0.0, 0.0};
static cv::Vec2d odom_variance {0.00001, 0.0005};

struct EulerAngle {
    double x, y, z;
};

EulerAngle quaternion_to_euler_angle(geometry_msgs::Quaternion q) {
    return {
        std::atan2(2 * (q.w * q.x + q.y * q.z), 1 - 2 * (q.x * q.x + q.y * q.y)),
        std::asin(2 * (q.w * q.y - q.z * q.x)), 
        std::atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z))
    };
}

cv::Mat function_h(double x, double y, double theta)
{
    static cv::Mat result {cv::Mat::ones({2, 2}, CV_64F)};
    result.at<double>(1, 0) = 0;
    result.at<double>(0, 1) = x * std::sin(theta) - y * std::cos(theta);
    return result;
}

void odom_callback(const nav_msgs::Odometry::ConstPtr& msg)
{
  EulerAngle rot = quaternion_to_euler_angle(msg->pose.pose.orientation);
  auto pos = msg->pose.pose.position;

  cv::Vec3d pose {pos.x, pos.y, pos.z};
  cv::Vec3d pose_estimate = pose - last_pose;
  cv::Vec2d transformed_pose_estimate {std::sqrt(pose_estimate[0]*pose_estimate[0] + pose_estimate[1]*pose_estimate[1]), pose_estimate[2]};

  std::unique_lock<std::mutex> lck(particle_filter_mtx);
  particle_filter->sample_pose(transformed_pose_estimate, odom_variance);
}

void scan_callback(const sensor_msgs::LaserScan::ConstPtr& msg)
{
  std::vector<double> points;
  points.resize(msg->ranges.size());
  std::copy(msg->ranges.begin(), msg->ranges.end(), points.begin());

  auto extracted = extract_landmarks(points);

  std::stringstream ss;
  {
    std::unique_lock<std::mutex> lck(particle_filter_mtx);
    particle_filter->observe_landmarks(extracted);
    particle_filter->resample(0.7);
    ss << particle_filter->get_particles()[0].get_pose();
  }
  ROS_INFO("Position: %s", ss.str().c_str());
}

int main(int argc, char* argv[])
{
  // This must be called before anything else ROS-related
  ros::init(argc, argv, "fast_slam_cpp");

  // Create a ROS node handle
  ros::NodeHandle nh;
  ros::Subscriber sub1 = nh.subscribe("odom", 1000, odom_callback);
  ros::Subscriber sub2 = nh.subscribe("scan", 1000, scan_callback);

  double Qtdata[] = {0.01, 0, 0, 0.003};

  {
    std::unique_lock<std::mutex> lck(particle_filter_mtx);
    particle_filter = std::make_shared<ParticleFilter>(100, cv::Mat {2, 2, CV_64F, Qtdata}, function_h);
  }

  ROS_INFO("Initialized particle filter, waiting for data...");

  ros::spin();

}