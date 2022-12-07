#include <ros/ros.h>

int main(int argc, char* argv[])
{
  // This must be called before anything else ROS-related
  ros::init(argc, argv, "fast_slam_cpp");

  // Create a ROS node handle
  // ros::NodeHandle nh;

  ROS_INFO("Hello, World!");
}