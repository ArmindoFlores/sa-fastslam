#include <chrono>
#include <fstream>
#include <iostream>

#include <boost/json/src.hpp>

#include "LandmarkExtractor.hpp"
#include "ParticleFilter.hpp"

cv::Mat function_h(double x, double y, double theta)
{
    static cv::Mat result {cv::Mat::ones({2, 2}, CV_64F)};
    result.at<double>(1, 0) = 0;
    result.at<double>(0, 1) = x * std::sin(theta) - y * std::cos(theta);
    return result;
}

int main()
{
    std::ifstream datapoints_file {"reading.json"};
    std::string content {
        std::istreambuf_iterator<char>(datapoints_file),
        std::istreambuf_iterator<char>()
    };
    datapoints_file.close();

    boost::json::value ranges {boost::json::parse(content)};
    auto array = ranges.as_array()[0].as_array();

    std::vector<double> points;
    for (const auto& value : array) {
        points.push_back(value.as_double());
    }

    srand(time(nullptr));

    double data[] = {0.01, 0, 0, 0.003};
    cv::Mat Qt {2, 2, CV_64F, data};   
    cv::Vec3d base_pose {0.0, 0.0, 0.0}; 

    ParticleFilter pf {200, Qt, function_h};

    auto start = std::chrono::high_resolution_clock::now();
    auto extracted = extract_landmarks(points);
    std::cout << "Extracted " << extracted.size() << " landmarks!" << std::endl;
    pf.observe_landmarks(extracted);
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Time elapsed: " << (end - start).count() / 1e6 << " ms" << std::endl; 

    for (const auto& particle : pf.get_particles()) {
        std::cout << particle << std::endl;
    }
}