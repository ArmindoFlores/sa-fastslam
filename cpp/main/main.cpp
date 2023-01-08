#include <chrono>
#include <fstream>
#include <iostream>
#include <random>

#include <boost/json/src.hpp>

#include "IterativeClosestPoint.hpp"
#include "Utils.hpp"

int main()
{
    double _Qt[] = {0.01, 0, 0, 0.003};
    double _H[] = {1, 1.419638946328426, 0, 1};
    double _covariance[] = {0.01, -0.01404438283438678, -0.01404438283438678, 0.02272446891988181};

    cv::Mat Qt {2, 2, CV_64F, _Qt};
    cv::Mat H {2, 2, CV_64F, _H};
    cv::Mat H_transposed;
    cv::transpose(H, H_transposed);
    cv::Mat covariance {2, 2, CV_64F, _covariance};

    std::cout << "Qt: " << std::endl;
    std::cout << Qt << std::endl;

    std::cout << "H: " << std::endl;
    std::cout << H << std::endl;

    std::cout << "H': " << std::endl;
    std::cout << H_transposed << std::endl;

    std::cout << "covariance: " << std::endl;
    std::cout << covariance << std::endl;

    auto Q = H * covariance * H_transposed + Qt;

    std::cout << Q << std::endl;
    std::cout << Q.inv() << std::endl;
}

/*
int main()
{
    std::default_random_engine generator;
    std::uniform_real_distribution<double> uniform(-0.025, 0.025);

    std::ifstream datapoints_file {"reading.json"};
    std::string content {
        std::istreambuf_iterator<char>(datapoints_file),
        std::istreambuf_iterator<char>()
    };
    datapoints_file.close();

    boost::json::value ranges {boost::json::parse(content)};
    auto array = ranges.as_array()[0].as_array();

    std::vector<cv::Vec3d> baseline_points;
    for (std::size_t i = 0; i < array.size(); i++) {
        const auto& value = array[i];
        if (value.as_double() == 0)
            continue;
        auto cartesian = polar_to_cartesian(2 * M_PI * i / (double) array.size(), value.as_double());
        baseline_points.emplace_back(cartesian[0], cartesian[1], 1.0);
    }

    double rotation_angle = 10 * M_PI / 180.0;
    std::vector<cv::Vec3d> noisy_points (baseline_points);
    cv::Vec3d translation_vector {2.0, -1.0, 0.0};
    cv::Matx33d rotation_matrix {
        std::cos(rotation_angle), -std::sin(rotation_angle), 0.0,
        std::sin(rotation_angle), std::cos(rotation_angle), 0.0,
        0.0, 0.0, 1.0
    };

    for (auto& point : noisy_points) {
        auto temp = rotation_matrix * point;
        point[0] = temp[0];
        point[1] = temp[1];
        point[2] = temp[2];
        point += translation_vector;
        point[0] += uniform(generator);
        point[1] += uniform(generator);
    }

    auto result = iterative_closest_point(baseline_points, noisy_points);
    std::cout << result.A << std::endl;
    std::cout << result.b << std::endl;

}
*/