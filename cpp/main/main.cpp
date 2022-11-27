#include <chrono>
#include <fstream>
#include <iostream>

#include <boost/json/src.hpp>

#include "Landmark.hpp"
#include "LandmarkExtractor.hpp"


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

    auto start = std::chrono::high_resolution_clock::now();
    auto result = RANSAC(points, 10, 20, 0.01, 10, 10);
    auto end = std::chrono::high_resolution_clock::now();

    std::cout << "Number of lines found: " << result.size() << std::endl;
    std::cout << "Time elapsed: " << (end - start).count() / 1e6 << " ms" << std::endl; 

    for (const auto& line : result) {
        std::cout << Landmark(line.a, line.b, line.c) << std::endl;
    }
}