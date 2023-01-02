#include "IterativeClosestPoint.hpp"
#include "Utils.hpp"
#include <iostream>

std::optional<cv::Mat> least_squares(const cv::Mat& X, const cv::Mat& y)
{
    // FIXME: isn't this completely wrong?
    cv::Mat Xt;
    cv::transpose(X, Xt);

    try {
        return (Xt * X).inv() * Xt * y;
    }
    catch (cv::Exception&) {
        return {};
    }
}

ICPSolution iterative_closest_point(const std::vector<cv::Vec3d>& baseline, std::vector<cv::Vec3d> new_scan, double error, std::size_t max_iterations)
{
    cv::Vec3d baseline_mean {0.0, 0.0, 0.0}, new_scan_mean {0.0, 0.0, 0.0};

    for (const auto& point : baseline)
        baseline_mean += point;
    baseline_mean *= 1.0 / baseline.size();
    for (const auto& point : new_scan)
        new_scan_mean += point;
    new_scan_mean *= 1.0 / new_scan.size();

    cv::Vec3d translation_vector = new_scan_mean - baseline_mean;
    for (auto& point : new_scan)
        point -= translation_vector;

    double current_error = std::numeric_limits<double>::infinity();
    std::size_t iteration = 0;
    cv::Mat final_solution {cv::Mat::eye({3, 3}, CV_64F)};
    while (current_error > error && iteration < max_iterations) {
        iteration++;

        // Compute the nearest neighbour of every point
        std::vector<cv::Vec3d> neighbours(new_scan.size());
        for (std::size_t i = 0; i < new_scan.size(); i++) {
            const auto& point = new_scan[i];
            const cv::Vec3d& possible_nearest_neighbour = baseline[0];
            double min_distance = std::numeric_limits<double>::infinity();
            for (const auto& neighbour : baseline) {
                double distance = cv::norm(point - neighbour);
                if (distance < min_distance) {
                    distance = min_distance;
                }
            }
            neighbours[i] = possible_nearest_neighbour;
        }
        cv::Mat A {(int) new_scan.size(), 3, CV_64F, cv::Scalar(0.0)};
        cv::Mat b {(int) baseline.size(), 3, CV_64F, cv::Scalar(0.0)};
        for (int row = 0; row < A.rows; row++) {
            *A.ptr<double>(row, 0) = new_scan[row][0];
            *A.ptr<double>(row, 1) = new_scan[row][1];
            *A.ptr<double>(row, 2) = new_scan[row][2];
        }
        for (int row = 0; row < b.rows; row++) {
            *b.ptr<double>(row, 0) = baseline[row][0];
            *b.ptr<double>(row, 1) = baseline[row][1];
            *b.ptr<double>(row, 2) = baseline[row][2];
        }

        auto solution = least_squares(A, b);
        if (!solution.has_value())
            continue;

        // FIXME: compute the error
        // current_error = ...;
        final_solution = final_solution * solution.value();
        for (std::size_t i = 0; i < new_scan.size(); i++) {
            cv::Mat transposed_vector;
            cv::transpose(new_scan[i], transposed_vector);
            new_scan[i] = (cv::Mat) (transposed_vector * solution.value());
        }
    }
    return {final_solution, translation_vector};
}