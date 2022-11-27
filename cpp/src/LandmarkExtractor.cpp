#include "LandmarkExtractor.hpp"

#include <iostream>
#include <cmath>
#include <optional>

double drandom()
{
    return rand() / ((double) RAND_MAX + 1);
}

cv::Vec2d polar_to_cartesian(const cv::Vec2d& vec)
{
    return polar_to_cartesian(vec(0), vec(1));
}

cv::Vec2d polar_to_cartesian(double r, double theta)
{
    if (r > 0.001)
        return r * cv::Vec2d {std::cos(theta), std::sin(theta)};
    return 5000 * cv::Vec2d {std::cos(theta), std::sin(theta)};
}

std::optional<cv::Mat> linear_regression(const std::vector<cv::Vec2d>& points)
{
    cv::Mat X {(int) points.size(), 2, CV_64F, cv::Scalar::all(1)}, y {(int) points.size(), 1, CV_64F, cv::Scalar::all(1)};
    for (size_t i = 0; i < points.size(); i++) {
        X.at<double>(i, 0) = points[i].val[0];
        y.at<double>(i, 0) = points[i].val[1];
    }
    cv::Mat Xt;
    cv::transpose(X, Xt);

    try {
        return (Xt * X).inv() * Xt * y;
    }
    catch (cv::Exception&) {
        return {};
    }
}

enum class RepresentationType {
    X,
    Y
};

std::vector<RANSACResult> RANSAC(
    const std::vector<double> ranges,
    std::size_t N, 
    std::size_t C, 
    double X, 
    double D, 
    std::size_t S
)
{
    std::vector<RANSACResult> features;

    if (ranges.size() < C)
        return features;

    std::vector<cv::Vec2d> cartesian (ranges.size());
    std::vector<ssize_t> available {};

    for (std::size_t i = 0; i < ranges.size(); i++) {
        cartesian[i] = polar_to_cartesian(ranges[i], (double) i / ranges.size() * M_PI * 2);
    }

    for (std::size_t i = 0; i < ranges.size(); i++) {
        if (ranges[i] > 0.001)
            available.push_back(i);
    }

    ssize_t sample_range = D / 360 * ranges.size();
    std::vector<ssize_t> samples;

    std::size_t n = 0;
    auto rotmat = cv::Mat(2, 2, CV_64F);
    auto invrotmat = cv::Mat(2, 2, CV_64F);

    cv::Mat p_start, p_end;
    std::vector<ssize_t> close_enough;
    std::vector<cv::Vec2d> selected(S+1);
    std::vector<cv::Vec2d> near_points;

    while (n++ < N && available.size() > 0) {
        samples.clear();

        // Chose a random sample
        ssize_t index = (std::size_t) (drandom() * available.size());
        ssize_t R_index = available[index];
        // double R = ranges[R_index];

        // Gather all points within D degrees of the choosen sample
        for (int offset = 1; offset <= sample_range; offset++) {
            ssize_t element1 = available[(index + offset) % available.size()];
            ssize_t element2 = available[(index - offset) % available.size()];
            if (std::abs(element1 - R_index) <= sample_range)
                samples.push_back(element1);
            if (std::abs(element2 - R_index) <= sample_range)
                samples.push_back(element2);
        }


        // Abort if there aren't enough samples
        if (samples.size() < S)
            continue;

        // Create rotation matrix
        double theta = drandom() * 2 * M_PI;
        double cosine = std::cos(theta);
        double sine = std::sin(theta);
        rotmat.row(0).col(0) = cosine;
        rotmat.row(0).col(1) = -sine;
        rotmat.row(1).col(0) = sine;
        rotmat.row(1).col(1) = cosine;
        invrotmat.row(0).col(0) = cosine;
        invrotmat.row(0).col(1) = sine;
        invrotmat.row(1).col(0) = -sine;
        invrotmat.row(1).col(1) = cosine;

        // Pick S random points and convert them to cartesian coordinates
        std::random_shuffle(samples.begin(), samples.end());
        for (size_t i = 0; i < S; i++) {
            selected[i] = (cv::Mat) (rotmat * cartesian[samples[i]]);
        }
        selected[S] = (cv::Mat) (rotmat * cartesian[R_index]);

        // Apply linear regression
        std::optional<cv::Mat> lstsq_result_opt = linear_regression(selected);
        if (!lstsq_result_opt.has_value())
            continue;
        cv::Mat& lstsq_result = lstsq_result_opt.value();

        // Calculate line parameters
        double slope = 0, offset = 0;
        RepresentationType rt;
        p_start = invrotmat * cv::Vec2d {0, lstsq_result.at<double>(1, 0)};
        p_end = invrotmat * cv::Vec2d {1, lstsq_result.at<double>(0, 0) + lstsq_result.at<double>(1, 0)};
        if (std::abs(p_start.at<double>(0, 0) - p_end.at<double>(0, 0)) > 0.001) {
            slope = (p_start.at<double>(1, 0) - p_end.at<double>(1, 0)) / (p_start.at<double>(0, 0) - p_end.at<double>(0, 0));
            rt = RepresentationType::Y;
        }
        else {
            slope = (p_start.at<double>(0, 0) - p_end.at<double>(0, 0)) / (p_start.at<double>(1, 0) - p_end.at<double>(1, 0));
            rt = RepresentationType::X;
        }
        offset = p_start.at<double>(1, 0) - slope * p_start.at<double>(0, 0);

        // Find all readings within X meters of the line

        double denominator = 1.0 / std::pow(slope*slope + 1, 0.5);
        auto it = available.begin();
        close_enough.clear();
        while (it != available.end()) {
            double distance, x = cartesian[*it][0], y = cartesian[*it][1];
            if (rt == RepresentationType::Y)
                distance = std::abs(slope * x - y + offset) * denominator;
            else
                distance = std::abs(slope * y - x + offset) * denominator;

            bool removed = false;

            if (distance < X)
                close_enough.push_back(*it);
            else if (distance < 5 * X) {
                removed = true;
                it = available.erase(it);
            }
            if (!removed) 
                it++;
        }

        // If we find a good enough line
        if (close_enough.size() >= C) {
            // Rotate all points close to the line by the same matrix as before
            near_points.resize(close_enough.size());
            for (size_t i = 0; i < close_enough.size(); i++)
                near_points[i] = (cv::Mat) (rotmat * cartesian[close_enough[i]]);

            // Apply linear regression to get the final line
            std::optional<cv::Mat> final_result_opt = linear_regression(near_points);
            if (!final_result_opt.has_value())
                continue;
            cv::Mat& final_result = final_result_opt.value();

            // Calculate r^2
            double mean = 0, SST = 0, SSE = 0, beta0 = final_result.at<double>(1, 0), beta1 = final_result.at<double>(0, 0);
            for (size_t i = 0; i < near_points.size(); i++)
                mean += near_points[i][1];
            mean /= near_points.size();
            for (size_t i = 0; i < near_points.size(); i++) {
                double prediction = beta0 + beta1 * near_points[i][0];
                double error = near_points[i][1] - prediction;
                double diff = near_points[i][1] - mean;
                SSE += error * error;
                SST += diff * diff;
            }
            double r2 = 1 - SSE / SST; 

            if (r2 < 0.9)
                continue;

            p_start = invrotmat * cv::Vec2d {0, final_result.at<double>(1, 0)};
            p_end = invrotmat * cv::Vec2d {1, final_result.at<double>(0, 0) + final_result.at<double>(1, 0)};

            if (std::abs(p_start.at<double>(0, 0) - p_end.at<double>(0, 0)) > 0.001) {
                slope = (p_start.at<double>(1, 0) - p_end.at<double>(1, 0)) / (p_start.at<double>(0, 0) - p_end.at<double>(0, 0));
                rt = RepresentationType::Y;
            }
            else {
                slope = (p_start.at<double>(0, 0) - p_end.at<double>(0, 0)) / (p_start.at<double>(1, 0) - p_end.at<double>(1, 0));
                rt = RepresentationType::X;
            }
            offset = p_start.at<double>(1, 0) - slope * p_start.at<double>(0, 0);
            if (rt == RepresentationType::Y)
                features.push_back(RANSACResult {slope, -1.0, offset, r2});
            else
                features.push_back(RANSACResult {-1.0, slope, offset, r2});
        }
    }

    return features;
}

/* std::vector<cv::Vec3d> extract_features(const std::vector<double>& ranges, int threshold)
{
    std::vector<cv::Point2f> points;
    double max_range = 1;
    for (size_t i = 0; i < ranges.size(); i++) {
        if (ranges[i] > 0.001) {
            if (ranges[i] > max_range)
                max_range = ranges[i];
            auto v = polar_to_cartesian(ranges[i], (M_PI_2 * i) / ranges.size());
            points.emplace_back((float) v[0], (float) v[1]);
        }
    }

    cv::Mat lines;
    std::vector<cv::Vec3d> lines3d;
    cv::HoughLinesPointSet(points, lines, 10, threshold, 0, max_range, 0.01, 0, M_PI / 2, M_PI / 360);
    lines.copyTo(lines3d);
    return lines;
}
 */
