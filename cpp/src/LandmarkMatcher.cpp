#include "LandmarkMatcher.hpp"

LandmarkMatcher::LandmarkMatcher(const cv::Mat& Qt, int minimum_observations, double distance_threshold, int max_invalid_landmarks)
    : Qt(Qt)
    , minimum_observations(minimum_observations)
    , distance_threshold(distance_threshold)
    , max_invalid_landmarks(max_invalid_landmarks)
{}
LandmarkMatcher::LandmarkMatcher(const LandmarkMatcher& l)
    : Qt(l.Qt)
    , minimum_observations(l.minimum_observations)
    , distance_threshold(l.distance_threshold)
    , max_invalid_landmarks(l.max_invalid_landmarks)
    , filters(l.filters)
{}
LandmarkMatcher::LandmarkMatcher(LandmarkMatcher&& l)
    : Qt(std::move(l.Qt))
    , minimum_observations(std::move(l.minimum_observations))
    , distance_threshold(std::move(l.distance_threshold))
    , max_invalid_landmarks(std::move(l.max_invalid_landmarks))
    , filters(std::move(l.filters))
{}
LandmarkMatcher& LandmarkMatcher::operator = (const LandmarkMatcher& l)
{
    l.Qt.copyTo(Qt);
    minimum_observations = l.minimum_observations;
    distance_threshold = l.distance_threshold;
    max_invalid_landmarks = l.max_invalid_landmarks;
    filters.clear();
    for (const auto& ekf_pair : l.filters)
        filters.emplace_back(ekf_pair);
    return *this;
}
LandmarkMatcher& LandmarkMatcher::operator = (LandmarkMatcher&& l)
{
    Qt = std::move(l.Qt);
    minimum_observations = l.minimum_observations;
    distance_threshold = l.distance_threshold;
    max_invalid_landmarks = l.max_invalid_landmarks;
    filters = std::move(filters);
    return *this;
}

std::optional<KalmanFilter> LandmarkMatcher::observe(const Landmark& landmark, std::function<cv::Mat(double, double, double)> h_func, const cv::Vec3d& pose)
{
    double threshold_distance_squared = distance_threshold*distance_threshold;
    std::optional<KalmanFilter> match {};
    
    // Convert observed landmark to world space
    cv::Vec2d params = landmark.get_parameters();
    double theta = params[1] + pose[2];
    Landmark worldspace_landmark {params[0] + pose[0] * std::cos(theta) + pose[1] * std::sin(theta), theta};
    cv::Vec2d new_params {worldspace_landmark.get_parameters()};
    cv::Vec2d position {worldspace_landmark.closest_point(pose[0], pose[1])};

    std::size_t closest = filters.max_size();
    double closest_distance = -1.0;

    // Try to find a previously seen landmark that matches the observed one
    for (std::size_t i = 0; i < filters.size(); i++) {
        auto pair = filters[i];
        cv::Vec2d position_diff {position - pair.first->get_landmark().closest_point(pose[0], pose[1])};
        double landmark_distance_squared = position_diff.dot(position_diff);

        if (landmark_distance_squared < threshold_distance_squared && (closest_distance < 0 || landmark_distance_squared < closest_distance)) {
            // Found possible match
            closest = i;
            closest_distance = landmark_distance_squared;
        }
    }
    if (closest_distance >= 0) {
        // Found a match
        cv::Mat H {h_func(pose[0], pose[1], new_params[1])};
        filters[closest].first->update(new_params, H);
        filters[closest].second++;
        if (filters[closest].second >= (std::size_t) minimum_observations)
            match = *filters[closest].first;
    }
    else {
        // No matches were found, add a new landmark
        cv::Mat H_inv {h_func(pose[0], pose[1], new_params[1]).inv()}, H_inv_transposed;
        cv::transpose(H_inv, H_inv_transposed);

        filters.emplace_back(std::make_shared<KalmanFilter>(
            worldspace_landmark,
            H_inv_transposed * Qt * H_inv,
            Qt
        ), 1);
    }
    /*if (filters.size() > (std::size_t) max_invalid_landmarks) {
        std::sort(filters.begin(), filters.end(), [](const std::pair<std::shared_ptr<KalmanFilter>, std::size_t> f1, const std::pair<std::shared_ptr<KalmanFilter>, std::size_t> f2){
            return f1.second > f2.second;
        });
        std::size_t diff = filters.size() - max_invalid_landmarks;
        for (std::size_t i = 0; i < diff; i++)
            filters.pop_back();
    }*/

    return match;
}

std::vector<Landmark> LandmarkMatcher::get_map() const
{
    std::vector<Landmark> result;
    for (const auto& ekf_pair : filters) {
        if (ekf_pair.second > (std::size_t) minimum_observations)
            result.push_back(ekf_pair.first->view_landmark());
    }
    return result;
}

std::vector<Landmark> LandmarkMatcher::get_full_map() const
{
    std::vector<Landmark> result;
    for (const auto& ekf_pair : filters)
        result.push_back(ekf_pair.first->view_landmark());
    return result;
}