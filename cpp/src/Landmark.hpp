#ifndef _H_LANDMARK_
#define _H_LANDMARK_

#include <iostream>
#include <opencv2/core.hpp>

/*
    This class represents a linear landmark, characterized by an angle and its
    distance to the origin.
*/
class Landmark {
public:
    /*
        Create a new Landmark object

        \param r the landmark's distance to the origin
        \param theta the landmark's angle to the X axis
    */
    Landmark(double r=0, double theta=0);
    /*
        Create a new Landmark object from the coefficients of the line equation
        ax + by + c = 0
    */
    Landmark(double a, double b, double c);
    Landmark(const Landmark&) = default;
    Landmark(Landmark&&);
    Landmark& operator = (const Landmark&);
    Landmark& operator = (Landmark&&);

    /*
        Update this landmark's parameters

        \param r this landmark's distance to the origin
        \param theta this landmark's angle to the X axis
    */
    void update_parameters(double r, double theta);

    /*
        Get this landmark's parameters

        \return A vector [r theta]
    */
    cv::Vec2d get_parameters() const;

    /*
        Compute the closest point on the landmark to specified position

        \param position point to compute
        \return The computed position
    */
    cv::Vec2d closest_point(const cv::Vec2d& position) const;

    /*
        Compute the closest point on the landmark to specified position

        \param x x coordinate of the point to compute
        \param y y coordinate of the point to compute
        \return The computed position
    */
    cv::Vec2d closest_point(double x, double y) const;

    /*
        The equation describing the line this landmark represents
        
        \return A vector [a b c] where ax+by+c=0 is the line's equation
    */
    cv::Vec3d equation() const;

    friend std::ostream& operator << (std::ostream&, const Landmark&);

private:
    double r, theta;
};

#endif