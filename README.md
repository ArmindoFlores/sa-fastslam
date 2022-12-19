# Autonomous Systems Project (2021-2022)

## Goal
The goal of this project is to implement the FastSLAM algorithm to enable a TurtleBot robot to map a room by itself using laser scans.

## C++ Version
To build the C++ version you will need the OpenCV library installed. Additionally, Boost is required to run the test program to parse JSON in the input file. [Here](https://docs.opencv.org/4.x/d7/d9f/tutorial_linux_install.html) are the instructions on how to install OpenCV and [here](https://www.boost.org/doc/libs/1_80_0/more/getting_started/unix-variants.html) is the same fot Boost.

### Build/Run Instructions
To build the Fast SLAM library, change into the `cpp` directory and run `make`. This will generate the executable `bin/main` as well was the static library `lib/libslam.a`. If, instead, only the library is needed, run `make lib`.

To build the node, go to the root directory of the project (`catkin_workspace`) and run `source devel/setup.bash` followed by `catkin_make`. You should only do this after building the __libslam__ library.

Finally, to run the node, run `rosrun fast_slam_cpp fast_slam_cpp_node` after bringing `roscore`.