cmake_minimum_required(VERSION 3.0)

include_directories("/usr/include/eigen3")

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})

add_subdirectory(imageBasics)
add_subdirectory(steroVision)
