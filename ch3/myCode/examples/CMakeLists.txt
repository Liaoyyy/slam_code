cmake_minimum_required(VERSION 3.0)
project(plotTrajectory)
add_executable(plotTrajectory plotTrajectory.cpp)

find_package(Pangolin)
if(Pangolin_FOUND)
    target_link_libraries(plotTrajectory ${Pangolin_LIBRARIES})
else(Pangolin_FOUND)
    message(FATAL_ERROR "Pangolin not found")
endif(Pangolin_FOUND)

include_directories("/usr/include/eigen3")
