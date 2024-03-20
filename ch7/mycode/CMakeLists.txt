cmake_minimum_required(VERSION 3.0)

project(mycode)
set(CMAKE_BUILD_TYPE Debug)
find_package(OpenCV REQUIRED)

include_directories("/usr/include/eigen3/")

add_executable(orb_self orb_self.cpp)
add_executable(ICP_problem ICP_problem.cpp)

target_link_libraries(orb_self ${OpenCV_LIBS})
target_link_libraries(ICP_problem ${OpenCV_LIBS} fmt)