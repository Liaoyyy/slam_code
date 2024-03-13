cmake_minimum_required(VERSION 3.0)

project(mycode)

find_package(OpenCV REQUIRED)

add_executable(orb_self orb_self.cpp)

target_link_libraries(orb_self ${OpenCV_LIBS})