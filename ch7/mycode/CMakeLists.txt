cmake_minimum_required(VERSION 3.0)
project(mycode)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "-std=c++14 -O3")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
message("${PROJECT_SOURCE_DIR}")

include_directories("/usr/include/eigen3")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})


find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})

add_executable(orb_self orb_self.cpp)
add_executable(ICP_problem ICP_problem.cpp)
add_executable(PnP_problem PnP_problem.cpp)

target_link_libraries(orb_self ${OpenCV_LIBS})
target_link_libraries(ICP_problem ${OpenCV_LIBS} fmt)
target_link_libraries(PnP_problem ${OpenCV_LIBS} ${CERES_LIBRARIES})

add_executable(ICP_ceres ICP_ceres_version.cpp)
target_link_libraries(ICP_ceres ${OpenCV_LIBS} ${CERES_LIBRARIES})
