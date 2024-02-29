cmake_minimum_required(VERSION 3.0)
project(fitting)

set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_CXX_FLAGS "-std=c++14 -O3")

list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)

include_directories("/usr/include/eigen3")

find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

find_package(Ceres REQUIRED)
include_directories(${CERES_INCLUDE_DIRS})


find_package(G2O REQUIRED)
include_directories(${G2O_INCLUDE_DIRS})

# add_executable(Newton Newton.cpp)
# target_link_libraries(Newton ${OpenCV_LIBS} ${G2O_LIBRARIES}) 

# add_executable(gaussNewton gaussNewton.cpp)
# target_link_libraries(gaussNewton ${OpenCV_LIBS})

add_executable(ceresCurveFitting ceresCurveFitting.cpp)
target_link_libraries(ceresCurveFitting ${CERES_LIBRARIES} ${OpenCV_LIBRARIES})

add_executable(g2o g2o.cpp)
target_link_libraries(g2o ${OpenCV_LIBS} ${G2O_CORE_LIBRARY} ${G2O_STUFF_LIBRARY})