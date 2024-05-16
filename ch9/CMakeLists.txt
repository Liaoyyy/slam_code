cmake_minimum_required(VERSION 2.8)

project(bundle_adjustment)
set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-O3 -std=c++14")

include_directories("/usr/include/suitesparse/")

# list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
if (POLICY CMP0072)
  set(OpenGL_GL_PREFERENCE LEGACY)
endif()

find_package(g2o REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(Sophus REQUIRED)
# find_package(CSparse REQUIRED)

#SET(G2O_LIBS g2o_csparse_extension g2o_stuff g2o_core cxsparse)
#SET(G2O_LIBS /usr/local/lib/libg2o_csparse_extension.so /usr/local/lib/libg2o_stuff.so cxsparse)
SET(G2O_LIBS g2o::g2o_ceres_ad g2o::stuff g2o::opengl_helper g2o::core 
    g2o::g2o_hierarchical_library g2o::g2o_simulator_library   
    g2o::types_slam2d g2o::types_slam2d_addons  
    g2o::solver_pcg g2o::solver_dense g2o::solver_slam2d_linear 
    g2o::solver_structure_only g2o::solver_csparse g2o::csparse_extension g2o::solver_cholmod g2o::solver_eigen 
    cxsparse)

include_directories(${PROJECT_SOURCE_DIR} ${EIGEN3_INCLUDE_DIR} ${CSPARSE_INCLUDE_DIR})

add_library(bal_common common.cpp)
add_executable(bundle_adjustment_g2o bundle_adjustment_g2o.cpp)
add_executable(bundle_adjustment_ceres bundle_adjustment_ceres.cpp)

target_link_libraries(bundle_adjustment_ceres ${CERES_LIBRARIES} bal_common fmt)
target_link_libraries(bundle_adjustment_g2o ${G2O_LIBS} bal_common fmt)
