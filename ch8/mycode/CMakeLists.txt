cmake_minimum_required(VERSION  3.0)
project(ch8_mycode)
add_executable(learnOpenCV learnOpenCV.cpp)
add_executable(LK  LK.cpp LK_driver.cpp)

find_package(OpenCV REQUIRED)
target_link_libraries(learnOpenCV ${OpenCV_LIBS})
target_link_libraries(LK ${OpenCV_LIBS})
