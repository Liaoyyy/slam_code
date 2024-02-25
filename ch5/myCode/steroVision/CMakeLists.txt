project(steroVision)

find_package(Pangolin REQUIRED)

add_executable(steroVision steroVision.cpp)
target_link_libraries(steroVision ${OpenCV_LIBS})
target_link_libraries(steroVision ${Pangolin_LIBS})