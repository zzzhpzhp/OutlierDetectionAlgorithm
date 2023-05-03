cmake_minimum_required(VERSION 3.5)
project(COF CXX)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenCV REQUIRED)

add_executable(cof main.cpp)
target_link_libraries(cof ${OpenCV_LIBS})