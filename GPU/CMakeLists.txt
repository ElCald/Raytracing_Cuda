# CMake minimal requiered version
cmake_minimum_required(VERSION 3.18)

# RayTracing
project(RayTracing LANGUAGES CXX CUDA)

# Turn on warnings for the compilator
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Release)

# Adding subdirectories (src, tests)
add_subdirectory(src/)
add_subdirectory(tests/)