cmake_minimum_required(VERSION 3.14)
project(strategy_cpp)

set(CMAKE_CXX_STANDARD 17)

add_subdirectory(pybind11)
pybind11_add_module(strategy_cpp bindings.cpp)
