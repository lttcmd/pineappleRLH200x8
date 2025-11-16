cmake_minimum_required(VERSION 3.18)
project(ofc_cpp LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
  message(STATUS "pybind11 not found via CONFIG; falling back to module mode")
  include(FetchContent)
  FetchContent_Declare(
    pybind11
    GIT_REPOSITORY https://github.com/pybind/pybind11.git
    GIT_TAG v2.11.1
  )
  FetchContent_MakeAvailable(pybind11)
endif()

add_subdirectory(cpp)


