cmake_minimum_required(VERSION 3.21)

project(pds_project LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# set(CMAKE_HIP_ARCHITECTURES "gfx1101")

add_executable(${CMAKE_PROJECT_NAME} main.cpp)

set_source_files_properties(main.cpp PROPERTIES LANGUAGE HIP)
