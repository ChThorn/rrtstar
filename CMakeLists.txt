cmake_minimum_required(VERSION 3.10)
project(rrtstar)

# Change to C++11
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(rrtstar_planner
    src/rrt_star.cpp
    src/main.cpp
)

target_include_directories(rrtstar_planner
    PRIVATE
        ${PROJECT_SOURCE_DIR}/include
)

if(MSVC)
    target_compile_options(rrtstar_planner PRIVATE /W4)
else()
    target_compile_options(rrtstar_planner PRIVATE -Wall -Wextra -Wpedantic)
endif()