cmake_minimum_required(VERSION 3.10)

project(kokoro.cpp)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(EXECUTABLE_OUTPUT_PATH ${CMAKE_BINARY_DIR}/bin)
set(CMAKE_CXX_STANDARD 20)

include_directories(/usr/local/include /opt/homebrew/include)
link_directories(/usr/local/lib /opt/homebrew/lib)

find_package(Boost REQUIRED COMPONENTS program_options)

add_subdirectory(src)