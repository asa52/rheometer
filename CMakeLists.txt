cmake_minimum_required(VERSION 2.8.4)
set(CMAKE_TOOLCHAIN_FILE ${CMAKE_SOURCE_DIR}/cmake/ArduinoToolchain.cmake)
set(PROJECT_NAME rheometer)
project(${PROJECT_NAME})

set(${CMAKE_PROJECT_NAME}_SKETCH rheometer.ino)
set(${CMAKE_PROJECT_NAME}_BOARD due)
set(${CMAKE_PROJECT_NAME}_PORT COM3)
generate_arduino_firmware(${CMAKE_PROJECT_NAME})
