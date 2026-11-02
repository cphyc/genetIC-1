cmake_minimum_required(VERSION 3.18)
project(genetic_random LANGUAGES CXX)

find_package(Python REQUIRED COMPONENTS Development.Module)
find_package(GSL REQUIRED)
find_package(OpenMP)

Python_add_library(_random MODULE WITH_SOABI ../genetic/_random.cpp)
target_compile_features(_random PRIVATE cxx_std_17)
target_compile_options(_random PRIVATE -O3)
target_link_libraries(_random PRIVATE GSL::gsl)
if(OpenMP_CXX_FOUND)
  target_link_libraries(_random PRIVATE OpenMP::OpenMP_CXX)
endif()

install(TARGETS _random LIBRARY DESTINATION genetic RUNTIME DESTINATION genetic)
