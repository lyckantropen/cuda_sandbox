cmake_minimum_required(VERSION 3.20 FATAL_ERROR)

if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
  set(CMAKE_CUDA_ARCHITECTURES 70)
endif()

project(cuda_sandbox LANGUAGES CXX CUDA)

# download xtl with FetchContent
include(FetchContent)
FetchContent_Declare(
  xtl
  GIT_REPOSITORY git@github.com:xtensor-stack/xtl.git
  GIT_TAG 0.7.7
)
FetchContent_MakeAvailable(xtl)

# download xtensor with FetchContent
FetchContent_Declare(
  xtensor
  GIT_REPOSITORY git@github.com:xtensor-stack/xtensor.git
  GIT_TAG 0.25.0
)
FetchContent_MakeAvailable(xtensor)

add_library(cuda_proxy INTERFACE)
target_compile_features(cuda_proxy INTERFACE cxx_std_20)
target_link_libraries(cuda_proxy INTERFACE xtl xtensor)

subdirs(vector_addition)
subdirs(simplify_path)
subdirs(matrix_multiplication)
subdirs(histogram)
subdirs(reduce)

if (PROJECT_IS_TOP_LEVEL)
    # Create symlink to compile_commands.json for IDE to pick it up
    execute_process(
        COMMAND ${CMAKE_COMMAND} -E create_symlink
            ${CMAKE_BINARY_DIR}/compile_commands.json
            ${CMAKE_CURRENT_SOURCE_DIR}/compile_commands.json
    )
endif()