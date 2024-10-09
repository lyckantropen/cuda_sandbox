add_executable(histogram histogram.cu host_histogram.cpp)
target_link_libraries(histogram cuda_proxy)
