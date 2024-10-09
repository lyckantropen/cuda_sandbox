add_executable(reduce reduce.cu)
target_link_libraries(reduce cuda_proxy)
