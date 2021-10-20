cmake_minimum_required(VERSION 3.13)

add_subdirectory("test_1_initial_cupy_reference/")
add_subdirectory("test_2_simple_python_to_c_caster/")
add_subdirectory("test_3_simple_c_to_python_caster/")
add_subdirectory("test_4_simple_bidirect_caster/")
add_subdirectory("test_5_simple_pointer_caster/")
add_subdirectory("test_6_cuda_fft_algorithm/")
add_subdirectory("test_7_cupy_reference_management/")
add_subdirectory("test_8_cupy_reference_cuda_interface/")

