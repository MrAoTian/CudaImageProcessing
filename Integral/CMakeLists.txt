cmake_minimum_required(VERSION 3.10)

project(CudaIntegral)

# set(CPACK_GENERATOR "ZIP")
# include(CPack)

find_package(OpenCV REQUIRED)

find_package(CUDA)
if (NOT CUDA_FOUND)
  message(STATUS "CUDA not found. Project will not be built.")
endif(NOT CUDA_FOUND)

if (WIN32)
  set(EXTRA_CXX_FLAGS "/DVERBOSE /D_CRT_SECURE_NO_WARNINGS ")
  list(APPEND CUDA_NVCC_FLAGS "-arch=sm_86;--compiler-options;-O2;-DVERBOSE") 
endif()
if (UNIX)
  if (APPLE)
    set(EXTRA_CXX_FLAGS "-DVERBOSE -msse2")
    list(APPEND CUDA_NVCC_FLAGS "-arch=sm_86;--compiler-options;-O2;-DVERBOSE") 
  else()
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
    list(APPEND CUDA_NVCC_FLAGS "-lineinfo;-ccbin;/usr/bin/g++;--compiler-options;-O2;-D_FORCE_INLINES;-DVERBOSE_NOT") 
  endif()
endif()

set(cuda_sources
  integral_d.cu
  integral_d.h
)  

set(sources
  main.cpp
  cuda_utils.h
)

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR} 
)

SET(CUDA_SEPARABLE_COMPILATION ON)

cuda_add_executable(cuda_integral ${cuda_sources} ${sources} OPTIONS -arch=sm_86)

set_target_properties(cuda_integral PROPERTIES
  COMPILE_FLAGS "${EXTRA_CXX_FLAGS}"			   
)

target_link_libraries(cuda_integral ${CUDA_cudadevrt_LIBRARY} ${OpenCV_LIBS}
  ${CUDA_nppc_LIBRARY} 
  ${CUDA_nppial_LIBRARY} 
  ${CUDA_nppicc_LIBRARY} 
  ${CUDA_nppcom_LIBRARY} 
  ${CUDA_nppidei_LIBRARY} 
  ${CUDA_nppif_LIBRARY} 
  ${CUDA_nppig_LIBRARY} 
  ${CUDA_nppim_LIBRARY} 
  ${CUDA_nppist_LIBRARY} 
  ${CUDA_nppisu_LIBRARY} 
  ${CUDA_nppitc_LIBRARY} 
  ${CUDA_npps_LIBRARY} 
)


