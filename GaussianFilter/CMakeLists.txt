cmake_minimum_required(VERSION 3.20)

project(CudaGauss)

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
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -DCMAKE_BUILD_TYPE=Release")
    list(APPEND CUDA_NVCC_FLAGS "-lineinfo;-ccbin;/usr/bin/g++;--compiler-options;-O2;-D_FORCE_INLINES;-DVERBOSE_NOT") 
  endif()
endif()

set(cuda_sources
  gaussian.cu
  gaussian.h
)  

# set(sources
#   main.cpp
# )

include_directories(
  ${CMAKE_CURRENT_SOURCE_DIR} 
)

SET(CUDA_SEPARABLE_COMPILATION ON)

cuda_add_executable(cuda_gauss ${cuda_sources} OPTIONS -arch=sm_86)

set_target_properties(cuda_gauss PROPERTIES
  COMPILE_FLAGS "${EXTRA_CXX_FLAGS}"			   
)

target_link_libraries(cuda_gauss 
  ${CUDA_cudadevrt_LIBRARY} 
  ${OpenCV_LIBS}
  ${TIFF_LIBRARIES}
  ${CUDA_cublas_LIBRARY} 
  ${CUDA_cublasLt_LIBRARY} 
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

