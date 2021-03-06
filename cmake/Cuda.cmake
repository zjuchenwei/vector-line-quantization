# configure cuda

find_package(CUDA QUIET REQUIRED)
if(CUDA_FOUND)
    include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
 
    list(APPEND CUDA_LINKER_LIBS ${CUDA_CUDART_LIBRARY} ${CUDA_curand_LIBRARY} ${CUDA_CUBLAS_LIBRARIES})
else(CUDA_FOUND)
    message(STATUS "Could not locate cuda, disabling cuda support.")
    set(BUILD_WITH_GPU OFF)
    return()
endif(CUDA_FOUND)
if(NOT DEFINED CUDA_TOOLKIT_SAMPLE_DIR)
  set(CUDA_TOOLKIT_SAMPLE_DIR ${CUDA_TOOLKIT_ROOT_DIR}/samples)
endif()
set(FAISS_CUDA_ADDITIONAL_INC_PATH ${CUDA_INCLUDE_DIRS} ${CUDA_TOOLKIT_SAMPLE_DIR}/common/inc/)
include_directories(${FAISS_CUDA_ADDITIONAL_INC_PATH})
# set cuda flags
if (CMAKE_BUILD_TYPE STREQUAL "Debug")
    list(APPEND CUDA_NVCC_FLAGS "-arch=sm_52;-D_FORCE_INLINES;-D_MWAITXINTRIN_H_INCLUDED;-D__STRICT_ANSI__;-std=c++11;-DVERBOSE;-g;-lineinfo;-Xcompiler;-ggdb")
else()
    list(APPEND CUDA_NVCC_FLAGS "-arch=sm_52;-D_FORCE_INLINES;-D_MWAITXINTRIN_H_INCLUDED;-D__STRICT_ANSI__;-std=c++11;-lm;-DVERBOSE;-O3;-DNDEBUG;-Xcompiler;-DNDEBU")
endif()
set(CUDA_PROPAGATE_HOST_FLAGS OFF)
