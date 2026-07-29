# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
# CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# 'cudaCommon' is a single set of sources, but it might get compiled
# differently for differnet backends - different compiler, different
# defines, and in particular different namespace - so we actually have
# t have a different cmake target for each 'isntantiation' of this
# code.
find_package(CUDAToolkit REQUIRED)

add_library(barney_rtc_cudaCommon_${backend} STATIC
  Device.cu
  Texture.cu
  TextureData.cu
#  cuda-helper.cpp
)
target_link_libraries(barney_rtc_cudaCommon_${backend} PUBLIC
  barney_config
  barney_config_${backend}
  #  $<BUILD_INTERFACE:cuBQL_cuda_float3_static>
  cuBQL
  CUDA::cudart_static
#  owl::owl_static
)
target_include_directories(barney_rtc_cudaCommon_${backend} PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}
)
rtc_library_properties(barney_rtc_cudaCommon_${backend})
#set_property(
#  TARGET barney_rtc_optix 
#  APPEND
#  PROPERTY INTERFACE_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
#set_property(
#  TARGET barney_rtc_optix 
#  APPEND
#  PROPERTY INTERFACE_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda#>)
#if (BARNEY_MPI)
#  add_library(barney_mpi_optix ${MPI_SOURCES})
#  target_link_libraries(barney_mpi_optix PUBLIC barney_optix MPI::MPI_C)
#  rtc_library_properties(barney_mpi_optix)
#endif()
