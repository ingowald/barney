# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
# CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(barney_rtc_optix STATIC
  Buffer.cpp
  Device.cpp
  Group.cpp
  Denoiser.cpp
  Geom.cpp
)
target_link_libraries(barney_rtc_optix PUBLIC
  barney_config
  #  barney_rtc_optix
  #  $<BUILD_INTERFACE:cuBQL_cuda_float3_static>
  cuBQL
  owl::owl_static
)
target_include_directories(barney_rtc_optix PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}
)


#add_library(barney_rtc_optix ${HOST_SOURCES})
#foreach(src  ${DEVICE_PROGRAM_SOURCES})
#  get_filename_component(basename "${src}" NAME_WE)
#  embed_ptx(
#    OUTPUT_TARGET      barney-${basename}-ptx
#    PTX_LINK_LIBRARIES barney_config_ptx barney_rtc_optix
#    SOURCES            ${src}
#  )
#  target_link_libraries(barney_rtc_optix PRIVATE barney-${basename}-ptx)
#endforeach()

rtc_library_properties(barney_rtc_optix)
set_property(
  TARGET barney_rtc_optix 
  APPEND
  PROPERTY INTERFACE_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
set_property(
  TARGET barney_rtc_optix 
  APPEND
  PROPERTY INTERFACE_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
#if (BARNEY_MPI)
#  add_library(barney_mpi_optix ${MPI_SOURCES})
#  target_link_libraries(barney_mpi_optix PUBLIC barney_optix MPI::MPI_C)
#  rtc_library_properties(barney_mpi_optix)
#endif()
