# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
# CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

add_library(barney_optix ${HOST_SOURCES})
foreach(src  ${DEVICE_PROGRAM_SOURCES})
  get_filename_component(basename "${src}" NAME_WE)
  embed_ptx(
    OUTPUT_TARGET      barney-${basename}-ptx
    PTX_LINK_LIBRARIES barney_config_ptx barney_rtc_optix
    SOURCES            ${src}
  )
  target_link_libraries(barney_optix PRIVATE barney-${basename}-ptx)
endforeach()
rtc_library_properties(barney_optix)
target_link_libraries(barney_optix PUBLIC
  barney_config
  barney_rtc_optix
  $<BUILD_INTERFACE:cuBQL_cuda_float3_static>
)
set_property(
  TARGET barney_optix 
  APPEND
  PROPERTY INTERFACE_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
set_property(
  TARGET barney_optix 
  APPEND
  PROPERTY INTERFACE_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
if (BARNEY_MPI)
  add_library(barney_mpi_optix ${MPI_SOURCES})
  target_link_libraries(barney_mpi_optix PUBLIC barney_optix MPI::MPI_C)
  rtc_library_properties(barney_mpi_optix)
endif()
