# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
# CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# "backend" variable gets set by parent dir, and is one of either
# 'cpu', 'cuda', 'optix', 'hip' or 'hiprt'

include(${backend}.cmake)

if (NOT (TARGET barney_config_${backend}))
  message(AUTHOR_WARNING "Could not set up / configure '${backend}' backend")
  return()
endif()

# ==================================================================
add_library(barney_rtc_${backend} INTERFACE)
target_include_directories(barney_rtc_${backend} INTERFACE
  ${PROJECT_SOURCE_DIR}
)
# ==================================================================
if (BARNEY_BACKEND_CUDA OR BARNEY_BACKEND_OPTIX OR BARNEY_BACKEND_HIPRT)
  if (BARNEY_HAVE_HIP)
    rtc_mark_hip_sources(
      cudaCommon/Device.cu
      cudaCommon/Texture.cu
      cudaCommon/TextureData.cu)
  endif()
  set(BACKEND_CUDA_COMMON_SOURCES
    cudaCommon/Device.h
    cudaCommon/Device.cu
    cudaCommon/Texture.h
    cudaCommon/Texture.cu
    cudaCommon/TextureData.h
    cudaCommon/TextureData.cu
    cudaCommon/cuda-helper.cpp
    )
  list(APPEND RTCORE_SOURCES ${BACKEND_CUDA_COMMON_SOURCES})

  add_library(barney_rtc_cuda_common STATIC
    ${BACKEND_CUDA_COMMON_SOURCES}
  )

  target_link_libraries(barney_rtc_cuda_common PUBLIC
    barney_rtc_config
    )
    if (TARGET rtcore_exp_config)
      target_link_libraries(barney_rtc_cuda_common PUBLIC
        rtcore_exp_config
      )
  endif()
set_target_properties(barney_rtc_cuda_common
  PROPERTIES
  CXX_VISIBILITY_PRESET hidden
  POSITION_INDEPENDENT_CODE ON
  CUDA_RESOLVE_DEVICE_SYMBOLS  ON
  )
  rtc_library_properties(barney_rtc_cuda_common)
else()
  add_library(barney_rtc_cuda_common INTERFACE)
endif()

# ==================================================================
if (BARNEY_BACKEND_OPTIX)
  add_library(barney_rtc_optix STATIC
    optix/Denoiser.h
    optix/Denoiser.cpp
    optix/Device.h
    optix/Device.cpp
    optix/Group.h
    optix/Group.cpp
    optix/Geom.h
    optix/Geom.cpp
    optix/Buffer.h
    optix/Buffer.cpp
    )
  target_link_libraries(barney_rtc_optix PUBLIC
    barney_rtc_config
    barney_rtc_cuda_common
    owl_static
    )
  target_compile_definitions(barney_rtc_optix
    PUBLIC -DBARNEY_RTC_OPTIX=1)
  rtc_library_properties(barney_rtc_optix)
  if (BARNEY_DISABLE_DENOISING)
    target_compile_definitions(barney_rtc_optix
      PUBLIC -DOPTIX_DISABLE_DENOISING=1)
  endif()
else()
  # dummy "no-op" library; empty, but it exists, so code below can
  # just link to it without having to do any checking
  add_library(barney_rtc_optix INTERFACE) 
endif()

# ==================================================================
if (BARNEY_BACKEND_CUDA)
  if (BARNEY_HAVE_HIP)
    rtc_mark_hip_sources(
      cuda/cudaDevice.cu
      cuda/cudaGroup.cu)
  endif()
  set(BACKEND_CUDA_SOURCES
    cuda/Device.h
    cuda/cudaDevice.cu
    cuda/Buffer.h
    cuda/Buffer.cpp
    cuda/Group.h
    cuda/cudaGroup.cu
    cuda/Geom.h
    cuda/Geom.cpp
    cuda/TraceKernel.h
    cuda/TraceKernel.cpp
    cuda/GeomType.h
    cuda/GeomType.cpp
  )
  add_library(barney_rtc_cuda STATIC
    ${BACKEND_CUDA_SOURCES}
  )
  list(APPEND RTCORE_SOURCES ${BACKEND_CUDA_SOURCES})
  target_link_libraries(barney_rtc_cuda PUBLIC
    barney_rtc_config
    barney_rtc_cuda_common
  )
  if (USE_EXP)
    target_link_libraries(barney_rtc_cuda PUBLIC
      rtcore_exp_config
    )
  endif()

  
  target_compile_definitions(barney_rtc_cuda
    PUBLIC -DBARNEY_RTC_CUDA=1)
  rtc_library_properties(barney_rtc_cuda)

else()
  # dummy "no-op" library; empty, but it exists, so code below can
  # just link to it without having to do any checking
  add_library(barney_rtc_cuda INTERFACE)
endif()


# ==================================================================
if (BARNEY_BACKEND_HIPRT)
  message("Enabling HIPRT hardware-RT backend")
  # the hiprt backend's sources are HIP TUs (they launch kernels and pull in
  # HIPRT's inline device traversal); compile them with hipcc and -fgpu-rdc so
  # the function-pointer SBT device-links exactly as the software backend.
  set_source_files_properties(
    hiprt/Device.cpp
    hiprt/Group.cpp
    hiprt/Buffer.cpp
    hiprt/Geom.cpp
    hiprt/GeomType.cpp
    hiprt/TraceKernel.cpp
    PROPERTIES LANGUAGE HIP COMPILE_OPTIONS "-fgpu-rdc")
  add_library(barney_rtc_hiprt STATIC
    hiprt/Device.h
    hiprt/Device.cpp
    hiprt/Buffer.h
    hiprt/Buffer.cpp
    hiprt/Geom.h
    hiprt/Geom.cpp
    hiprt/GeomType.h
    hiprt/GeomType.cpp
    hiprt/Group.h
    hiprt/Group.cpp
    hiprt/TraceKernel.h
    hiprt/TraceKernel.cpp
    )
  target_link_libraries(barney_rtc_hiprt PUBLIC
    barney_rtc_config
    barney_rtc_cuda_common
    hiprt::hiprt
    )
  target_compile_definitions(barney_rtc_hiprt
    PUBLIC -DBARNEY_RTC_HIPRT=1)
  rtc_library_properties(barney_rtc_hiprt)
else()
  add_library(barney_rtc_hiprt INTERFACE)
endif()


# ==================================================================
# ==================================================================
add_library(barney_rtc INTERFACE)
if (BARNEY_BACKEND_OPTIX)
  target_link_libraries(barney_rtc INTERFACE barney_rtc_optix)
endif()
if (BARNEY_BACKEND_EMBREE)
  target_link_libraries(barney_rtc INTERFACE barney_rtc_embree)
endif()
if (BARNEY_BACKEND_CUDA)
  target_link_libraries(barney_rtc INTERFACE barney_rtc_cuda)
endif()
if (BARNEY_BACKEND_HIPRT)
  target_link_libraries(barney_rtc INTERFACE barney_rtc_hiprt)
endif()


if (COMMAND configure_cuda_source)
  # macro already defined
else()
  macro(configure_cuda_source src)
    if (BARNEY_HAVE_HIP)
      # on AMD GPUs the same .cu/.dev.cu sources are compiled with hipcc;
      # -fgpu-rdc gives relocatable device code so the function-pointer SBT
      # dispatch resolves at device-link time (the analogue of nvcc -rdc=true).
      get_filename_component(ext "${src}" EXT)
      if (("${ext}" STREQUAL ".cu") OR ("${ext}" STREQUAL ".dev.cu"))
        set_source_files_properties(${src} PROPERTIES
          LANGUAGE HIP
          COMPILE_OPTIONS "-fgpu-rdc"
        )
      endif()
    elseif (NOT BARNEY_HAVE_CUDA)
      get_filename_component(ext "${src}" EXT)
      if (("${ext}" STREQUAL ".cu") OR ("${ext}" STREQUAL ".dev.cu"))
        message("setting language of ${src} to CXX")
        set_source_files_properties(${src} PROPERTIES LANGUAGE CXX)
        if (NOT WIN32)
          set_source_files_properties(${src} PROPERTIES COMPILE_OPTIONS "-xc++")
        endif()
      endif()
    else()
      get_filename_component(ext "${src}" EXT)
      if (("${ext}" STREQUAL ".cu") OR ("${ext}" STREQUAL ".dev.cu"))
        set_source_files_properties(${src}
          # -rdc=true is for relocatable code, can be resolved at link time.
          PROPERTIES COMPILE_OPTIONS "--extended-lambda;-rdc=true"
        )
      endif()
    endif()
  endmacro()
endif()

foreach(src ${RTCORE_SOURCES})
  configure_cuda_source(${src})
endforeach()
