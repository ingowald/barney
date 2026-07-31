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
  $<BUILD_INTERFACE:barney_config_${backend}>
  cuBQL
  CUDA::cudart_static
)
target_include_directories(barney_rtc_cudaCommon_${backend} PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}
)
rtc_library_properties(barney_rtc_cudaCommon_${backend})
