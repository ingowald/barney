# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
# CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ------------------------------------------------------------------
# 'cpu' backend: instantiates all the barney classes, device
# programs, etcpp - except the api itself - in an embree configuration
# that'll run on the host.
# ------------------------------------------------------------------

# ==================================================================
message("enabling CPU backend (via embree)")

macro(rtc_library_properties lib)
endmacro()

macro(rtc_build_device_sources libname)
  add_library(${libname} STATIC ${ARGN})
  rtc_library_properties(${libname})
endmacro()

if (APPLE AND ${CMAKE_SYSTEM_PROCESSOR} STREQUAL "arm64")
  set(COMPILE_FOR_ARM ON)
elseif (UNIX AND ${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
  set(COMPILE_FOR_ARM ON)
else()
  set(COMPILE_FOR_ARM OFF)
endif()

option(EMBREE_GEOMETRY_CURVE          "" OFF)
option(EMBREE_GEOMETRY_GRID           "" OFF)
option(EMBREE_GEOMETRY_INSTANCE       "" ON)
option(EMBREE_GEOMETRY_INSTANCE_ARRAY "" ON)
option(EMBREE_GEOMETRY_POINT          "" ON)
option(EMBREE_GEOMETRY_QUAD           "" ON)
option(EMBREE_GEOMETRY_SUBDIVISION    "" OFF)
option(EMBREE_GEOMETRY_TRIANGLE       "" ON)
option(EMBREE_GEOMETRY_USER           "" ON)
if (COMPILE_FOR_ARM)
  option(EMBREE_ISA_NEON           "" OFF)
  option(EMBREE_ISA_NEON2X         "" ON)
else()
  option(EMBREE_ISA_AVX            "" OFF)
  option(EMBREE_ISA_AVX2           "" OFF)
  option(EMBREE_ISA_AVX512         "" OFF)
  option(EMBREE_ISA_SSE42          "" ON)
  option(EMBREE_ISA_SSE2           "" OFF)
endif()
option(EMBREE_ISPC_SUPPORT             "" OFF)
option(EMBREE_STATIC_LIB               "" ON)
option(EMBREE_STAT_COUNTERS            "" OFF)
option(EMBREE_SYCL_SUPPORT             "" OFF)
option(EMBREE_TUTORIALS                "" OFF)
option(EMBREE_TUTORIALS_GLFW           "" OFF)
set(EMBREE_TASKING_SYSTEM "INTERNAL" CACHE STRING "" FORCE)
set(EMBREE_MAX_INSTANCE_LEVEL_COUNT "1" CACHE STRING "" FORCE)
set(EMBREE_API_NAMESPACE embree_for_barney CACHE STRING "" FORCE)
add_subdirectory(../submodules/embree builddir_rtc_embree EXCLUDE_FROM_ALL)
add_library(embree_local INTERFACE)
target_link_libraries(embree_local INTERFACE embree tasking)
rtc_library_properties(embree)
rtc_library_properties(tasking)
rtc_library_properties(sys)
target_compile_definitions(embree_local INTERFACE TASKING_INTERNAL)
target_include_directories(embree_local INTERFACE
  ${CMAKE_CURRENT_SOURCE_DIR}/../submodules/embree/
  ${CMAKE_CURRENT_SOURCE_DIR}/../submodules/embree/common
)
target_include_directories(embree PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../submodules/embree/
  ${CMAKE_CURRENT_SOURCE_DIR}/../submodules/embree/common
)
target_include_directories(tasking PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../submodules/embree/
  ${CMAKE_CURRENT_SOURCE_DIR}/../submodules/embree/common
)
if (WIN32)
  target_compile_options(embree_local 
    INTERFACE
    /D__SSE__ /D__SSE2__ /D__SSE3__ /D__SSSE3__ /D__SSE4_1__ /D__SSE4_2__
  )
  rtc_library_properties(embree_local)
elseif(COMPILE_FOR_ARM)
  target_compile_options(embree_local
    INTERFACE
    -flax-vector-conversions -fsigned-char
  )
endif()

message("adding barney_rtc_cpu")
add_library(barney_rtc_cpu STATIC
  ${CMAKE_CURRENT_LIST_DIR}/cpu/Float16.cpp
  ${CMAKE_CURRENT_LIST_DIR}/cpu/Compute.cpp
  ${CMAKE_CURRENT_LIST_DIR}/cpu/Device.cpp
  ${CMAKE_CURRENT_LIST_DIR}/cpu/Buffer.cpp
  ${CMAKE_CURRENT_LIST_DIR}/cpu/Texture.cpp
  ${CMAKE_CURRENT_LIST_DIR}/cpu/GeomType.cpp
  ${CMAKE_CURRENT_LIST_DIR}/cpu/Geom.cpp
  ${CMAKE_CURRENT_LIST_DIR}/cpu/Triangles.cpp
  ${CMAKE_CURRENT_LIST_DIR}/cpu/UserGeom.cpp
  ${CMAKE_CURRENT_LIST_DIR}/cpu/Group.cpp
  ${CMAKE_CURRENT_LIST_DIR}/cpu/Denoiser.cpp
)
target_include_directories(barney_rtc_cpu PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}
  )
target_link_libraries(barney_rtc_cpu PUBLIC
#  barney_rtc_common
  embree_local
)
target_compile_definitions(barney_rtc_cpu PUBLIC
  -DBARNEY_NS=${BARNEY_NS}
)
rtc_library_properties(barney_rtc_cpu)
if (BARNEY_OIDN_CPU)
  target_link_libraries(barney_rtc_cpu PUBLIC OpenImageDenoise)
endif()
target_link_libraries(barney_rtc_cpu PUBLIC
  $<BUILD_INTERFACE:cuBQL_cpu_float3_static>
)

macro(rtc_configure_source)
  foreach(src ${ARGN})
    get_filename_component(ext "${src}" EXT)
    if (ext STREQUAL ".cu")
      set_source_files_properties(${src} PROPERTIES
        LANGUAGE CXX
      )
    endif()
  endforeach()
endmacro()

