# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
# CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# ------------------------------------------------------------------
# global options
# ------------------------------------------------------------------

# ----------- enable/disable different devices we can build
option(BARNEY_BACKEND_CUDA  "Build (native) CUDA Backend" ON)
option(BARNEY_BACKEND_OPTIX "Build OptiX Backend"         ON)
option(BARNEY_BACKEND_CPU   "Build CPU Backend"           ON)
option(BARNEY_BACKEND_HIP   "Build CPU Backend"           ON)
option(BARNEY_BACKEND_HIPRT "Build CPU Backend"           ON)
option(BARNEY_MPI           "Enable MPI Support"          OFF)

# ------------------------------------------------------------------
# cmake project/output config
# ------------------------------------------------------------------
set(BARNEY_VERSION_MAJOR 1)
set(BARNEY_VERSION_MINOR 2)
set(BARNEY_VERSION_PATCH 0)
set(BARNEY_VERSION ${BARNEY_VERSION_MAJOR}.${BARNEY_VERSION_MINOR}.${BARNEY_VERSION_PATCH})

# ----------- which versions of owl/cubql we expect for correct build 
set(EXPECTED_OWL_VERSION "1.2.9")
set(EXPECTED_CUBQL_VERSION "1.3.1")

cmake_minimum_required(VERSION 3.24)

set(CMAKE_CUDA_USE_STATIC_CUDA_RUNTIME ON)

cmake_policy(SET CMP0104 NEW)
cmake_policy(SET CMP0079 NEW)
cmake_policy(SET CMP0076 NEW)
cmake_policy(SET CMP0063 NEW)

if (NOT (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR}))
  set(BARNEY_IS_SUBPROJECT ON)
else()
  set(BARNEY_IS_SUBPROJECT OFF)
endif()

set(CMAKE_BUILD_TYPE_INIT "Release")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)

set(BARNEY_PROJECT_NAME="barney")
project(Barney VERSION ${BARNEY_VERSION} LANGUAGES C CXX)

if (CMAKE_BUILD_TYPE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Release" "Debug" 
    "MinSizeRel" "RelWithDebInfo")
endif()
include(GNUInstallDirs)

if (NOT BARNEY_IS_SUBPROJECT)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED  TRUE)
set(CMAKE_CXX_VISIBILITY_PRESET  hidden)
set(CMAKE_CUDA_VISIBILITY_PRESET hidden)

# barney's own cmake modules (e.g. Findhiprt.cmake for the HIPRT backend)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

find_package(anari 0.15.0 COMPONENTS code_gen)

option(BARNEY_AUTO_GENERATE_QUERIES "auto-generate the anari device queries" OFF)




# ==================================================================
# external projects
# ==================================================================

# ------------------------------------------------------------------
# owl - even if we build only for cpu (where we wouldn't use the owl
# _library_) we'd still need the owl common classes ... so this
# _always_ gets included first
# ------------------------------------------------------------------
if (BARNEY_HAVE_CUDA)
  set(OWL_CUDA_STATIC ON)
  message(" checking ${CMAKE_CURRENT_LIST_DIR}/../optix")
  if (EXISTS "${CMAKE_CURRENT_LIST_DIR}/submodules/optix/include/optix.h")
    set(OptiX_INSTALL_DIR ${CMAKE_CURRENT_LIST_DIR}/submodules/optix)
  elseif (EXISTS "${CMAKE_CURRENT_LIST_DIR}/../optix/include/optix.h")
    message(" found ${CMAKE_CURRENT_LIST_DIR}/../optix")
    set(OptiX_INSTALL_DIR ${CMAKE_CURRENT_LIST_DIR}/../optix)
  else()
  endif()
  set(OWL_BUILD_VIEWER OFF)
  add_subdirectory(submodules/owl EXCLUDE_FROM_ALL)
  if ((NOT (DEFINED OWL_VERSION)) OR (${OWL_VERSION} VERSION_LESS ${EXPECTED_OWL_VERSION}))
    message(FATAL_ERROR " OWL version is too old. make sure to update your owl submodule")
endif()
else()
  add_library(owl-config INTERFACE)
  target_include_directories(owl-config INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}/submodules/owl/include
  )
endif()
# ------------------------------------------------------------------


# ------------------------------------------------------------------
# cubql
# ------------------------------------------------------------------

add_subdirectory(submodules/cuBQL EXCLUDE_FROM_ALL)

# ------------------------------------------------------------------
# mpi
# ------------------------------------------------------------------

if (BARNEY_MPI)
  message("#barney: configuring MPI")
  find_package(MPI)
  if (MPI_FOUND)
    message("#barney: MPI found, including in default barney and banari builds")
  else()
    message("#barney: MPI build requested, but MPI not found... disabling")
    set(BARNEY_MPI OFF)
  endif()
endif()


# ==================================================================
# actual build config
# ==================================================================

#------------------------------------------------------------------
# include rtcore subdir - this will declare the different backend
# configs. configs that are disabled via BARNEY_BACKEND_XYZ=OFF will
# not be *attempted* to be built; those that are enabled in this
# config will be attempted to build, but may still get disabled by
# rtcore/ if certain requirements are not met (eg, not having CUDA or
# HIP compiler(s)
# ------------------------------------------------------------------
set(BARNEY_NS ${BARNEY_PROJECT_NAME})

# build_barney(config device) -> tests if config exists as a target,
# and if so build device (as well as device_mpi, if enabled) with this
# config.
macro(build_barney)
  foreach(backend ${ARGN})
    string(TOUPPER BARNEY_BACKEND_${backend} FLAG_NAME)
    if (NOT (${FLAG_NAME}))
      message("#barney: backend '${backend}' disabled by user request")
    else()
      add_subdirectory(barney build_barney_${backend})
      add_subdirectory(anari  build_banari_${backend})
    endif()
  endforeach()
endmacro()

build_barney(cpu)
build_barney(cuda)
build_barney(optix)
build_barney(hip)
build_barney(hiprt)



# ==================================================================
option(BARNEY_DISABLE_DENOISING "DISable denoising" OFF)
option(BARNEY_HAVE_NANOVDB "Include Support for NanoVDB" ON)
option(BARNEY_USE_MULTI_SCATTERING
  "Enable volumetric multi-scattering (HG phase, volume bounce tracking)"
  OFF)

# ------------------------------------------------------------------
# External field plugins
# ------------------------------------------------------------------
if(DEFINED BARNEY_EXTERNAL_FIELDS_DIR AND EXISTS "${BARNEY_EXTERNAL_FIELDS_DIR}/CMakeLists.txt")
  message("#barney: including external fields from ${BARNEY_EXTERNAL_FIELDS_DIR}")
  add_subdirectory("${BARNEY_EXTERNAL_FIELDS_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/external_fields")
endif()

# ------------------------------------------------------------------
# cmake install stuff
# ------------------------------------------------------------------

set(BARNEY_CMAKE_INSTALL_DESTINATION
  ${CMAKE_INSTALL_LIBDIR}/cmake/barney-${BARNEY_VERSION}
)

include(CMakePackageConfigHelpers)


set(BARNEY_INSTALL_TARGETS)
foreach(backend ${ARGN})
  if (TARGET anari_library_${BARNEY_PLATFORM_NAME}_${backend})
    list(APPEND BARNEY_INSTALL_TARGETS
      anari_library_${BARNEY_PLATFORM_NAME}_${backend})
  endif()
  if (TARGET anari_library_${BARNEY_PLATFORM_NAME}_${backend}_mpi)
    list(APPEND BARNEY_INSTALL_TARGETS
      anari_library_${BARNEY_PLATFORM_NAME}_${backend}_mpi)
  endif()
endforeach()

install(
  TARGETS
  ${BARNEY_INSTALL_TARGETS}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  NAMELINK_SKIP
  # on Windows put the dlls into bin
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  # ... and the import lib into the devel package
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

