# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


set(BARNEY_VERSION_MAJOR 1)
set(BARNEY_VERSION_MINOR 1)
set(BARNEY_VERSION_PATCH 2)

cmake_minimum_required(VERSION 3.24)

set(CMAKE_CUDA_USE_STATIC_CUDA_RUNTIME ON)

cmake_policy(SET CMP0104 NEW)
cmake_policy(SET CMP0079 NEW)
cmake_policy(SET CMP0076 NEW)
cmake_policy(SET CMP0063 NEW)

option(BARNEY_MPI "Enable MPI Support" OFF)

if (NOT (${CMAKE_CURRENT_SOURCE_DIR} STREQUAL ${CMAKE_SOURCE_DIR}))
  set(BARNEY_IS_SUBPROJECT ON)
else()
  set(BARNEY_IS_SUBPROJECT OFF)
endif()

set(CMAKE_BUILD_TYPE_INIT "Release")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)




# project command is required to come after cmake_minimum_required command.
set(BARNEY_VERSION ${BARNEY_VERSION_MAJOR}.${BARNEY_VERSION_MINOR}.${BARNEY_VERSION_PATCH})
project(Barney VERSION ${BARNEY_VERSION} LANGUAGES C CXX)
include(GNUInstallDirs)

option(USE_EXP "Experimental Pluggable Backend" OFF)
mark_as_advanced(USE_EXP)

if (USE_EXP)
  include(rtcore/exp_external_backends)
elseif (NOT BARNEY_DISABLE_CUDA) 
  include(CheckLanguage)
  check_language(CUDA)
  if (CMAKE_CUDA_COMPILER)
    if (WIN32)
      set(BARNEY_CUDA_ARCHITECTURES_INIT "all-major")
    else()
      set(BARNEY_CUDA_ARCHITECTURES_INIT "native")
    endif()
    set(CMAKE_CUDA_ARCHITECTURES
      "${BARNEY_CUDA_ARCHITECTURES_INIT}" CACHE STRING
      "Which CUDA architecture to build for")
    
    enable_language(CUDA)
    set(BARNEY_HAVE_CUDA ON)
  else()
    message(AUTHOR_WARNING "#barney: no CUDA compiler found; disabling cuda/optix backend")
    set(BARNEY_HAVE_CUDA OFF)
  endif()
else()
  message("explicitly disabling cuda")
  set(BARNEY_HAVE_CUDA OFF)
  set(CMAKE_CUDA_ARCHITECTURES)
endif()

if (CMAKE_BUILD_TYPE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Release" "Debug" 
    "MinSizeRel" "RelWithDebInfo")
endif()

if (BARNEY_HAVE_CUDA)
  message("#barney: building for CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
endif()
if (NOT BARNEY_IS_SUBPROJECT)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
endif()

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)


# ------------------------------------------------------------------
# owl - even if we build only for cpu (we we wouldn't use the owl
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
else()
  add_library(owl-config INTERFACE)
  target_include_directories(owl-config INTERFACE
    ${CMAKE_CURRENT_LIST_DIR}/submodules/owl/include
    )
endif()
# ------------------------------------------------------------------


# ------------------------------------------------------------------
add_library(barney_config INTERFACE)
target_link_libraries(barney_config
  INTERFACE
  owl-config
  )
target_include_directories(barney_config INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/3rdParty/nanovdb/include/>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/3rdParty/glm/include/>
)
# ------------------------------------------------------------------
# configure which backend(s) to build
# ------------------------------------------------------------------
if (USE_EXP)
  # leave it to the external plugins to configure themselves - not
  # currently working
elseif (BARNEY_HAVE_CUDA)
  option(BARNEY_BACKEND_CUDA "Enable (native-)CUDA Backend?" OFF)
  option(BARNEY_BACKEND_OPTIX "Enable OptiX Backend?" ON)
  option(BARNEY_BACKEND_EMBREE "Enable Embree Backend?" OFF)
else()
  set(BARNEY_BACKEND_CUDA OFF)
  set(BARNEY_BACKEND_OPTIX OFF)
  set(BARNEY_BACKEND_EMBREE ON)
endif()

# ==================================================================
option(BARNEY_DISABLE_DENOISING "DISable denoising" OFF)
option(BARNEY_HAVE_NANOVDB "Include Support for NanoVDB" ON)

option(BARNEY_USE_EXTERNAL_CUBQL "Use External CuBQL dir" OFF)
set(BARNEY_CUBQL_BVH_WIDTH 4 CACHE INT "CuBQL BVH width")


# ==================================================================
if (BARNEY_DISABLE_DENOISING)
  set(BARNEY_DISABLE_OIDN ON)
  set(BARNEY_OIDN_CPU OFF)
  set(BARNEY_OIDN_CPU OFF)
else()
  option(BARNEY_DISABLE_OIDN "DISable OIDN even if found?" OFF)
  if (NOT BARNEY_DISABLE_OIDN)
    find_package(OpenImageDenoise)
    if (OIDN_DEVICE_CPU)
      message("found OIDN *CPU* device")
      set(BARNEY_OIDN_CPU ON)
    endif()
    if (OIDN_DEVICE_GPU)
      message("found OIDN *GPU* device")
      set(BARNEY_OIDN_GPU ON)
    endif()
  endif()
  message("BARNEY_OIDN_CPU ${BARNEY_OIDN_CPU}")
endif()

# ------------------------------------------------------------------
# rtcore
# ------------------------------------------------------------------

if (NOT (TARGET cuBQL))
  if (BARNEY_HAVE_CUDA)
  else()
    set(CUBQL_DISABLE_CUDA ON)
  endif()
  if (BARNEY_USE_EXTERNAL_CUBQL)
    add_subdirectory(../cuBQL EXCLUDE_FROM_ALL builddir_cuBQL)
  else()
    add_subdirectory(submodules/cuBQL EXCLUDE_FROM_ALL)
  endif()
endif()

if (BARNEY_HAVE_CUDA)
  option(BARNEY_CUBQL_HOST "Use CUBQL host builder" OFF)
else()
  set(BARNEY_CUBQL_HOST ON)
endif()
if (BARNEY_CUBQL_HOST)
  target_compile_definitions(barney_config INTERFACE -DBARNEY_CUBQL_HOST=1)
  target_link_libraries(barney_config INTERFACE cuBQL)
endif()
target_link_libraries(barney_config
  INTERFACE
  cuBQL
)


# ------------------------------------------------------------------
# barney ray tracing core
# ------------------------------------------------------------------
add_subdirectory(rtcore)
  

# ------------------------------------------------------------------
# barney system, api, and mpi stuff
# ------------------------------------------------------------------
if (BARNEY_MPI)
  message("#barney: configuring mpi")
  find_package(MPI)
  if (MPI_FOUND)
    message("#barney: MPI found, including in default barney and banari builds")
  else()
    message("#barney: MPI build requested, but MPI not found... disabling")
    set(BARNEY_MPI OFF)
  endif()
endif()

add_subdirectory(barney)

# ------------------------------------------------------------------
# External field plugins
# ------------------------------------------------------------------
if(DEFINED BARNEY_EXTERNAL_FIELDS_DIR AND EXISTS "${BARNEY_EXTERNAL_FIELDS_DIR}/CMakeLists.txt")
  message("#barney: including external fields from ${BARNEY_EXTERNAL_FIELDS_DIR}")
  add_subdirectory("${BARNEY_EXTERNAL_FIELDS_DIR}" "${CMAKE_CURRENT_BINARY_DIR}/external_fields")
endif()

# ------------------------------------------------------------------
# ANARI device
# ------------------------------------------------------------------
message("configuring anari")
add_subdirectory(anari)



# ------------------------------------------------------------------
# cmake install stuff
# ------------------------------------------------------------------

set(BARNEY_CMAKE_INSTALL_DESTINATION
  ${CMAKE_INSTALL_LIBDIR}/cmake/barney-${BARNEY_VERSION}#${PROJECT_VERSION}
)

include(CMakePackageConfigHelpers)

configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/barney/common/barneyConfig.cmake.in"
  "${PROJECT_BINARY_DIR}/barneyConfig.cmake"
  INSTALL_DESTINATION
  ${BARNEY_CMAKE_INSTALL_DESTINATION}
)

message("BARNEY_VERSION ${BARNEY_VERSION}")
write_basic_package_version_file(
    "barneyConfigVersion.cmake"
    VERSION ${BARNEY_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
  ${CMAKE_CURRENT_BINARY_DIR}/barneyConfig.cmake
  ${CMAKE_CURRENT_BINARY_DIR}/barneyConfigVersion.cmake
  DESTINATION
  ${BARNEY_CMAKE_INSTALL_DESTINATION}
  )


install(DIRECTORY ${CMAKE_CURRENT_LIST_DIR}/cmake
  DESTINATION
    ${BARNEY_CMAKE_INSTALL_DESTINATION}
  FILES_MATCHING
    PATTERN *.cmake
    PATTERN Findbarney.cmake EXCLUDE
)

set(BARNEY_INSTALL_TARGETS barney barney_static)
if (TARGET barney_mpi)
  list(APPEND BARNEY_INSTALL_TARGETS barney_mpi barney_mpi_static)
endif()
if (TARGET anari_library_barney)
  list(APPEND BARNEY_INSTALL_TARGETS anari_library_barney)
endif()
if (TARGET anari_library_barney_mpi)
  list(APPEND BARNEY_INSTALL_TARGETS anari_library_barney_mpi)
endif()

install(
  TARGETS
  ${BARNEY_INSTALL_TARGETS}
  EXPORT barney-config
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  NAMELINK_SKIP
  # on Windows put the dlls into bin
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  # ... and the import lib into the devel package
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
install(EXPORT barney-config
  DESTINATION ${BARNEY_CMAKE_INSTALL_DESTINATION}
  NAMESPACE barney::
  )

# configures the public api
configure_file(
  ${PROJECT_SOURCE_DIR}/barney/include/barney.h
  include/barney/barney.h
)
if (BARNEY_MPI)
  configure_file(
    ${PROJECT_SOURCE_DIR}/barney/include/barney_mpi.h
    include/barney/barney_mpi.h
  )
endif()
# internal configurations that shoudln't be visible in external api
configure_file(
  ${PROJECT_SOURCE_DIR}/barney/common/barneyConfig.h.in
  # we put this into builddir/include where build will find it - but
  # this will *not* get installed
  include/barney/barneyConfig.h
  )

install(
  DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/barney
  DESTINATION ${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_INCLUDEDIR}
  )

