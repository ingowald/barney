# ======================================================================== #
# Copyright 2023-2024 Ingo Wald                                            #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #

set(BARNEY_VERSION_MAJOR 0)
set(BARNEY_VERSION_MINOR 8)
set(BARNEY_VERSION_PATCH 0)

cmake_minimum_required(VERSION 3.12)

if(${CMAKE_VERSION} VERSION_GREATER_EQUAL 3.18)
  cmake_policy(SET CMP0104 NEW)
endif()

if (WIN32)
  # MPI is off by default on windows. whoever wants to use mpi under windows
  # probably better know what he's doing, anyway
  option(BARNEY_MPI "Enable MPI Support" OFF)
else()
  option(BARNEY_MPI "Enable MPI Support" ON)
endif()

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

option(BARNEY_DISABLE_CUDA "Explicitly DISable CUDA even if available?" OFF)
if (NOT BARNEY_DISABLE_CUDA) 
  include(CheckLanguage)
  check_language(CUDA)
  if (CMAKE_CUDA_COMPILER)
    message("#barney: CUDA _FOUND_! building both cuda and host libs")
    # set cmake cuda arch - this ahs to happen _before_
    # enable_language(cuda), or cmake will default to '52', which is
    # almost certainly going to be wrong
    if (NOT CMAKE_CUDA_ARCHITECTURES)
      if (${CMAKE_VERSION} VERSION_LESS 3.24)
        # 'native' only appeared in cmake 3.24
        message(AUTHOR_WARNING
	  " You're using a cmake < 3.24; we cannot use the 'native' or \n"
	  " 'all-major' targets with that, so will default to '70;80'\n"
	  " If you want/need to build for other architectures, either\n"
	  " set this to something else in the cmake dialog, or call\n"
	  " cmake with an explicit list such as, for example\n"
	  "     cmake -DCMAKE_CUDA_ARCHITECTURES=\"75;80;86\" ...\n"
	  )
        set(BARNEY_CUDA_ARCHITECTURES_INIT "70;80")
      else()
        set(BARNEY_CUDA_ARCHITECTURES_INIT "native")
      endif()
      set(CMAKE_CUDA_ARCHITECTURES
        "${BARNEY_CUDA_ARCHITECTURES_INIT}" CACHE STRING
        "Which CUDA architecture to build for")
    endif()
    
    enable_language(CUDA)
    set(BARNEY_HAVE_CUDA ON)
  else()
    message(AUTHOR_WARNING
      "#barney: CUDA not found - you need to install CUDA to build barney")
    set(BARNEY_HAVE_CUDA OFF)
  endif()
else()
  set(BARNEY_HAVE_CUDA OFF)
endif()

if (CMAKE_BUILD_TYPE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Release" "Debug" 
    "MinSizeRel" "RelWithDebInfo")
endif()

message("#barney: building for CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
if (NOT BARNEY_IS_SUBPROJECT)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
endif()

#set(CMAKE_CXX_STANDARD 17)
#set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
 

# ------------------------------------------------------------------
# owl - even if we build only for cpu (we we wouldn't use the owl
# _library_) we'd still need the owl common classes ... so this
# _always_ gets included first
# ------------------------------------------------------------------
add_subdirectory(submodules/owl EXCLUDE_FROM_ALL)
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
  )
configure_file(
    ${PROJECT_SOURCE_DIR}/barney/include/barney.h
    include/barney/barney.h
    )

# ------------------------------------------------------------------


# ------------------------------------------------------------------
# configure which backend(s) to build
# ------------------------------------------------------------------
if (BARNEY_HAVE_CUDA)
  option(BARNEY_BACKEND_CUDA "Enable Cuda Backend?" ON)
  option(BARNEY_BACKEND_OPTIX "Enable OptiX Backend?" ON)
  
#  target_compile_definitions(barney_config INTERFACE
#    # we have cuda - as a *compiler*
#    -DBARNEY_HAVE_CUDA=1
#    )
else()
  message(AUTHOR_WARNING "#barney: no CUDA compiler found; disabling both cuda and optix backends")
endif()

option(BARNEY_BACKEND_EMBREE "Enable Embree Backend?" ON)

#if (TARGET cuBQL_cuda_float3)
#if (target cuBQL_cuda_float3)


# ------------------------------------------------------------------
# CUBQL
# ------------------------------------------------------------------

if (NOT (TARGET cuBQL))
  add_subdirectory(submodules/cuBQL EXCLUDE_FROM_ALL)
endif()

if (BARNEY_HAVE_CUDA)
  option(BARNEY_CUBQL_HOST "Use CUBQL host builder" OFF)
else()
  set(BARNEY_CUBQL_HOST ON)
endif()
if (BARNEY_CUBQL_HOST)
  target_compile_definitions(barney_config INTERFACE -DBARNEY_CUBQL_HOST=1)
  target_link_libraries(barney_config INTERFACE cuBQL)
  set(CUBQL_LIB cuBQL_cpu_float3)
else()
  set(CUBQL_LIB cuBQL_cuda_float3)
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
# ANARI device
# ------------------------------------------------------------------
add_subdirectory(anari)









# ------------------------------------------------------------------
# cmake install stuff
# ------------------------------------------------------------------

if (APPLE)
  list(APPEND CMAKE_INSTALL_RPATH "@loader_path")
  list(APPEND CMAKE_INSTALL_RPATH "@loader_path/../${CMAKE_INSTALL_LIBDIR}")
elseif(UNIX)
  list(APPEND CMAKE_INSTALL_RPATH "$ORIGIN")
  list(APPEND CMAKE_INSTALL_RPATH "$ORIGIN/../${CMAKE_INSTALL_LIBDIR}")
endif()


set(BARNEY_CMAKE_INSTALL_DESTINATION
  ${CMAKE_INSTALL_LIBDIR}/cmake/barney-${BARNEY_VERSION}#${PROJECT_VERSION}
)

install(EXPORT barney_Exports
  DESTINATION ${BARNEY_CMAKE_INSTALL_DESTINATION}
  NAMESPACE barney::
  )

project(anari_headers)
project_add_library(INTERFACE)
project_include_directories(
INTERFACE
  $<BUILD_INTERFACE:${GENERATED_HEADER_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/barney>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_LIST_DIR}/include>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
if (WIN32)
  project_compile_definitions(INTERFACE -D_USE_MATH_DEFINES)
endif()


include(CMakePackageConfigHelpers)

configure_package_config_file(
  "${PROJECT_SOURCE_DIR}/cmake/barneyConfig.cmake.in"
  "${PROJECT_BINARY_DIR}/barneyConfig.cmake"
INSTALL_DESTINATION
  ${BARNEY_CMAKE_INSTALL_DESTINATION}
)

message("BARNEY_VERSION ${BARNEY_VERSION}")
write_basic_package_version_file(
    "barneyConfigVersion.cmake"
#    VERSION ${PROJECT_VERSION}
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

install(
  TARGETS barney
#  owl-config
#  owl_static
#  cuBQL
#  barney_config
#  barney_rtc
#  barney_rtc_common
#  barney_rtc_cuda_common
#  barney_rtc_optix
#  barney-triangles-ptx
#  barney-spheres-ptx
#  barney-cylinders-ptx
#  barney-capsules-ptx
  EXPORT barney_Exports
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    NAMELINK_SKIP
  # on Windows put the dlls into bin
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  # ... and the import lib into the devel package
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)

if (BARNEY_MPI)
install(
  TARGETS barney_mpi
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  NAMELINK_SKIP
  # on Windows put the dlls into bin
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  # ... and the import lib into the devel package
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )
endif()

install(
  DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/include/barney
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
)


