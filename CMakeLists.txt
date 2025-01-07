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

#set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug;Release")
#if ((NOT WIN32) AND (NOT CMAKE_BUILD_TYPE) AND (NOT CMAKE_CONFIGURATION_TYPES))
#  message(STATUS "Setting build type to 'Release' as none was specified.")
#  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
# Set the possible values of build type for cmake-gui
#if (CMAKE_BUILD_TYPE)
#  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Release" "Debug" 
#    "MinSizeRel" "RelWithDebInfo")
#endif()



# project command is required to come after cmake_minimum_required command.
project(Barney VERSION 0.7.0 LANGUAGES C CXX)
include(CheckLanguage)
check_language(CUDA)
if (CMAKE_CUDA_COMPILER)
  message("#barney: CUDA _FOUND_! building both cuda and host libs")
  enable_language(CUDA)
else()
  message(AUTHOR_WARNING
    "#barney: CUDA not found - you need to install CUDA to build barney")
endif()

if (CMAKE_BUILD_TYPE)
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Release" "Debug" 
    "MinSizeRel" "RelWithDebInfo")
endif()


# ==================================================================
# configure cuda archs. cmake < 3.24 doesnt' have 'native' and
# 'all-major'. and if we're built as a subproject, we really do
# want the app to specify what to use, since only the app will
# know what the rest of the app uses.
# ==================================================================
if ((NOT CMAKE_CUDA_ARCHITECTURES)
    OR
    ((${CMAKE_VERSION} VERSION_LESS 3.24) AND ("${CMAKE_CUDA_ARCHITECTURES}" STREQUAL "52"))
    )
  if (${CMAKE_VERSION} VERSION_LESS 3.24)
    set(BARNEY_CUDA_ARCHITECTURES_INIT "70;80")
  else()
    set(BARNEY_CUDA_ARCHITECTURES_INIT "native")
  endif()
  set(BARNEY_CUDA_ARCHITECTURES
    "${BARNEY_CUDA_ARCHITECTURES_INIT}" CACHE STRING
    "Which CUDA architecture to build for")
  set(CMAKE_CUDA_ARCHITECTURES "${BARNEY_CUDA_ARCHITECTURES}")
endif()

message("#barney: building for CMAKE_CUDA_ARCHITECTURES=${CMAKE_CUDA_ARCHITECTURES}")
if (NOT BARNEY_IS_SUBPROJECT)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
endif()

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT (COMMAND CUDA_SOURCE))
  macro(CUDA_SOURCE fileName)
#    set_source_files_properties(${fileName}
#      PROPERTIES LANGUAGE CUDA
#      )
#    set_source_files_properties(${fileName}
#      PROPERTIES
#      CUDA_RESOLVE_DEVICE_SYMBOLS ON
#      CUDA_SEPARABLE_COMPILATION ON
#      CUDA_ARCHITECTURES "${BARNEY_CUDA_ARCHITECTURES}"
#      )
  endmacro()
endif()

# ------------------------------------------------------------------
# subprojects we depend on
# ------------------------------------------------------------------
if (NOT (TARGET owl::owl))
  set(OWL_CUDA_STATIC ON)
  add_subdirectory(submodules/owl EXCLUDE_FROM_ALL)
endif()


# TODO: add option to find if installed
option(BARNEY_CUBQL_HOST "Use CUBQL host builder" off)
if (NOT (TARGET cuBQL))
  add_subdirectory(submodules/cuBQL EXCLUDE_FROM_ALL)
endif()

# ------------------------------------------------------------------
# barney itself
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
#add_subdirectory(anari)

