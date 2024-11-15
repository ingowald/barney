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

if(CUDA_TOOLKIT_ROOT_DIR)
  # Somebody called find_cuda--they are trying to use the deprecated build.
  message(WARNING "CUDA_TOOLKIT_ROOT_DIR is set. "
    "This usually means that a parent project has called find_cuda, which "
    "has been deprecated in CMake since 2018 and is no longer supported by BARNEY. "
    "Instead, the parent project is required to use 'intermediate' or 'modern' "
    "CMake CUDA style with first-class CUDA language support (i.e., "
    "project(foo LANGUAGES C CXX CUDA)). "
  )
endif()

set(CMAKE_BUILD_TYPE_INIT "Release")
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

#set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug;Release")
if ((NOT WIN32) AND (NOT CMAKE_BUILD_TYPE) AND (NOT CMAKE_CONFIGURATION_TYPES))
#  message(STATUS "Setting build type to 'Release' as none was specified.")
#  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()

# project command is required to come after cmake_minimum_required command.
project(Barney VERSION 1.1.5 LANGUAGES C CXX CUDA)

if (NOT BARNEY_IS_SUBPROJECT)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
endif()


set(BARNEY_CUDA_ARCHITECTURES "auto" CACHE
  STRING "which CUDA arch to use for building cuBQL" )

if (BARNEY_CUDA_ARCHITECTURES STREQUAL "auto")
  # leave CMAKE_CUDA_ARCHITECTURES to whatever the may may or may not have set
else()
  set(CMAKE_CUDA_ARCHITECTURES ${BARNEY_CUDA_ARCHITECTURES})
endif()

if (NOT (DEFINED CMAKE_CUDA_ARCHITECTURES))

  # new cmake policy: make sure that CMAKE_CUDA_ARCHITECTURES is set
  # to _something_, even if it's only "OFF". iw - seems to not apply on ubuntu 20.04!?
  if (NOT CMAKE_VERSION VERSION_LESS "3.17")
    set(CMAKE_CUDA_ARCHITECTURES OFF)
  endif()
endif()


set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

macro(CUDA_SOURCE fileName)
  set_source_files_properties(${fileName}
    PROPERTIES LANGUAGE CUDA
    )
endmacro()

option(EXP_SAUCY "Saucy exping" OFF)
mark_as_advanced(EXP_SAUCY)
if (EXP_SAUCY)
  add_subdirectory(../saucy buildDir_saucy EXCLUDE_FROM_ALL)
endif()
# ------------------------------------------------------------------
# subprojects we depend on
# ------------------------------------------------------------------
if (NOT (TARGET owl::owl))
  add_subdirectory(submodules/owl EXCLUDE_FROM_ALL)
endif()


# swtich to 'find' if installed
option(BARNEY_USE_CUBQL "Use CUBQL library" true)
if (BARNEY_USE_CUBQL)
  add_subdirectory(submodules/cuBQL EXCLUDE_FROM_ALL)
endif()

# ------------------------------------------------------------------
# barney itself
# ------------------------------------------------------------------
add_subdirectory(barney)

# ------------------------------------------------------------------
# sample distributed viewer/renderer
# ------------------------------------------------------------------
#add_subdirectory(hayMaker)

# ------------------------------------------------------------------
# ANARI device
# ------------------------------------------------------------------
add_subdirectory(anari)

