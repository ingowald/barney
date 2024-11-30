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
if ((NOT WIN32) AND (NOT CMAKE_BUILD_TYPE) AND (NOT CMAKE_CONFIGURATION_TYPES))
#  message(STATUS "Setting build type to 'Release' as none was specified.")
#  set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
  # Set the possible values of build type for cmake-gui
  set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
    "MinSizeRel" "RelWithDebInfo")
endif()


# iw - do NOT use an 'option' - they can't do strings
set(BARNEY_CUDA_ARCHITECTURES "auto" CACHE STRING
  "Which CUDA architecture to build for")
if (BARNEY_CUDA_ARCHITECTURES STREQUAL "auto")
  if (NOT CMAKE_CUDA_ARCHITECTURES)
    # 'all-major', since cmake-3.23
    set(BARNEY_CUDA_ARCHITECTURES all-major)
  else()
    set(BARNEY_CUDA_ARCHITECTURES S{CMAKE_CUDA_ARCHITECTURES})
  endif()
  message("#barney: cuda-archs set to auto, selected '${BARNEY_CUDA_ARCHITECTURES}'")
else()
  message("#barney: explicit cuda archs '${BARNEY_CUDA_ARCHITECTURES}' requested by user")
endif()
  

# project command is required to come after cmake_minimum_required command.
project(Barney VERSION 0.7.0 LANGUAGES C CXX CUDA)

if (NOT BARNEY_IS_SUBPROJECT)
  set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
  set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
  set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR})
endif()

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if (NOT CUDA_SOURCE)
  macro(CUDA_SOURCE fileName)
    set_source_files_properties(${fileName}
      PROPERTIES LANGUAGE CUDA
      )
    set_source_files_properties(${fileName}
      PROPERTIES CUDA_ARCHITECTURES ${BARNEY_CUDA_ARCHITECTURES}
      )
  endmacro()
endif()



# ---------------------------------------------ge---------------------
# subprojects we depend on
# ------------------------------------------------------------------
if (NOT (TARGET owl::owl))
  add_subdirectory(submodules/owl EXCLUDE_FROM_ALL)
endif()


# TODO: add option to find if installed
option(BARNEY_CUBQL_HOST "Use CUBQL host builder" off)
if (NOT (TARGET cuBQL_cuda_float3))
  add_subdirectory(submodules/cuBQL EXCLUDE_FROM_ALL)
endif()

# ------------------------------------------------------------------
# barney itself
# ------------------------------------------------------------------
add_subdirectory(barney)

# ------------------------------------------------------------------
# ANARI device
# ------------------------------------------------------------------
add_subdirectory(anari)

