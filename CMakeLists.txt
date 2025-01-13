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


# project command is required to come after cmake_minimum_required command.
project(Barney VERSION 0.8.0 LANGUAGES C CXX)

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
else()
  message(AUTHOR_WARNING
    "#barney: CUDA not found - you need to install CUDA to build barney")
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
  set(OWL_CUDA_STATIC OFF)
  #set(OWL_CUDA_STATIC ON)
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
add_subdirectory(anari)

