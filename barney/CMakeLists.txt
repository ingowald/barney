# ======================================================================== #
# Copyright 2023-2023 Ingo Wald                                            #
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

add_library(barney SHARED
  barney.cpp
  Context.h
  Context.cpp
  LocalContext.h
  LocalContext.cu
  LocalFB.h
  LocalFB.cu
  )
target_link_libraries(barney
  PUBLIC
  mori)
target_include_directories(barney PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/include
  )

# ------------------------------------------------------------------
find_package(MPI REQUIRED)

add_library(barney_mpi SHARED
  mpi/impl.cpp
  mpi/MPIWrappers.h
  mpi/MPIWrappers.cpp
  mpi/MPIContext.h
  mpi/MPIContext.cpp
  mpi/DistFB.h
  mpi/DistFB.cu
  )
target_link_libraries(barney_mpi PUBLIC
  barney
  MPI::MPI_CXX
  )
target_include_directories(barney_mpi PUBLIC
  ${CMAKE_CURRENT_LIST_DIR}/include
  ${CMAKE_CURRENT_LIST_DIR}/..
  )
target_compile_definitions(barney_mpi PUBLIC -DBARNEY_MPI=1)


