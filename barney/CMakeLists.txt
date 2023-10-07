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
  Context.cu
  LocalContext.h
  LocalContext.cu
  FrameBuffer.h
  FrameBuffer.cu
  LocalFB.h
  LocalFB.cu
  Model.h
  Model.cu
  )
target_link_libraries(barney
  PUBLIC
  mori)
target_include_directories(barney PUBLIC
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/barney/include
  )

# ------------------------------------------------------------------
find_package(MPI REQUIRED)

add_library(barney_mpi SHARED
  mpiAPI.cpp
  MPIWrappers.h
  MPIWrappers.cpp
  MPIContext.h
  MPIContext.cpp
  DistFB.h
  DistFB.cu
  )
target_link_libraries(barney_mpi PUBLIC
  barney
  MPI::MPI_CXX
  )
target_compile_definitions(barney_mpi PUBLIC -DBARNEY_MPI=1)


