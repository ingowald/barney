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

add_library(barney-config INTERFACE)
target_include_directories(barney-config INTERFACE
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/barney/include
  ${CMAKE_CURRENT_LIST_DIR}/..
  )




embed_ptx(
  OUTPUT_TARGET      barney-spheres-ptx
  PTX_LINK_LIBRARIES owl::owl barney-config
  SOURCES            Spheres.dev.cu
)
embed_ptx(
  OUTPUT_TARGET      barney-triangles-ptx
  PTX_LINK_LIBRARIES owl::owl barney-config
  SOURCES            Triangles.dev.cu
)
embed_ptx(
  OUTPUT_TARGET      barney-traceRays-ptx
  PTX_LINK_LIBRARIES owl::owl barney-config
  SOURCES            traceRays.cu
)

add_library(barney SHARED
  barney.cpp
  DeviceContext.h
  DeviceContext.cu
  Context.h
  Context.cu
  DeviceGroup.h
  DeviceGroup.cpp
  LocalContext.h
  LocalContext.cu
  FrameBuffer.h
  FrameBuffer.cu
  LocalFB.h
  LocalFB.cu
  TiledFB.h
  TiledFB.cu
  Model.h
  Model.cu
  DataGroup.h
  DataGroup.cpp
  Spheres.h
  Spheres.cu
  Triangles.h
  Triangles.cu
  )
target_link_libraries(barney
  PUBLIC
  owl::owl 
  barney-config
  barney-spheres-ptx
  barney-triangles-ptx
  barney-traceRays-ptx  
  )
#target_include_directories(barney PUBLIC
#  )

set_target_properties(barney
  PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_USE_STATIC_CUDA_RUNTIME ON
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


