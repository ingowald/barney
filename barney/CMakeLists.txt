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
target_link_libraries(barney-config
  INTERFACE
  cuBQL_interface
  )
target_include_directories(barney-config INTERFACE
  ${PROJECT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}/barney/include
  ${CMAKE_CURRENT_LIST_DIR}/..
  )




# ray gen program that launches kernel to trace rays
embed_ptx(
  OUTPUT_TARGET      barney-traceRays-ptx
  PTX_LINK_LIBRARIES owl::owl barney-config
  SOURCES            kernels/traceRays.cu
  )
# ------------------------------------------------------------------
# surface geometry:
# ------------------------------------------------------------------
# ray-sphere geom intersectoin
embed_ptx(
  OUTPUT_TARGET      barney-spheres-ptx
  PTX_LINK_LIBRARIES owl::owl barney-config
  SOURCES            geometry/Spheres.dev.cu
  )
# ray triangles geom intersection
embed_ptx(
  OUTPUT_TARGET      barney-triangles-ptx
  PTX_LINK_LIBRARIES owl::owl barney-config
  SOURCES            geometry/Triangles.dev.cu
  )
# ray cylinders geom intersection
embed_ptx(
  OUTPUT_TARGET      barney-cylinders-ptx
  PTX_LINK_LIBRARIES owl::owl barney-config
  SOURCES            geometry/Cylinders.dev.cu
  )

# ------------------------------------------------------------------
# unstucture mesh variants:
# ------------------------------------------------------------------
embed_ptx(
  OUTPUT_TARGET      barney-UMeshRTXObjectSpace-ptx
  PTX_LINK_LIBRARIES owl::owl barney-config
  SOURCES            umesh/RTXObjectSpace.dev.cu
)
embed_ptx(
  OUTPUT_TARGET      barney-AWT-ptx
  PTX_LINK_LIBRARIES owl::owl barney-config
  SOURCES            umesh/AWT.dev.cu
)
embed_ptx(
  OUTPUT_TARGET      barney-UMeshMCAccelerator-ptx
  PTX_LINK_LIBRARIES owl::owl barney-config
  SOURCES            umesh/UMeshMCAccelerator.dev.cu
)

add_library(barney STATIC
  common/barney-common.h
  
  # actual device kernels for CUDA ray gen and shading
  kernels/generateRays.cu
  kernels/shadeRays.cu
  # all the barney "plumbing" stuff on the host
  DeviceContext.h
  DeviceContext.cpp
  Context.h
  Context.cpp
  DeviceGroup.h
  DeviceGroup.cpp
  LocalContext.h
  LocalContext.cpp
  # all frame buffer related stuff (DistFB in barney_mpi)
  fb/FrameBuffer.h
  fb/FrameBuffer.cpp
  fb/LocalFB.h
  fb/TiledFB.h
  fb/TiledFB.cu
  # model/group/data group handling
  Model.h
  Model.cpp
  Group.h
  Group.cpp
  DataGroup.h
  DataGroup.cpp
  # actual geometry types
  geometry/Geometry.h
  geometry/Geometry.cpp
  geometry/Spheres.h
  geometry/Spheres.cpp
  geometry/Triangles.h
  geometry/Triangles.cpp
  geometry/Cylinders.h
  geometry/Cylinders.cpp
  # volume data 
  volume/Volume.h
  volume/Volume.cpp
  volume/MCAccelerator.h
  volume/TransferFunction.h
  volume/TransferFunction.cpp
  volume/MCGrid.h
  volume/MCGrid.cu
  volume/MCAccelerator.h
  # *unstructured* volumes
  umesh/UMeshField.h
  umesh/UMeshField.cu
  umesh/CUBQLFieldSampler.h
  umesh/CUBQLFieldSampler.cpp
  umesh/RTXObjectSpace.h
  umesh/RTXObjectSpace.cu
  umesh/AWT.h
  umesh/AWT.cu
  umesh/UMeshMCAccelerator.h
  umesh/UMeshMCAccelerator.cu
  
  # implements the c99 barney api calls
  barney.cpp
  )
target_link_libraries(barney
  PUBLIC
  owl::owl
  cuBQL_impl
  barney-config
  barney-spheres-ptx
  barney-cylinders-ptx
  barney-triangles-ptx
  barney-traceRays-ptx  
  barney-UMeshRTXObjectSpace-ptx  
  barney-AWT-ptx  
  barney-UMeshMCAccelerator-ptx  
  )
#target_include_directories(barney PUBLIC
#  )

set_target_properties(barney
  PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_USE_STATIC_CUDA_RUNTIME ON
  CUDA_ARCHITECTURES OFF
)

# ------------------------------------------------------------------
find_package(MPI REQUIRED)

add_library(barney_mpi STATIC
  common/MPIWrappers.h
  common/MPIWrappers.cpp
  MPIContext.h
  MPIContext.cpp
  fb/DistFB.h
  fb/DistFB.cpp
  )
set_target_properties(barney_mpi
  PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_USE_STATIC_CUDA_RUNTIME ON
  CUDA_ARCHITECTURES OFF
)
target_link_libraries(barney_mpi PUBLIC
  barney
  MPI::MPI_CXX
  )
target_compile_definitions(barney_mpi PUBLIC -DBARNEY_MPI=1)


