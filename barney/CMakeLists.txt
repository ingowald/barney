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
  SOURCES            traceRays.cu
  )
# ------------------------------------------------------------------
# surface geometry:
# ------------------------------------------------------------------
# ray-sphere geom intersectoin
embed_ptx(
  OUTPUT_TARGET      barney-spheres-ptx
  PTX_LINK_LIBRARIES owl::owl barney-config
  SOURCES            Spheres.dev.cu
  )
# ray triangles geom intersection
embed_ptx(
  OUTPUT_TARGET      barney-triangles-ptx
  PTX_LINK_LIBRARIES owl::owl barney-config
  SOURCES            Triangles.dev.cu
  )

# ------------------------------------------------------------------
# unstucture mesh variants:
# ------------------------------------------------------------------
embed_ptx(
  OUTPUT_TARGET      barney-UMeshRTXObjectSpace-ptx
  PTX_LINK_LIBRARIES owl::owl barney-config
  SOURCES            unstructured/UMeshRTXObjectSpace.dev.cu
)
embed_ptx(
  OUTPUT_TARGET      barney-UMeshMCAccelerator-ptx
  PTX_LINK_LIBRARIES owl::owl barney-config
  SOURCES            unstructured/UMeshMCAccelerator.dev.cu
)

add_library(barney STATIC

  # actual device kernels for CUDA ray gen and shading
  generateRays.cu
  shadeRays.cu
  # all the barney "plumbing" stuff on the host
  DeviceContext.h
  DeviceContext.cu
  Context.h
  Context.cu
  DeviceGroup.h
  DeviceGroup.cpp
  LocalContext.h
  LocalContext.cu
  # all frame buffer related stuff (DistFB in barney_mpi)
  FrameBuffer.h
  FrameBuffer.cu
  LocalFB.h
  LocalFB.cu
  TiledFB.h
  TiledFB.cu
  # model/group/data group handling
  Model.h
  Model.cu
  Group.h
  Group.cpp
  DataGroup.h
  DataGroup.cpp
  # actual geometry types
  Geometry.h
  Geometry.cpp
  Spheres.h
  Spheres.cu
  Triangles.h
  Triangles.cu
  # actual volume types
  Volume.h
  Volume.cpp
  unstructured/MCAccelerator.h
  unstructured/MCAccelerator.cu
  unstructured/UMeshField.h
  unstructured/UMeshField.cu
  unstructured/TransferFunction.h
  unstructured/TransferFunction.cu
  unstructured/MCGrid.h
  unstructured/MCGrid.cu
  unstructured/MCAccelerator.h
  unstructured/UMeshMCAccelerator.cu
  unstructured/UMeshCUBQLSampler.cu
  unstructured/UMeshRTXObjectSpace.h
  unstructured/UMeshRTXObjectSpace.cu
  
#  unstructured/QuickClusters.cu
#  unstructured/SpatialAccelWithRTXSampling.h
#  unstructured/SpatialAccelWithRTXSampling.cu
#  unstructured/qbvh.h
#  unstructured/qbvh.cpp
  # implements the c99 barney api calls
  barney.cpp
  )
target_link_libraries(barney
  PUBLIC
  owl::owl
  cuBQL_impl
  barney-config
  barney-spheres-ptx
  barney-triangles-ptx
  barney-traceRays-ptx  
  barney-UMeshRTXObjectSpace-ptx  
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
  mpiAPI.cpp
  MPIWrappers.h
  MPIWrappers.cpp
  MPIContext.h
  MPIContext.cpp
  DistFB.h
  DistFB.cu
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


