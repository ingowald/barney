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

set(AWT_THRESHOLD "4" CACHE STRING "AWT switch-to-sampling threshold (if enabled)")

add_library(barney-config INTERFACE)
target_link_libraries(barney-config
  INTERFACE
  cuBQL_interface
  )
target_compile_definitions(barney-config INTERFACE -DAWT_THRESHOLD=${AWT_THRESHOLD})
  
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
# volume variants:
# ------------------------------------------------------------------
embed_ptx(
  OUTPUT_TARGET      barney-StructuredData-ptx
  PTX_LINK_LIBRARIES owl::owl barney-config
  SOURCES            volume/StructuredData.dev.cu
)
embed_ptx(
  OUTPUT_TARGET      barney-UMeshRTXObjectSpace-ptx
  PTX_LINK_LIBRARIES owl::owl barney-config
  SOURCES            umesh/os/RTXObjectSpace.dev.cu
)
embed_ptx(
  OUTPUT_TARGET      barney-UMeshMC-ptx
  PTX_LINK_LIBRARIES owl::owl barney-config
  SOURCES            umesh/mc/UMeshMC.dev.cu
)
embed_ptx(
  OUTPUT_TARGET      barney-AWT-ptx
  PTX_LINK_LIBRARIES owl::owl barney-config
  SOURCES            umesh/os/AWT.dev.cu
)

# ------------------------------------------------------------------
# amr variants:
# ------------------------------------------------------------------
embed_ptx(
  OUTPUT_TARGET      barney-BlockStructuredMC-ptx
  PTX_LINK_LIBRARIES owl::owl barney-config
  SOURCES            amr/BlockStructuredMC.dev.cu
)

add_library(barney STATIC
  common/barney-common.h
  common/cuda-helper.cpp
  
  # actual device kernels for CUDA ray gen and shading
  kernels/generateRays.cu
  kernels/shadeRays.cu
  # all the barney "plumbing" stuff on the host
  DeviceContext.h
  DeviceContext.cpp
  Context.h
  Context.cpp
  MultiPass.h
  MultiPass.cpp
  DeviceGroup.h
  DeviceGroup.cpp
  LocalContext.h
  LocalContext.cpp
  Object.h
  Object.cpp
  Camera.h
  Camera.cu
  # all frame buffer related stuff (DistFB in barney_mpi)
  fb/FrameBuffer.h
  fb/FrameBuffer.cpp
  fb/LocalFB.h
  fb/LocalFB.cpp
  fb/TiledFB.h
  fb/TiledFB.cu
  # model/group/data group handling
  GlobalModel.h
  GlobalModel.cpp
  Group.h
  Group.cpp
  material/Globals.h
  material/Globals.cu
  ModelSlot.h
  ModelSlot.cpp
  # general common/data-related stuff
  common/Texture.h
  common/Texture.cpp
  common/Data.h
  common/Data.cpp
  # lights
  light/Light.h
  light/Light.cpp
  # device-level materials
  material/device/Material.h
  material/device/Velvet.h
  material/device/Matte.h
  material/device/Metal.h
  material/device/Plastic.h
  # barney-level material types
  material/host/Material.h
  material/host/Material.cpp
  material/host/Velvet.h
  material/host/Velvet.cpp
  material/host/Plastic.h
  material/host/Plastic.cpp
  material/host/Matte.h
  material/host/Matte.cpp
  material/host/Metal.h
  material/host/Metal.cpp
  material/host/MetallicPaint.h
  material/host/MetallicPaint.cpp
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
  volume/ScalarField.h
  volume/ScalarField.cpp
  volume/MCAccelerator.h
  volume/TransferFunction.h
  volume/TransferFunction.cpp
  volume/MCGrid.h
  volume/MCGrid.cu
  volume/MCAccelerator.h
  # *structured* volumes
  volume/StructuredData.h
  volume/StructuredData.cu
  # *unstructured* volumes
  umesh/common/UMeshField.h
  umesh/common/UMeshField.cu
  # ... macro-cell based methods
  umesh/mc/UMeshCUBQLSampler.h
  umesh/mc/UMeshCUBQLSampler.cu
  # ... object-space methods
  umesh/os/RTXObjectSpace.h
  umesh/os/RTXObjectSpace.cu
  umesh/os/AWT.h
  umesh/os/AWT.cu
  # *amr* volumes
  amr/BlockStructuredField.h
  amr/BlockStructuredField.cu
  amr/BlockStructuredCUBQLSampler.h
  amr/BlockStructuredCUBQLSampler.cu
#  amr/BlockStructuredMCAccelerator.h
#  amr/BlockStructuredMCAccelerator.cu
  
  # implements the c99 barney api calls
  barney.cpp
  )
target_link_libraries(barney
  PUBLIC
  owl::owl
  cuBQL_impl
  barney-config
  barney-StructuredData-ptx
  barney-spheres-ptx
  barney-cylinders-ptx
  barney-triangles-ptx
  barney-traceRays-ptx  
  barney-UMeshRTXObjectSpace-ptx
  barney-UMeshMC-ptx
  barney-AWT-ptx  
  barney-BlockStructuredMC-ptx
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

if (BARNEY_MPI)
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
endif()


add_subdirectory(testing)

