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


# -------------------------------------------------------
set(AWT_THRESHOLD "4" CACHE STRING "AWT switch-to-sampling threshold (if enabled)")
target_compile_definitions(barney_config INTERFACE -DAWT_THRESHOLD=${AWT_THRESHOLD})

# -------------------------------------------------------
option(BARNEY_DISABLE_DENOISING "explicitly DISable denoiser?" OFF)
#option(BARNEY_DISABLE_OIDN "explicitly DISable OIDN?" OFF)

if (BARNEY_DISABLE_DENOISING)
  target_compile_definitions(barney_config INTERFACE -DBARNEY_DISABLE_DENOISING=1)
#elseif (NOT BARNEY_DISABLE_OIDN)
#  find_package(OpenImageDenoise)
#  if (OpenImageDenoise_FOUND)
#    message(INFO "Found OpenImageDenoise (OIDN), enabling it")
#    target_compile_definitions(barney_config INTERFACE -DBARNEY_HAVE_OIDN=1)
#    target_link_libraries(barney_config INTERFACE OpenImageDenoise)
#  endif()
endif()
# issue 5: if oidn is built without CUDA support it will break the
# barney build if we use it; so unless we figure out how to check that
# let's disable it.
target_compile_definitions(barney_config INTERFACE -DBARNEY_DISABLE_OIDN=1)
    
# -------------------------------------------------------

# ray gen program that launches kernel to trace rays
embed_ptx(
  OUTPUT_TARGET      barney-traceRays-ptx
  PTX_LINK_LIBRARIES barney_config
  SOURCES            kernels/traceRays.cu
  )
# ------------------------------------------------------------------
# surface geometry:
# ------------------------------------------------------------------
# ray-sphere geom intersection
embed_ptx(
  OUTPUT_TARGET      barney-Spheres-ptx
  PTX_LINK_LIBRARIES barney_config
  SOURCES            geometry/Spheres.dev.cu
  )
# ray triangles geom intersection
embed_ptx(
  OUTPUT_TARGET      barney-triangles-ptx
  PTX_LINK_LIBRARIES barney_config
  SOURCES            geometry/Triangles.dev.cu
  )
# ray cylinders geom intersection
embed_ptx(
  OUTPUT_TARGET      barney-cylinders-ptx
  PTX_LINK_LIBRARIES barney_config
  SOURCES            geometry/Cylinders.dev.cu
  )
# ray capsules geom intersection
embed_ptx(
  OUTPUT_TARGET      barney-capsules-ptx
  PTX_LINK_LIBRARIES barney_config
  SOURCES            geometry/Capsules.dev.cu
  )
# ray cones geom intersection
#embed_ptx(
#  OUTPUT_TARGET      barney-cones-ptx
#  PTX_LINK_LIBRARIES barney_config
#  SOURCES            geometry/Cones.dev.cu
#  )

# ------------------------------------------------------------------
# volume variants:
# ------------------------------------------------------------------
embed_ptx(
  OUTPUT_TARGET      barney-StructuredData-ptx
  PTX_LINK_LIBRARIES barney_config
  SOURCES            volume/StructuredData.dev.cu
)
embed_ptx(
  OUTPUT_TARGET      barney-UMeshRTXObjectSpace-ptx
  PTX_LINK_LIBRARIES barney_config
  SOURCES            umesh/os/RTXObjectSpace.dev.cu
)
embed_ptx(
  OUTPUT_TARGET      barney-UMeshMC-ptx
  PTX_LINK_LIBRARIES barney_config
  SOURCES            umesh/mc/UMeshMC.dev.cu
)
embed_ptx(
  OUTPUT_TARGET      barney-AWT-ptx
  PTX_LINK_LIBRARIES barney_config
  SOURCES            umesh/os/AWT.dev.cu
)

# ------------------------------------------------------------------
# amr variants:
# ------------------------------------------------------------------
embed_ptx(
  OUTPUT_TARGET      barney-BlockStructuredMC-ptx
  PTX_LINK_LIBRARIES barney_config
  SOURCES            amr/BlockStructuredMC.dev.cu
  )

set(BARNEY_SOURCES
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
  fb/FrameBuffer.cu
  fb/LocalFB.h
  fb/LocalFB.cpp
  fb/TiledFB.h
  fb/TiledFB.cu
  # model/group/data group handling
  GlobalModel.h
  GlobalModel.cpp
  render/Renderer.h
  render/Renderer.cpp
  Group.h
  Group.cpp

  render/RayQueue.h
  render/RayQueue.cpp
  render/World.h
  render/World.cpp
  
  #  material/Globals.h
  #  material/Globals.cu
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
  light/DirLight.h
  light/DirLight.cu
  light/PointLight.h
  light/PointLight.cu
  light/QuadLight.h
  light/QuadLight.cu
  light/EnvMap.h
  light/EnvMap.cu
  # samplers
  render/Sampler.h
  render/Sampler.cpp
  # materials
  material/Material.h
  material/Material.cpp
  material/AnariPBR.h
  material/AnariPBR.cpp
  material/AnariMatte.h
  material/AnariMatte.cpp
  #material/device/Material.h
  #material/device/Velvet.h
  #material/device/VisRTX.h
  #material/device/Matte.h
  #material/device/Blender.h
  #material/device/Glass.h
  #material/device/Metal.h
  #material/device/Plastic.h
  
  #material/host/VisRTX.h
  #material/host/VisRTX.cpp
  #material/host/Velvet.h
  #material/host/Velvet.cpp
  #material/host/Plastic.h
  #material/host/Plastic.cpp
  #material/host/Matte.h
  #material/host/Matte.cpp
  #material/host/Blender.h
  #material/host/Blender.cpp
  #material/host/Glass.h
  #material/host/Glass.cpp
  #material/host/Metal.h
  #material/host/Metal.cpp
  #material/host/MetallicPaint.h
  #material/host/MetallicPaint.cpp
  # actual geometry types
  geometry/Geometry.h
  geometry/Geometry.cpp
  geometry/Spheres.h
  geometry/Spheres.cpp
  geometry/Triangles.h
  geometry/Triangles.cpp
  geometry/Cylinders.h
  geometry/Cylinders.cpp
  geometry/Capsules.h
  geometry/Capsules.cpp
  #  geometry/Cones.h
  #  geometry/Cones.cpp
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

  
#  umesh/common/UMeshField.h
#  umesh/common/UMeshField.cu
#  # ... macro-cell based methods
#  umesh/mc/UMeshCUBQLSampler.h
#  umesh/mc/UMeshCUBQLSampler.cu
#  # ... object-space methods
#  umesh/os/RTXObjectSpace.h
#  umesh/os/RTXObjectSpace.cu
#  umesh/os/AWT.h
#  umesh/os/AWT.cu
#  # *amr* volumes
#  amr/BlockStructuredField.h
#  amr/BlockStructuredField.cu
#  amr/BlockStructuredCUBQLSampler.h
#  amr/BlockStructuredCUBQLSampler.cu
  #  amr/BlockStructuredMCAccelerator.h
  #  amr/BlockStructuredMCAccelerator.cu
  
  # implements the c99 barney api calls
  barney.cpp
  )

CUDA_SOURCE(kernels/generateRays.cu)
CUDA_SOURCE(kernels/shadeRays.cu)
CUDA_SOURCE(Camera.cu)
CUDA_SOURCE(fb/FrameBuffer.cu)
CUDA_SOURCE(fb/TiledFB.cu)
CUDA_SOURCE(fb/LocalFB.cu)
CUDA_SOURCE(light/DirLight.cu)
CUDA_SOURCE(light/PointLight.cu)
CUDA_SOURCE(light/QuadLight.cu)
CUDA_SOURCE(light/EnvMap.cu)
CUDA_SOURCE(kernels/shadeRays.cu)
CUDA_SOURCE(volume/MCGrid.cu)
CUDA_SOURCE(volume/StructuredData.cu)
CUDA_SOURCE(umesh/common/UMeshField.cu)
CUDA_SOURCE(umesh/mc/UMeshCUBQLSampler.cu)
CUDA_SOURCE(umesh/os/RTXObjectSpace.cu)
CUDA_SOURCE(umesh/os/AWT.cu)
CUDA_SOURCE(amr/BlockStructuredField.cu)
CUDA_SOURCE(amr/BlockStructuredCUBQLSampler.cu)
  
# ------------------------------------------------------------------
# STATIC version of barney
# ------------------------------------------------------------------
add_library(barney_static
#  OBJECT
  STATIC
  ${BARNEY_SOURCES}
  )
set(BARNEY_STATIC_DEPENDENCIES
  barney-Spheres-ptx
  barney-StructuredData-ptx
  barney-cylinders-ptx
  barney-capsules-ptx
  barney-triangles-ptx
  barney-traceRays-ptx  
#  barney-UMeshRTXObjectSpace-ptx
#  barney-UMeshMC-ptx
#  barney-AWT-ptx  
#  barney-BlockStructuredMC-ptx
  owl_static
  ${CUBQL_LIB}_static
  )
target_link_libraries(barney_static
  PUBLIC
  barney_config
  barney_rtc
  ${BARNEY_STATIC_DEPENDENCIES}
  )
set_target_properties(barney_static
  PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_USE_STATIC_CUDA_RUNTIME ON
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
  )
# ------------------------------------------------------------------
# SHARED version of barney
# ------------------------------------------------------------------
add_library(barney SHARED
  ${BARNEY_SOURCES}
  )
target_link_libraries(barney PUBLIC
  barney_config
  barney_rtc
  ${BARNEY_STATIC_DEPENDENCIES}
  )
set_target_properties(barney
  PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_USE_STATIC_CUDA_RUNTIME ON
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
  )
set_target_properties(barney PROPERTIES INSTALL_RPATH "$ORIGIN")

if (BARNEY_MPI)
#  find_package(MPI REQUIRED)

  # ------------------------------------------------------------------
  # STATIC version of barney_mpi
  # ------------------------------------------------------------------
  add_library(barney_mpi_static STATIC
    ${BARNEY_SOURCES}
    common/MPIWrappers.h
    common/MPIWrappers.cpp
    MPIContext.h
    MPIContext.cpp
    fb/DistFB.h
    fb/DistFB.cpp
    )
  set_target_properties(barney_mpi_static
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
    CUDA_USE_STATIC_CUDA_RUNTIME ON
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
  target_link_libraries(barney_mpi_static PUBLIC
    barney_config
    barney_static
    barney_rtc
    MPI::MPI_CXX
    )
  target_compile_definitions(barney_mpi_static PUBLIC -DBARNEY_MPI=1)


  
  # ------------------------------------------------------------------
  # SHARED version of barney_mpi
  # ------------------------------------------------------------------
  add_library(barney_mpi SHARED
    ${BARNEY_SOURCES}
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
    CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
    )
  target_link_libraries(barney_mpi PUBLIC
    barney_config
    barney_rtc
    ${BARNEY_STATIC_DEPENDENCIES}
#    owl::owl_static
#    ${CUBQL_LIB}
    MPI::MPI_CXX
#    barney-Spheres-ptx
#    barney-StructuredData-ptx
#    barney-cylinders-ptx
#    barney-capsules-ptx
#    barney-triangles-ptx
#    barney-traceRays-ptx  
#    barney-UMeshRTXObjectSpace-ptx
#    barney-UMeshMC-ptx
#    barney-AWT-ptx  
#    barney-BlockStructuredMC-ptx
    )
  target_compile_definitions(barney_mpi PUBLIC -DBARNEY_MPI=1)
  set_target_properties(barney_mpi PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()



#install(TARGETS cuBQL_cuda_float3 owl barney barney_mpi
#  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
#  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
#)

