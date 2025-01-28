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

if (BARNEY_DISABLE_DENOISING)
  target_compile_definitions(barney_config INTERFACE -DBARNEY_DISABLE_DENOISING=1)
endif()
    
# -------------------------------------------------------

if (BARNEY_BACKEND_OPTIX)
  add_library(barney_config_ptx INTERFACE)
  target_link_libraries(barney_config_ptx INTERFACE barney_config)
  target_compile_definitions(barney_config_ptx INTERFACE -DBARNEY_COMPILE_OPTIX_PROGRAMS=1)

  # ray gen program that launches kernel to trace rays
  embed_ptx(
    OUTPUT_TARGET      barney-traceRays-ptx
    PTX_LINK_LIBRARIES barney_config_ptx
    SOURCES            kernels/traceRays.cu
    )
  # ------------------------------------------------------------------
  # surface geometry:
  # ------------------------------------------------------------------
  # ray-sphere geom intersection
  embed_ptx(
    OUTPUT_TARGET      barney-spheres-ptx
    PTX_LINK_LIBRARIES barney_config_ptx
    SOURCES            geometry/Spheres.dev.cu
    )
  # ray triangles geom intersection
  embed_ptx(
    OUTPUT_TARGET      barney-triangles-ptx
    PTX_LINK_LIBRARIES barney_config_ptx
    SOURCES            geometry/Triangles.dev.cu
    )
  # ray cylinders geom intersection
  embed_ptx(
    OUTPUT_TARGET      barney-cylinders-ptx
    PTX_LINK_LIBRARIES barney_config_ptx
    SOURCES            geometry/Cylinders.dev.cu
    )
  # ray capsules geom intersection
  embed_ptx(
    OUTPUT_TARGET      barney-capsules-ptx
    PTX_LINK_LIBRARIES barney_config_ptx
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
    PTX_LINK_LIBRARIES barney_config_ptx
    SOURCES            volume/StructuredData.dev.cu
    EMBEDDED_SYMBOL_NAMES MCAccel_Structured_ptx
    )
  embed_ptx(
    OUTPUT_TARGET      barney-UMeshRTXObjectSpace-ptx
    PTX_LINK_LIBRARIES barney_config_ptx
    SOURCES            umesh/os/RTXObjectSpace.dev.cu
    )
  embed_ptx(
    OUTPUT_TARGET      barney-UMeshMC-ptx
    PTX_LINK_LIBRARIES barney_config_ptx
    SOURCES            umesh/mc/UMeshMC.dev.cu
    )
  embed_ptx(
    OUTPUT_TARGET      barney-AWT-ptx
    PTX_LINK_LIBRARIES barney_config_ptx
    SOURCES            umesh/os/AWT.dev.cu
    )

  # ------------------------------------------------------------------
  # amr variants:
  # ------------------------------------------------------------------
  embed_ptx(
    OUTPUT_TARGET      barney-BlockStructuredMC-ptx
    PTX_LINK_LIBRARIES barney_config_ptx
    SOURCES            amr/BlockStructuredMC.dev.cu
    )
endif()

set(BARNEY_SOURCES
  geometry/Triangles.dev.cu
  geometry/Capsules.dev.cu
  geometry/Spheres.dev.cu
  
  # unstructured meshes
  umesh/common/UMeshField.h
  umesh/common/UMeshField.cu
  umesh/mc/UMeshCUBQLSampler.h
  umesh/mc/UMeshCUBQLSampler.cu
  umesh/mc/UMeshMC.dev.cu
  
  # *structured* volumes
  volume/StructuredData.h
  volume/StructuredData.cu
  volume/StructuredData.dev.cu
  
  # actual geometry types
  geometry/Geometry.h
  geometry/Geometry.cpp
  geometry/Triangles.h
  geometry/Triangles.cpp
  geometry/Cylinders.h
  geometry/Cylinders.cpp
  geometry/Capsules.h
  geometry/Capsules.cpp
  geometry/Spheres.h
  geometry/Spheres.cpp
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

  # samplers
  render/Sampler.h
  render/Sampler.cpp
  
  common/barney-common.h
  common/cuda-helper.cpp
  
  # actual device kernels for CUDA ray gen and shading
  kernels/generateRays.cu
  kernels/shadeRays.cu
  kernels/traceRays.cu
  # all the barney "plumbing" stuff on the host
  DeviceContext.h
  DeviceContext.cpp
  Context.h
  Context.cpp
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
  render/GeometryAttributes.h
  render/GeometryAttributes.cpp

  render/RayQueue.h
  render/RayQueue.cpp
  render/World.h
  render/World.cpp
  render/MaterialRegistry.h
  render/MaterialRegistry.cpp
  render/SamplerRegistry.h
  render/SamplerRegistry.cpp
  
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

  # materials
  material/Material.h
  material/Material.cpp
  material/AnariPBR.h
  material/AnariPBR.cpp
  material/AnariMatte.h
  material/AnariMatte.cpp

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


# ==================================================================
if (BARNEY_HAVE_CUDA)
  macro(CUDA_SOURCE fileName)
    # nothing to do, leave it to cuda to compile this.
  endmacro()
else()
  # we don't actually _have_ cuda - tell a regular C++ compiler to
  # compile this source, and hope it has nothing in it that wouldn't
  # work on off-the-shelf c++
  macro(CUDA_SOURCE fileName)
    if (WIN32)
      set_source_files_properties(${fileName}
        PROPERTIES
        LANGUAGE CXX
        )
      
    else()
      set_source_files_properties(${fileName}
        PROPERTIES
        LANGUAGE CXX
        COMPILE_OPTIONS   "-xc++"
        )
    endif()
  endmacro()
endif()
# ==================================================================


CUDA_SOURCE(kernels/traceRays.cu)
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
# barney itself
# ------------------------------------------------------------------
add_library(barney SHARED
  ${BARNEY_SOURCES}
  )
target_link_libraries(barney PUBLIC
  # base config and rtcore - these go into barney, one way or another
  $<BUILD_INTERFACE:barney_config>
  $<BUILD_INTERFACE:barney_rtc_common>
  $<BUILD_INTERFACE:barney_rtc_optix>
  $<BUILD_INTERFACE:barney_rtc_cuda_common>
  $<BUILD_INTERFACE:barney_rtc_embree>
  )

if (BARNEY_BACKEND_OPTIX)
  # optix embedded ptx code strings for the precompiled kernels. these
  # only go into barney if optix mode is enabled (because otherwise we
  # cannot even compile them)
  target_link_libraries(barney PUBLIC
    $<BUILD_INTERFACE:barney-triangles-ptx>
    $<BUILD_INTERFACE:barney-capsules-ptx>
    $<BUILD_INTERFACE:barney-spheres-ptx>
    #
    $<BUILD_INTERFACE:barney-StructuredData-ptx>
    #
    $<BUILD_INTERFACE:barney-traceRays-ptx>
    )
endif()
set_target_properties(barney
  PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_USE_STATIC_CUDA_RUNTIME ON
  CUDA_ARCHITECTURES "${CMAKE_CUDA_ARCHITECTURES}"
  )

if (BARNEY_MPI)
  
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
    barney
    $<BUILD_INTERFACE:MPI::MPI_CXX>
    )
  target_compile_definitions(barney_mpi PUBLIC -DBARNEY_MPI=1)
  set_target_properties(barney_mpi PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()



