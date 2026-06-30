# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
# CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# "backend" variable gets set by parent dir, and is one of either
# 'cpu', 'cuda', 'optix', 'hip' or 'hiprt'

message("building barney w/ backend '${backend}'")
include(../rtcore/${backend}.cmake)

if (NOT (TARGET barney_rtc_${backend}))
  # we have no rtcore config for this backend - either because it was
  # disabled by user, or because it lacks something to build. cannot
  # bild this backend, return.
  message("#barney: no valid rtcore config found for backend '${backend}'")
  return()
endif()

set(BARNEY_DEVICE_NAME ${BARNEY_PROJECT_NAME}_${backend})

add_library(barney_config_${backend} INTERFACE)
target_link_libraries(barney_config_${backend} INTERFACE barney_rtc_${backend})

if(MSVC) # That's also clang-cl
  # /bigobj is needed for bigger binding projects due to the limit to 64k
  # addressable sections
  set_property(
    TARGET barney_config_${backend}
    APPEND
    PROPERTY INTERFACE_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
endif()

target_include_directories(barney_config_${backend} INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${CMAKE_CURRENT_BINARY_DIR}/include/>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/3rdParty/nanovdb/include/>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/3rdParty/glm/include/>
)
if (WIN32)
  target_compile_definitions(barney_config_${backend} INTERFACE -DNOMINMAX)
  target_compile_options(barney_config_${backend} INTERFACE
    $<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler="/Zc:preprocessor">
  )
endif()


#############################################################################
# main step - create the actual 'backends'; ie, the 'instantiations' of
# barney's actuall classes, algorithms, kernels, programs, etc, for a
# given rtcore.
#############################################################################

# define all the host-side sources
set(HOST_SOURCES

  globalTrace/RQSBase.h
  globalTrace/RQSBase.cpp
  globalTrace/RQSLocal.h
  globalTrace/RQSLocal.cu
  
  # all the barney "plumbing" stuff on the host
  WorkerTopo.h
  WorkerTopo.cpp
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

  # samplers
  render/Sampler.h
  render/Sampler.cpp
  
  # actual device kernels for CUDA ray gen and shading
  kernels/generateRays.cu
  kernels/shadeRays.cu
  kernels/traceRays.cu
  
  umesh/common/UMeshField.h
  umesh/common/UMeshField.cu
  umesh/mc/UMeshCuBQLSampler.h
  umesh/mc/UMeshCuBQLSampler.cu
 
  amr/BlockStructuredCuBQLSampler.cu
  amr/BlockStructuredField.cu
  
  # *structured* volumes
  volume/StructuredData.h
  volume/StructuredData.cu
  # *NanoVB* volumes
  volume/NanoVDB.h
  volume/NanoVDB.cu
  
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
  geometry/Cones.h
  geometry/Cones.cpp
  geometry/IsoSurface.h
  geometry/IsoSurface.cpp
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
  
  common/barney-common.h
  
  render/RayQueue.h
  render/RayQueue.cpp
  render/World.h
  render/World.cpp
  render/MaterialRegistry.h
  render/MaterialRegistry.cpp
  render/SamplerRegistry.h
  render/SamplerRegistry.cpp
  
  ModelSlot.h
  ModelSlot.cpp
  
  # general csommon/data-related stuff
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
)

# all the source that eventually go into barney_mpi
set(MPI_SOURCES
  fb/DistFB.h
  fb/DistFB.cu
  MPIContext.h
  MPIContext.cpp
  globalTrace/RQSMPI.h
  globalTrace/RQSMPI.cu
  globalTrace/TwoStage.h
  globalTrace/TwoStage.cu
  globalTrace/All2all.h
  globalTrace/All2all.cu
  common/MPIWrappers.h
  common/MPIWrappers.cpp
)

set(DEVICE_PROGRAM_SOURCES
  geometry/Spheres.dev.cu
  geometry/Triangles.dev.cu
  geometry/Cylinders.dev.cu
  geometry/Capsules.dev.cu
  geometry/Cones.dev.cu
  volume/StructuredData.dev.cu
  volume/NanoVDB.dev.cu
  umesh/mc/UMeshMC.dev.cu
  amr/BlockStructuredMC.dev.cu
  kernels/traceRays.dev.cu
  api/barney.cu
)

# this will build a static library 'barney_device_programs' that
# either contains the compiled device programs, or library containing
# the precomiled ptx sources
rtc_build_device_sources(barney_${backend}_programs
  ${DEVICE_PROGRAM_SOURCES}
)

foreach(src ${HOST_SOURCES} ${DEVICE_PROGRAM_SOURCES})
  rtc_configure_source(${src})
endforeach()
    
# ------------------------------------------------------------------
# barney_static
# ------------------------------------------------------------------
add_library(${BARNEY_DEVICE_NAME} STATIC
  ${HOST_SOURCES}
)
target_link_libraries(${BARNEY_DEVICE_NAME} PUBLIC
  $<BUILD_INTERFACE:barney_${backend}_programs>
  $<BUILD_INTERFACE:barney_${backend}_config>
)
rtc_library_properties(${BARNEY_DEVICE_NAME})

# =============================================================================
# create frontends for barney_mpi and barney_mpi_static
# =============================================================================
if (BARNEY_MPI)
  # ------------------------------------------------------------------
  # barney_mpi
  # ------------------------------------------------------------------
  add_library(${BARNEY_DEVICE_NAME}_mpi STATIC
    ${HOST_SOURCES} ${MPI_SOURCES}
  )
  target_link_libraries(${BARNEY_DEVICE_NAME}_mpi PUBLIC
    $<BUILD_INTERFACE:barney_${backend}_programs>
    $<BUILD_INTERFACE:barney_${backend}_config>
  )
  rtc_library_properties(${BARNEY_DEVICE_NAME}_mpi) 
  target_compile_definitions(${BARNEY_DEVICE_NAME}_mpi PRIVATE
    -Dbarney_mpi_STATIC=1
    -DBARNEY_MPI=1
  )
endif()


