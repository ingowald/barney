# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

if(MSVC) # That's also clang-cl
  # /bigobj is needed for bigger binding projects due to the limit to 64k
  # addressable sections
  set_property(
    TARGET barney_config
    APPEND
    PROPERTY INTERFACE_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
endif()

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr")

add_library(barney_config_ptx INTERFACE)
target_link_libraries(barney_config_ptx
  INTERFACE
  barney_config
)
target_compile_definitions(barney_config_ptx
  INTERFACE
  -DBARNEY_DEVICE_PROGRAM=1
)

macro(set_library_properties tgt)
  set_target_properties(${tgt}
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION   ON
    POSITION_INDEPENDENT_CODE    ON
    CUDA_USE_STATIC_CUDA_RUNTIME ON
    CUDA_RESOLVE_DEVICE_SYMBOLS  ON
    CXX_VISIBILITY_PRESET  hidden
    CUDA_VISIBILITY_PRESET hidden
  )
if (APPLE)
  set_target_properties(${tgt} PROPERTIES INSTALL_RPATH "$loader_path")
else()
  set_target_properties(${tgt} PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()
endmacro()


# in the barney backend all sources will only use host-side code, but
# we can still compile them with nvcc if it's avaible - so _if_ nvcc
# is available we leave the language for .cu files on 'cuda' in order
# to let the same sources be properly compiled for non-embree
# backends; but if there's no cuda vailable we have to change the
# language to CPP else cmake will just skip these sources.
macro(configure_cuda_source src)
  if (NOT BARNEY_HAVE_CUDA)
    get_filename_component(ext "${src}" EXT)
    if (("${ext}" STREQUAL ".cu") OR ("${ext}" STREQUAL ".dev.cu"))
      message("setting language of ${src} to CXX")
      set_source_files_properties(${src} PROPERTIES LANGUAGE CXX)
      if (NOT WIN32)
        set_source_files_properties(${src} PROPERTIES COMPILE_OPTIONS "-xc++")
      endif()
    endif()
  else()
    get_filename_component(ext "${src}" EXT)
    if (("${ext}" STREQUAL ".cu") OR ("${ext}" STREQUAL ".dev.cu"))
      set_source_files_properties(${src}
        # -rdc=true is for relocatable code, can be resolved at link time.
        PROPERTIES COMPILE_OPTIONS "--extended-lambda;-rdc=true"
      )
    endif()
#    set_property(
#      TARGET barney_config
#      APPEND
#      PROPERTY INTERFACE_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CXX>:/bigobj>)
  endif()
endmacro()








#############################################################################
# main step - crete the actual 'backends'; ie, the 'instantiations' of
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
 
#  umesh/os/AWT.cu
#  umesh/os/AWT.h

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
  
  #  material/Globals.h
  #  material/Globals.cu
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
#  umesh/os/AWT.dev.cu
  amr/BlockStructuredMC.dev.cu
  kernels/traceRays.dev.cu
)

# ------------------------------------------------------------------
# optix backend: instantiates all the barney classes,Data device
# programs, etcpp - except the api itself - in a optix
# configuration.
# ------------------------------------------------------------------
if (BARNEY_BACKEND_OPTIX)
  add_library(barney_optix ${HOST_SOURCES})
  foreach(src  ${DEVICE_PROGRAM_SOURCES})
    get_filename_component(basename "${src}" NAME_WE)
    embed_ptx(
      OUTPUT_TARGET      barney-${basename}-ptx
      PTX_LINK_LIBRARIES barney_config_ptx barney_rtc_optix
      SOURCES            ${src}
      )
    target_link_libraries(barney_optix PRIVATE barney-${basename}-ptx)
  endforeach()
  set_library_properties(barney_optix)
  target_link_libraries(barney_optix PUBLIC
    barney_config
    barney_rtc_optix
    $<BUILD_INTERFACE:cuBQL_cuda_float3_static>
  )
  set_property(
    TARGET barney_optix 
    APPEND
    PROPERTY INTERFACE_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>:--expt-relaxed-constexpr>)
  set_property(
    TARGET barney_optix 
    APPEND
    PROPERTY INTERFACE_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>:--extended-lambda>)
  if (BARNEY_MPI)
    add_library(barney_mpi_optix ${MPI_SOURCES})
    target_link_libraries(barney_mpi_optix PUBLIC barney_optix MPI::MPI_C)
    set_library_properties(barney_mpi_optix)
  endif()
endif()


foreach(src ${HOST_SOURCES} ${DEVICE_PROGRAM_SOURCES})
  configure_cuda_source(${src})
endforeach()

# ------------------------------------------------------------------
# embree backend: instantiates all the barney classes, device
# programs, etcpp - except the api itself - in an embree configuration
# that'll run on the host.
# ------------------------------------------------------------------
if (BARNEY_BACKEND_EMBREE)
  # in the barney backend all sources will only use host-side code,
  # but we can still compile them with nvcc if it's avaible - so
  # _if_ nvcc is available we leave the language for .cu files on
  # 'cuda' in order to let the same sources be properly compiled for
  # non-embree backends; but if there's no cuda vailable we have to
  # change the language to CPP else cmake will just skip these
  # sources.

  # ----------- barney_embree -----------
  add_library(barney_embree_programs STATIC ${DEVICE_PROGRAM_SOURCES})  
  target_link_libraries(barney_embree_programs PRIVATE barney_config barney_rtc_embree)
  target_compile_definitions(barney_embree_programs PRIVATE -DBARNEY_DEVICE_PROGRAM=1 -DBARNEY_RTC_EMBREE=1)
  set_library_properties(barney_embree_programs)

  add_library(barney_embree STATIC ${HOST_SOURCES})
  target_link_libraries(barney_embree PUBLIC barney_config)
  target_link_libraries(barney_embree PRIVATE barney_rtc_embree)
  target_link_libraries(barney_embree PRIVATE
    barney_embree_programs
    barney_config
    $<BUILD_INTERFACE:cuBQL_cpu_float3_static>)
  target_compile_definitions(barney_embree_programs PRIVATE -DBARNEY_RTC_EMBREE=1)
  set_library_properties(barney_embree)

  if (BARNEY_MPI)
    add_library(barney_mpi_embree ${MPI_SOURCES})
    target_compile_definitions(barney_mpi_embree PUBLIC -DBARNEY_RTC_EMBREE=1)
    target_link_libraries(barney_mpi_embree PUBLIC barney_config embree_local)
    target_link_libraries(barney_mpi_embree PUBLIC barney_embree MPI::MPI_C)
    set_library_properties(barney_mpi_embree)
  endif()
endif()
    


# ------------------------------------------------------------------
# cuda backend: instantiates all the barney classes, device
# programs, etcpp - except the api itself - in an cuda configuration
# that'll run on the gpu, but without requiring optix
# ------------------------------------------------------------------
if (BARNEY_BACKEND_CUDA)
  # ----------- barney_cuda -----------
  add_library(barney_cuda_programs STATIC ${DEVICE_PROGRAM_SOURCES})  
  target_link_libraries(barney_cuda_programs PUBLIC barney_config barney_rtc_cuda)
  target_compile_definitions(barney_cuda_programs PRIVATE -DBARNEY_DEVICE_PROGRAM=1)
  set_library_properties(barney_cuda_programs)

  add_library(barney_cuda STATIC ${HOST_SOURCES})
  target_link_libraries(barney_cuda PUBLIC barney_config barney_rtc_cuda)
  if (TARGET cuBQL_cuda_float3_static)
    target_link_libraries(barney_cuda PRIVATE
      $<BUILD_INTERFACE:cuBQL_cuda_float3_static>
      )
  endif()
  target_link_libraries(barney_cuda PRIVATE
    barney_cuda_programs
    barney_config
    )
  set_library_properties(barney_cuda)

  if (BARNEY_MPI)
    add_library(barney_mpi_cuda ${MPI_SOURCES})
    target_link_libraries(barney_mpi_cuda PUBLIC barney_cuda MPI::MPI_C)
    set_library_properties(barney_mpi_cuda)
  endif()
endif()
    




# =============================================================================
# create frontends for barney and barney_static
# =============================================================================
configure_cuda_source(api/barney.cu)

# ------------------------------------------------------------------
# barney
# ------------------------------------------------------------------
add_library(barney SHARED api/barney.cu)
target_link_libraries(barney PUBLIC
  $<BUILD_INTERFACE:barney_config>
)
set_library_properties(barney)
target_include_directories(barney PUBLIC
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

# ------------------------------------------------------------------
# barney_static
# ------------------------------------------------------------------
add_library(barney_static STATIC api/barney.cu)
target_include_directories(barney_static PUBLIC
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_compile_definitions(barney_static PUBLIC -Dbarney_STATIC=1)
target_link_libraries(barney_static PUBLIC $<BUILD_INTERFACE:barney_config>)
set_library_properties(barney_static)

# =============================================================================
# create frontends for barney_mpi and barney_mpi_static
# =============================================================================
if (BARNEY_MPI)
  # ------------------------------------------------------------------
  # barney_mpi
  # ------------------------------------------------------------------
  add_library(barney_mpi SHARED
    api/barney.cu
  )
  target_link_libraries(barney_mpi PUBLIC
    #    barney_config
    $<BUILD_INTERFACE:barney_config>
  )
  target_include_directories(barney_mpi PUBLIC
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )
  set_library_properties(barney_mpi) 
  target_compile_definitions(barney_mpi PRIVATE -DBARNEY_MPI=1)
 
  # ------------------------------------------------------------------
  # barney_mpi_static
  # ------------------------------------------------------------------
  add_library(barney_mpi_static STATIC api/barney.cu)
  target_compile_definitions(barney_mpi_static PUBLIC
    -Dbarney_mpi_STATIC=1 -DBARNEY_MPI=1)
  target_link_libraries(barney_mpi_static PUBLIC
    #   barney_config
    $<BUILD_INTERFACE:barney_config>
  )
  target_include_directories(barney_mpi_static PUBLIC
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  )
  set_library_properties(barney_mpi_static)
endif()


#############################################################################
# last step - link all optix/embree backend(s) to all
# mpi/nompi/static/shared frontends
#############################################################################

# link optix backend (if enabled) to all frontends
if (BARNEY_BACKEND_OPTIX)
  target_link_libraries(barney PRIVATE $<BUILD_INTERFACE:barney_optix>)
  target_link_libraries(barney_static PRIVATE $<BUILD_INTERFACE:barney_optix>)
  if (BARNEY_MPI)
    target_link_libraries(barney_mpi PUBLIC $<BUILD_INTERFACE:barney_mpi_optix>)
    target_link_libraries(barney_mpi_static PUBLIC $<BUILD_INTERFACE:barney_mpi_optix>)
  endif()
endif()
# link embree backend (if enabled) to all frontends
if (BARNEY_BACKEND_EMBREE)
  target_link_libraries(barney PRIVATE $<BUILD_INTERFACE:barney_embree>)
  target_link_libraries(barney_static PRIVATE $<BUILD_INTERFACE:barney_embree>)
  if (BARNEY_MPI)
    target_link_libraries(barney_mpi PUBLIC
      $<BUILD_INTERFACE:barney_mpi_embree>)
    target_link_libraries(barney_mpi_static PUBLIC
      $<BUILD_INTERFACE:barney_mpi_embree>)
  endif()
endif()

# link cuda backend (if enabled) to all frontends
if (BARNEY_BACKEND_CUDA)
  target_link_libraries(barney PRIVATE $<BUILD_INTERFACE:barney_cuda>)
  target_link_libraries(barney_static PRIVATE $<BUILD_INTERFACE:barney_cuda>)
  if (BARNEY_MPI)
    target_link_libraries(barney_mpi PUBLIC $<BUILD_INTERFACE:barney_mpi_cuda>)
    target_link_libraries(barney_mpi_static PUBLIC $<BUILD_INTERFACE:barney_mpi_cuda>)
  endif()
endif()



# ##################################################################
# final lib properties
# ##################################################################
set_library_properties(barney)
if (BARNEY_MPI)
  set_library_properties(barney_mpi)
endif()
