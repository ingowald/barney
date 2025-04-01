set(CMAKE_CXX_VISIBILITY_PRESET  hidden)
set(CMAKE_CUDA_VISIBILITY_PRESET hidden)

macro(rtc_library_properties tgt)
  set_target_properties(${tgt}
    PROPERTIES
    CUDA_SEPARABLE_COMPILATION   ON
    POSITION_INDEPENDENT_CODE    ON
    CUDA_USE_STATIC_CUDA_RUNTIME ON
    CUDA_RESOLVE_DEVICE_SYMBOLS  ON
    CXX_VISIBILITY_PRESET  hidden
    CUDA_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN ON
  )
if (APPLE)
  set_target_properties(${tgt} PROPERTIES INSTALL_RPATH "$loader_path")
else()
  set_target_properties(${tgt} PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()
endmacro()

# ==================================================================

target_include_directories(barney_config INTERFACE
  ${PROJECT_SOURCE_DIR}
  )
target_link_libraries(barney_config INTERFACE owl-config)


# ==================================================================
add_library(barney_rtc_config INTERFACE
  )
target_link_libraries(barney_rtc_config INTERFACE barney_config)
set_property(TARGET barney_rtc_config PROPERTY
  CXX_STANDARD 20)



# ==================================================================
add_library(barney_rtc_common INTERFACE
  )
target_link_libraries(barney_rtc_common INTERFACE barney_rtc_config)


# ==================================================================
if (BARNEY_BACKEND_EMBREE)
  message("enabling EMBREE backend")
  if (APPLE AND ${CMAKE_SYSTEM_PROCESSOR} STREQUAL "arm64")
    set(COMPILE_FOR_ARM ON)
  elseif (UNIX AND ${CMAKE_SYSTEM_PROCESSOR} STREQUAL "aarch64")
    set(COMPILE_FOR_ARM ON)
  else()
    set(COMPILE_FOR_ARM OFF)
  endif()

  option(EMBREE_GEOMETRY_CURVE          "" OFF)
  option(EMBREE_GEOMETRY_GRID           "" OFF)
  option(EMBREE_GEOMETRY_INSTANCE       "" ON)
  option(EMBREE_GEOMETRY_INSTANCE_ARRAY "" ON)
  option(EMBREE_GEOMETRY_POINT          "" ON)
  option(EMBREE_GEOMETRY_QUAD           "" ON)
  option(EMBREE_GEOMETRY_SUBDIVISION    "" OFF)
  option(EMBREE_GEOMETRY_TRIANGLE       "" ON)
  option(EMBREE_GEOMETRY_USER           "" ON)
  if (COMPILE_FOR_ARM)
    option(EMBREE_ISA_NEON           "" OFF)
    option(EMBREE_ISA_NEON2X         "" ON)
  else()
    option(EMBREE_ISA_AVX            "" OFF)
    option(EMBREE_ISA_AVX2           "" OFF)
    option(EMBREE_ISA_AVX512         "" OFF)
    option(EMBREE_ISA_SSE42          "" ON)
    option(EMBREE_ISA_SSE2           "" OFF)
  endif()
  option(EMBREE_ISPC_SUPPORT             "" OFF)
  option(EMBREE_STATIC_LIB               "" ON)
  option(EMBREE_STAT_COUNTERS            "" OFF)
  option(EMBREE_SYCL_SUPPORT             "" OFF)
  option(EMBREE_TUTORIALS                "" OFF)
  option(EMBREE_TUTORIALS_GLFW           "" OFF)
  set(EMBREE_TASKING_SYSTEM "INTERNAL" CACHE STRING "" FORCE)
  set(EMBREE_MAX_INSTANCE_LEVEL_COUNT "1" CACHE STRING "" FORCE)
  set(EMBREE_API_NAMESPACE embree_for_barney CACHE STRING "" FORCE)
  add_subdirectory(../submodules/embree builddir_rtc_embree EXCLUDE_FROM_ALL)
  add_library(embree_local INTERFACE)
  target_link_libraries(embree_local INTERFACE embree tasking)
  rtc_library_properties(embree)
  rtc_library_properties(tasking)
  rtc_library_properties(sys)
  target_compile_definitions(embree_local INTERFACE TASKING_INTERNAL)
  target_include_directories(embree_local INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/../submodules/embree/
    ${CMAKE_CURRENT_SOURCE_DIR}/../submodules/embree/common
  )
  target_include_directories(embree PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../submodules/embree/
    ${CMAKE_CURRENT_SOURCE_DIR}/../submodules/embree/common
  )
  target_include_directories(tasking PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/../submodules/embree/
    ${CMAKE_CURRENT_SOURCE_DIR}/../submodules/embree/common
  )
  if (WIN32)
    target_compile_options(embree_local 
      INTERFACE
      /D__SSE__ /D__SSE2__ /D__SSE3__ /D__SSSE3__ /D__SSE4_1__ /D__SSE4_2__
      )
      rtc_library_properties(embree_local)
  elseif(COMPILE_FOR_ARM)
    target_compile_options(embree_local INTERFACE -flax-vector-conversions -fsigned-char)
  endif()

#  add_library(barney_rtc_embree OBJECT #STATIC
  add_library(barney_rtc_embree STATIC
    embree/Float16.cpp
    embree/EmbreeBackend.cpp
    embree/Compute.cpp
    embree/Device.cpp
    embree/Buffer.cpp
    embree/Texture.cpp
    embree/GeomType.cpp
    embree/Geom.cpp
    embree/Triangles.cpp
    embree/UserGeom.cpp
    embree/Group.cpp
    embree/Denoiser.cpp
    )
  target_link_libraries(barney_rtc_embree PUBLIC
    barney_rtc_common
    embree_local
    )
  target_compile_definitions(barney_rtc_embree
    PUBLIC -DBARNEY_RTC_EMBREE=1)
  rtc_library_properties(barney_rtc_embree)
  if (BARNEY_OIDN_CPU)
    target_link_libraries(barney_rtc_embree PUBLIC OpenImageDenoise)
  endif()
#  target_link_libraries(barney_config INTERFACE embree_local)
else()
  add_library(barney_rtc_embree INTERFACE
    # dummy
    )
endif()

# ==================================================================
if (BARNEY_BACKEND_CUDA OR BARNEY_BACKEND_OPTIX)
  add_library(barney_rtc_cuda_common STATIC
    cudaCommon/Device.h
    cudaCommon/Device.cu
    cudaCommon/Texture.h
    cudaCommon/Texture.cu
    cudaCommon/TextureData.h
    cudaCommon/TextureData.cu
    cudaCommon/cuda-helper.cpp
    )
  target_link_libraries(barney_rtc_cuda_common PUBLIC
    barney_rtc_config
    )
set_target_properties(barney_rtc_cuda_common
  PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CXX_VISIBILITY_PRESET hidden
  CUDA_VISIBILITY_PRESET hidden
  POSITION_INDEPENDENT_CODE ON
  CUDA_USE_STATIC_CUDA_RUNTIME ON
  CUDA_SEPARABLE_COMPILATION ON
  POSITION_INDEPENDENT_CODE ON
  CUDA_RESOLVE_DEVICE_SYMBOLS ON
  CUDA_USE_STATIC_CUDA_RUNTIME ON
  )
  rtc_library_properties(barney_rtc_cuda_common)
else()
  add_library(barney_rtc_cuda_common INTERFACE) # DUMMY!
endif()


# ==================================================================
if (BARNEY_BACKEND_OPTIX)
  message("ENabling optix backend")
  add_library(barney_rtc_optix STATIC
    optix/Denoiser.h
    optix/Denoiser.cpp
    optix/Device.h
    optix/Device.cpp
    optix/Group.h
    optix/Group.cpp
    optix/Geom.h
    optix/Geom.cpp
    optix/Buffer.h
    optix/Buffer.cpp
    )
  target_link_libraries(barney_rtc_optix PUBLIC
    barney_rtc_config
    barney_rtc_cuda_common
    owl_static
    )
  target_compile_definitions(barney_rtc_optix
    PUBLIC -DBARNEY_RTC_OPTIX=1)
  rtc_library_properties(barney_rtc_optix)
  if (BARNEY_DISABLE_DENOISING)
    target_compile_definitions(barney_rtc_optix
      PUBLIC -DOPTIX_DISABLE_DENOISING=1)
  endif()
else()
  add_library(barney_rtc_optix INTERFACE) # DUMMY!
endif()


# ==================================================================
if (BARNEY_BACKEND_CUDA)
  message("ENabling native cuda backend")
  add_library(barney_rtc_cuda STATIC
    cuda/Device.h
    cuda/Device.cu
    cuda/Buffer.h
    cuda/Buffer.cpp
    cuda/Group.h
    cuda/Group.cu
    cuda/Geom.h
    cuda/Geom.cpp
    cuda/TraceKernel.h
    cuda/TraceKernel.cpp
    cuda/GeomType.h
    cuda/GeomType.cpp
    )
  target_link_libraries(barney_rtc_cuda PUBLIC
    barney_rtc_config
    barney_rtc_cuda_common
#    owl_static
    )
  target_compile_definitions(barney_rtc_cuda
    PUBLIC -DBARNEY_RTC_CUDA=1)
  rtc_library_properties(barney_rtc_cuda)

  if (USE_HIP)
    set_source_files_properties(cuda/Buffer.cpp PROPERTIES LANGUAGE HIP)
    set_source_files_properties(cuda/Geom.cpp PROPERTIES LANGUAGE HIP)
    set_source_files_properties(cuda/GeomType.cpp PROPERTIES LANGUAGE HIP)
    set_source_files_properties(cuda/TraceKernel.cpp PROPERTIES LANGUAGE HIP)
    
  endif()
  
else()
  add_library(barney_rtc_cuda INTERFACE) # DUMMY!
endif()


# ==================================================================
# ==================================================================
add_library(barney_rtc INTERFACE
  )
if (BARNEY_BACKEND_OPTIX)
  target_link_libraries(barney_rtc INTERFACE barney_rtc_optix)
endif()
if (BARNEY_BACKEND_EMBREE)
  target_link_libraries(barney_rtc INTERFACE barney_rtc_embree)
endif()
if (BARNEY_BACKEND_CUDA)
  target_link_libraries(barney_rtc INTERFACE barney_rtc_cuda)
endif()



