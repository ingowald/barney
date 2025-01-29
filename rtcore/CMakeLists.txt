
# ==================================================================
option(BARNEY_DISABLE_OIDN "DISable OIDN even if found?" OFF)
if (NOT BARNEY_DISABLE_OIDN)
  find_package(OpenImageDenoise)
  if (OIDN_DEVICE_CPU)
    message("found OIDN *CPU* device support")
    set(BARNEY_OIDN_CPU ON)
  endif()
  if (OIDN_DEVICE_GPU)
    message("found OIDN *GPU* device support")
    set(BARNEY_OIDN_GPU ON)
  endif()
endif()

# ==================================================================

target_include_directories(barney_config INTERFACE
#  ${CMAKE_CURRENT_SOURCE_DIR}
  ${PROJECT_SOURCE_DIR}
  )
if (NOT BARNEY_HAVE_CUDA)
  target_link_libraries(barney_config INTERFACE owl-config)
endif()


# ==================================================================
add_library(barney_rtc_config INTERFACE
  )
target_link_libraries(barney_rtc_config INTERFACE barney_config)
set_property(TARGET barney_rtc_config PROPERTY
  CXX_STANDARD 20)



# ==================================================================
add_library(barney_rtc_common OBJECT
  common/Backend.cpp
  common/RTCore.cpp
  rtc.cpp
  )
target_link_libraries(barney_rtc_common PUBLIC barney_rtc_config)


# ==================================================================
if (BARNEY_BACKEND_EMBREE)
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

  add_subdirectory(../submodules/embree builddir_rtc_embree EXCLUDE_FROM_ALL)
  add_library(embree_local INTERFACE)
  target_link_libraries(embree_local INTERFACE embree tasking)
  target_compile_definitions(embree_local INTERFACE TASKING_INTERNAL)
  target_include_directories(embree_local INTERFACE
    ${CMAKE_CURRENT_SOURCE_DIR}/../submodules/embree/common)
  if (WIN32)
    target_compile_options(embree_local 
      INTERFACE
      /D__SSE__ /D__SSE2__ /D__SSE3__ /D__SSSE3__ /D__SSE4_1__ /D__SSE4_2__
      )
  elseif(COMPILE_FOR_ARM)
    target_compile_options(embree_local INTERFACE -flax-vector-conversions -fsigned-char)
  endif()

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
    )
  target_link_libraries(barney_rtc_embree PUBLIC
    barney_rtc_common
    embree_local
    )
#  target_link_libraries(barney_rtc_config INTERFACE embree_local)
  target_link_libraries(barney_config INTERFACE embree_local)
  #  target_link_libraries(barney_rtc PUBLIC barney_rtc_embree)
else()
  add_library(barney_rtc_embree INTERFACE
    # dummy
    )
endif()

# ==================================================================
if (BARNEY_BACKEND_CUDA OR BARNEY_BACKEND_OPTIX)
  add_library(barney_rtc_cuda_common OBJECT #STATIC
    cuda/CUDACommon.cu
    )
  target_link_libraries(barney_rtc_cuda_common PUBLIC
    barney_rtc_config
#    barney_rtc_common
#    cuda
    )
else()
  add_library(barney_rtc_cuda_common INTERFACE) # DUMMY!
endif()

# ==================================================================
if (BARNEY_BACKEND_CUDA)
  message("ENabling CUDA backend")
#  target_compile_definitions(barney_config INTERFACE -DBARNEY_BACKEND_CUDA=1)
  add_library(barney_rtc_cuda STATIC
    cuda/CUDABackend.cu
    )
#  target_link_libraries(barney_rtc_cuda PUBLIC barney_rtc_cuda_common)
  target_link_libraries(barney_rtc PUBLIC barney_rtc_cuda)
else()
  add_library(barney_rtc_cuda INTERFACE) # DUMMY!
endif()

# ==================================================================
if (BARNEY_BACKEND_OPTIX)
  message("ENabling optix backend")
#  target_compile_definitions(barney_config INTERFACE -DBARNEY_BACKEND_OPTIX=1)
  add_library(barney_rtc_optix OBJECT# STATIC
    optix/OptixBackend.cpp
    )
  target_link_libraries(barney_rtc_optix PUBLIC
    barney_rtc_config
    owl_static
#    barney_rtc_common
    #owl
    )
else()
  add_library(barney_rtc_optix INTERFACE) # DUMMY!
endif()





