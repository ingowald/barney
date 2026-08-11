# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
# CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# "backend" variable gets set by parent dir, and is one of either
# 'cpu', 'cuda', 'optix', 'hip' or 'hiprt'

set(BANARI_SOURCES
  Array.cpp
  BarneyGlobalState.cpp
  Camera.cpp
  Device.cpp
  Frame.cpp
  Geometry.cpp
  Group.cpp
  Instance.cpp
#  Library.cpp
  Material.cpp
  Object.cpp
  Renderer.cpp
  Sampler.cpp
  SpatialField.cpp
  Surface.cpp
  Volume.cpp
  World.cpp
  # -----------
  light/Light.h
  light/Light.cpp
  light/HDRI.h
  light/HDRI.cpp
  light/Directional.h
  light/Directional.cpp
  light/Point.h
  light/Point.cpp

  # 'force' a single cuda file into the library, so cmake will
  # properly recognize this as requiring linking with cuda.
  dummy.cu
)
if (BARNEY_MPI)
  set(BANARI_MPI_SOURCES
    MPIDevice.cpp)
else()
  set(BANARI_MPI_SOURCES )
endif()

rtc_configure_source(dummy.cu)


if (BARNEY_AUTO_GENERATE_QUERIES)
  # json files that specify configs only valid on specific backends
#  file(GLOB BACKEND_DEVICE_JSONS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/json/${backend} "*.json")
  foreach(file ${BACKEND_DEVICE_JSONS} ${COMMON_DEVICE_JSONS})
    file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${file} INPUT ${file})
  endforeach()

  file(GLOB_RECURSE API_JSONS ${ANARI_CODE_GEN_ROOT}/api/*.json)
  file(GLOB_RECURSE MY_JSONS ${CMAKE_CURRENT_SOURCE_DIR}/json/${backend}/*.json)
  
  add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/generated_queries.h ${CMAKE_CURRENT_BINARY_DIR}/generated_queries.cpp
    COMMAND ${Python3_EXECUTABLE} ${ANARI_CODE_GEN_ROOT}/generate_queries.py
#    --json ${CMAKE_CURRENT_BINARY_DIR}
    #${EXTRA_JSON_OPTION}
    #--prefix barney_${backend}
    # this will create <bulddir>/generated_queries.h/cpp
    --prefix generated 
    --json ${ANARI_CODE_GEN_ROOT}
    --json ${CMAKE_CURRENT_SOURCE_DIR}/json/${backend}/
    --device ${CMAKE_CURRENT_SOURCE_DIR}/json/${backend}/barney_device.json
    --namespace barney_${backend}
    #--output ${CMAKE_CURRENT_BINARY_DIR}/generated_queries.cpp
    --output ${CMAKE_CURRENT_BINARY_DIR}
    WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    DEPENDS ${CORE_JSONS} ${MY_JSONS}
  )
  set(QUERIES_SOURCES ${CMAKE_CURRENT_BINARY_DIR}/generated_queries.cpp)
  set(QUERIES_INCLUDE ${CMAKE_CURRENT_BINARY_DIR})
else()
  # todo
  set(QUERIES_SOURCES)
endif()


# helper 'interface-'library that contains common settings for both
# static and shared anari
#add_library(banari-${backend} INTERFACE)
#target_link_libraries(banari-config-${backend} INTERFACE barney_config)
#set_target_properties(banari-config-${backend} PROPERTIES POSITION_INDEPENDENT_CODE ON)
#target_include_directories(banari-config-${backend}
#  INTERFACE
#  ${CMAKE_CURRENT_LIST_DIR}/include
#  ${CMAKE_CURRENT_LIST_DIR}
#  ${CMAKE_CURRENT_BINARY_DIR}
#)

# ==================================================================
# the 'full' barney+anari, library+device, for a single backend, in
# static linkage (so it can be used in possibly multiple shared libraries)
# ==================================================================
message("anari_library_barney_${backend}_static ")
add_library(anari_library_barney_${backend}_static STATIC
  ${BANARI_SOURCES}
  ${QUERIES_SOURCES}
  ${BANARI_MPI_SOURCES}
)
target_include_directories(anari_library_barney_${backend}_static PRIVATE
  ${QUERIES_INCLUDE})
target_link_libraries(anari_library_barney_${backend}_static PUBLIC
  barney_${backend}_static
  anari::helium
)
rtc_library_properties(anari_library_barney_${backend}_static)


# ==================================================================
# the 'full' barney+anari, as a real shared loadable library, but
# containing only a single backend
# ==================================================================
add_library(anari_library_barney_${backend} SHARED SingleDeviceLibrary.cpp)
target_link_libraries(anari_library_barney_${backend} PUBLIC
  anari_library_barney_${backend}_static
)
set_target_properties(anari_library_barney_${backend}
  PROPERTIES
  POSITION_INDEPENDENT_CODE    ON
  VISIBILITY_INLINES_HIDDEN    ON
  CUDA_SEPARABLE_COMPILATION   ON
  CUDA_USE_STATIC_CUDA_RUNTIME ON
  CUDA_RESOLVE_DEVICE_SYMBOLS  ON
  CUDA_VISIBILITY_PRESET       hidden
  CXX_VISIBILITY_PRESET        hidden
)




install(
  TARGETS
  anari_library_barney_${backend}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  NAMELINK_SKIP
  # on Windows put the dlls into bin
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  # ... and the import lib into the devel package
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
