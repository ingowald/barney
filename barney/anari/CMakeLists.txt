# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
# CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# "backend" variable gets set by parent dir, and is one of either
# 'cpu', 'cuda', 'optix', 'hip' or 'hiprt'

#set(BARNEY_INSTALL_TARGETS)
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
set(MPI_SOURCES
  MPIDevice.cpp
)

rtc_configure_source(dummy.cu)

if (NOT BARNEY_AUTO_GENERATE_QUERIES)
  LIST(APPEND BANARI_SOURCES generated/anari_library_barney_queries.cpp)
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
message("anari_library_${BARNEY_DEVICE_NAME}_static ")
add_library(anari_library_${BARNEY_DEVICE_NAME}_static STATIC ${BANARI_SOURCES})
target_link_libraries(anari_library_${BARNEY_DEVICE_NAME}_static PUBLIC
  ${BARNEY_DEVICE_NAME}_static
  #$<BUILD_INTERFACE:barney_config_${backend}>
#  $<BUILD_INTERFACE:banari-config-${backend}>
  anari::helium
)
rtc_library_properties(anari_library_${BARNEY_DEVICE_NAME}_static)

# ==================================================================
# the 'full' barney+anari, as a real shared loadable library, but
# containing only a single backend
# ==================================================================
message("anari_library_${BARNEY_DEVICE_NAME}")
add_library(anari_library_${BARNEY_DEVICE_NAME} SHARED SingleDeviceLibrary.cpp)
target_link_libraries(anari_library_${BARNEY_DEVICE_NAME} PUBLIC
  anari_library_${BARNEY_DEVICE_NAME}_static
)
set_target_properties(anari_library_${BARNEY_DEVICE_NAME}
  PROPERTIES
  POSITION_INDEPENDENT_CODE    ON
  VISIBILITY_INLINES_HIDDEN    ON
  CUDA_SEPARABLE_COMPILATION   ON
  CUDA_USE_STATIC_CUDA_RUNTIME ON
  CUDA_RESOLVE_DEVICE_SYMBOLS  ON
  CUDA_VISIBILITY_PRESET       hidden
  CXX_VISIBILITY_PRESET        hidden
)

# json files that specify configs global to all of barney, no matter the backend
file(GLOB COMMON_DEVICE_JSONS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/json/common "*.json")

# json files that specify configs only valid on specific backends
file(GLOB BACKEND_DEVICE_JSONS RELATIVE ${CMAKE_CURRENT_SOURCE_DIR}/json/${backend} "*.json")
foreach(file ${BACKEND_DEVICE_JSONS} ${COMMON_DEVICE_JSONS})
  file(GENERATE OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/${file} INPUT ${file})
endforeach()

if (BARNEY_AUTO_GENERATE_QUERIES)
  anari_generate_queries(
    DEVICE_TARGET anari_library_${BARNEY_DEVICE_NAME}
    # CPP_NAMESPACE BARNEY_DEVICE_NAME ${BANARI_NS}
    CPP_NAMESPACE ${BARNEY_DEVICE_NAME}::anari
    JSON_ROOT_LOCATION ${CMAKE_CURRENT_BINARY_DIR}
    JSON_DEFINITIONS_FILE ${CMAKE_CURRENT_BINARY_DIR}/barney_device.json
    JSON_EXTENSION_FILES ${BACKEND_DEVICE_JSONS} ${COMMON_DEVICE_JSONS}
  )
endif()

if (BARNEY_MPI)
  if (BARNEY_AUTO_GENERATE_QUERIES)
    anari_generate_queries(
      DEVICE_TARGET anari_library_${BARNEY_DEVICE_NAME}_mpi
      # CPP_NAMESPACE ${BANARI_NS}
      CPP_NAMESPACE ${BARNEY_DEVICE_NAME}::anari
      JSON_ROOT_LOCATION ${CMAKE_CURRENT_BINARY_DIR}
      JSON_DEFINITIONS_FILE ${CMAKE_CURRENT_BINARY_DIR}/barney_device.json
      JSON_EXTENSION_FILES ${DEVICE_JSONS}
    )
  endif()
endif()

#message("anari BARNEY_INSTALL_TARGETS ${BARNEY_INSTALL_TARGETS}")
#set(BARNEY_INSTALL_TARGETS ${BARNEY_INSTALL_TARGETS} PARENT_SCOPE)
# install(
#   TARGETS
#   ${BARNEY_INSTALL_TARGETS}
#   LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
#   NAMELINK_SKIP
#   # on Windows put the dlls into bin
#   RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
#   # ... and the import lib into the devel package
#   ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
# )
