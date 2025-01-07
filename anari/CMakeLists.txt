## Copyright 2023 Ingo Wald
## SPDX-License-Identifier: Apache-2.0

option(BARNEY_BUILD_ANARI "Build ANARI device" ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

if (NOT (TARGET anari))
  find_package(anari 0.12.0 COMPONENTS code_gen)
  if (NOT anari_FOUND)
    message("barney: ANARI SDK not found - skipping this")
    return()
  endif()
endif()

include(GNUInstallDirs)

if (NOT BANARI_SKIP_QUERIES)
  ## Code generation ##
  anari_generate_queries(
#  DEVICE_TARGET ${PROJECT_NAME}
#  CPP_NAMESPACE visrtx
#  JSON_ROOT_LOCATION ${CMAKE_CURRENT_BINARY_DIR}
#  JSON_DEFINITIONS_FILE ${CMAKE_CURRENT_BINARY_DIR}/visrtx_device.json
#  JSON_EXTENSION_FILES ${DEVICE_JSONS}

DEVICE_TARGET barney
CPP_NAMESPACE BarneyDevice
JSON_DEFINITIONS_FILE ${CMAKE_CURRENT_SOURCE_DIR}/barney_device.json
JSON_ROOT_LOCATION ${CMAKE_CURRENT_BINARY_DIR}
JSON_EXTENSION_FILES ${DEVICE_JSONS}

#NAME barney
 #   PREFIX BarneyDevice
  #  CPP_NAMESPACE barney_device
   # JSON_DEFINITIONS_FILE ${CMAKE_CURRENT_SOURCE_DIR}/barney_device.json
    )
endif()

## Device target ##

project(anari_library_barney LANGUAGES CXX)

set(BANARI_SOURCES
  BarneyDeviceQueries.cpp
  BarneyGlobalState.cpp
  Camera.cpp
  Device.cpp
  Frame.cpp
  Geometry.cpp
  Group.cpp
  Instance.cpp
  Library.cpp
  Light.cpp
  Material.cpp
  Object.cpp
  Renderer.cpp
  Sampler.cpp
  SpatialField.cpp
  Surface.cpp
  Volume.cpp
  World.cpp
  )

# helper 'interface-'library that contains common settings for both
# static and shared anari
add_library(banari-config INTERFACE
  )
target_link_libraries(banari-config
  INTERFACE
  barney-config
  )
set_target_properties(banari-config
  PROPERTIES
  POSITION_INDEPENDENT_CODE ON
  )
#set_target_properties(banari-config
#  PROPERTIES
#  CUDA_ARCHITECTURES ${BARNEY_CUDA_ARCHITECTURES}
#  )
target_include_directories(banari-config
  INTERFACE
  ${CMAKE_CURRENT_LIST_DIR}/include
  ${CMAKE_CURRENT_LIST_DIR}
  ${CMAKE_CURRENT_BINARY_DIR}
  )

# ==================================================================
# default shared-linkage for anari loader library to load when barney
# device is requested
# ==================================================================
add_library(anari_library_barney SHARED
  ${BANARI_SOURCES}
  )

if (BARNEY_MPI)
  target_link_libraries(anari_library_barney
    PRIVATE
    banari-config
    "$<LINK_LIBRARY:WHOLE_ARCHIVE,barney_mpi_static>"
    )
else()
  target_link_libraries(anari_library_barney
    PRIVATE
    banari-config
    "$<LINK_LIBRARY:WHOLE_ARCHIVE,barney_static>"
    )
endif()
if (TARGET anari::helium)
  target_link_libraries(anari_library_barney PRIVATE
    anari::helium
    anari::anari
    )
else()
  target_link_libraries(anari_library_barney PRIVATE
    helium
    anari
    )
endif()

#set_target_properties(anari_library_barney
#  PROPERTIES
#  CUDA_SEPARABLE_COMPILATION ON
#  POSITION_INDEPENDENT_CODE ON
#  CUDA_RESOLVE_DEVICE_SYMBOLS ON
#  CUDA_USE_STATIC_CUDA_RUNTIME ON
#  CUDA_ARCHITECTURES ${BARNEY_CUDA_ARCHITECTURES}
#  )

include(GenerateExportHeader)
generate_export_header(anari_library_barney
  EXPORT_MACRO_NAME "BARNEY_LIBRARY_INTERFACE"
  )

## Installation ##

set_target_properties(anari_library_barney PROPERTIES INSTALL_RPATH "$ORIGIN")
install(TARGETS anari_library_barney
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )






# ==================================================================
# STATIC version of this device. need this for baking into pynari
# ==================================================================
add_library(anari_library_barney_static STATIC
  ${BANARI_SOURCES}
  )

if (BARNEY_MPI)
  target_link_libraries(anari_library_barney_static
    PRIVATE
    banari-config
    "$<LINK_LIBRARY:WHOLE_ARCHIVE,barney_mpi_static>"
    )
else()
  target_link_libraries(anari_library_barney_static
    PRIVATE
    banari-config
    "$<LINK_LIBRARY:WHOLE_ARCHIVE,barney_static>"
    )
endif()
if (TARGET anari::helium)
  # when building from installed source:
  target_link_libraries(anari_library_barney_static PUBLIC
    anari::helium
    anari_static
    )
else()
  target_link_libraries(anari_library_barney_static PUBLIC
    helium
    anari_static
    )
endif()
if (WIN32)
  target_compile_definitions(anari_library_barney_static PRIVATE -DNOGDI=1)
endif()


#set_target_properties(anari_library_barney_static
#  PROPERTIES
#  CUDA_ARCHITECTURES ${BARNEY_CUDA_ARCHITECTURES}
#  CUDA_SEPARABLE_COMPILATION ON
#  POSITION_INDEPENDENT_CODE ON
#  CUDA_RESOLVE_DEVICE_SYMBOLS ON
#  CUDA_USE_STATIC_CUDA_RUNTIME ON
#  )

#include(GenerateExportHeader)
#generate_export_header(anari_library_barney_static
#  EXPORT_MACRO_NAME "BARNEY_LIBRARY_INTERFACE"
#  )


