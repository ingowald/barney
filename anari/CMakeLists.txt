## Copyright 2023 Ingo Wald
## SPDX-License-Identifier: Apache-2.0

option(BARNEY_BUILD_ANARI "Build ANARI device" ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

find_package(anari 0.13.0 COMPONENTS code_gen)
if (NOT anari_FOUND)
  message("barney: ANARI SDK not found - skipping this")
  return()
endif()

include(GNUInstallDirs)

## Device target ##

project(anari_library_barney LANGUAGES CXX CUDA)


set(BANARI_SOURCES
dummy.cu
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
add_library(banari-config INTERFACE)
target_link_libraries(banari-config INTERFACE barney_config)
set_target_properties(banari-config PROPERTIES POSITION_INDEPENDENT_CODE ON)
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
add_library(anari_library_barney SHARED ${BANARI_SOURCES})

if (BARNEY_MPI)
  target_link_libraries(anari_library_barney
    PUBLIC
    $<BUILD_INTERFACE:banari-config>
    barney_mpi
    )
else()
if (BARNEY_HAVE_CUDA)
        set_source_files_properties(World.cpp
        PROPERTIES
        LANGUAGE CUDA
        )
        set_target_properties(anari_library_barney
        PROPERTIES
        LANGUAGE CUDA
        CUDA_SEPARABLE_COMPILATION ON
        )
  endif()
  target_link_libraries(anari_library_barney
    PUBLIC
    $<BUILD_INTERFACE:banari-config>
    #barney
    $<BUILD_INTERFACE:barney_static>
  )
endif()

target_link_libraries(anari_library_barney PUBLIC
  ${BARNEY_DEPENDS}
  )

target_link_libraries(anari_library_barney PRIVATE anari::helium anari::anari)
if(MSVC)
    target_compile_options(anari_library_barney INTERFACE
        "$<$<CONFIG:Debug>:/MTd>"
        "$<$<CONFIG:Release>:/MT>"
        "$<$<CONFIG:RelWithDebInfo>:/MT>"
        "$<$<CONFIG:MinSizeRel>:/MT>"
    )
endif()
set_target_properties(anari_library_barney
  PROPERTIES
  CUDA_SEPARABLE_COMPILATION ON
  CXX_VISIBILITY_PRESET hidden
  CUDA_VISIBILITY_PRESET hidden
  POSITION_INDEPENDENT_CODE ON
  CUDA_USE_STATIC_CUDA_RUNTIME ON
  LANGUAGE CUDA
  )

include(GenerateExportHeader)
generate_export_header(anari_library_barney
  EXPORT_MACRO_NAME "BARNEY_LIBRARY_INTERFACE"
)

## Installation ##

if (APPLE)
  set_target_properties(anari_library_barney PROPERTIES INSTALL_RPATH "$loader_path")
else()
  set_target_properties(anari_library_barney PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()
if (OFF)
install(TARGETS# barney
  anari_library_barney
  EXPORT barney-config
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
)
endif()

if (NOT BANARI_SKIP_QUERIES)
  ## Code generation ##
  anari_generate_queries(
    DEVICE_TARGET anari_library_barney
    CPP_NAMESPACE barney_device
    JSON_DEFINITIONS_FILE ${CMAKE_CURRENT_SOURCE_DIR}/barney_device.json
  )
endif()
