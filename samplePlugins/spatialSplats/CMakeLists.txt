# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
# CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set(plugin_name splats)

message("building plugin barney_${backend}_plugin_${plugin_name}")

# ==================================================================
# step 1: define the actual plugin library, which defined the actual
# anari::Geometry, anari::SpatialField etc objects we eventually want
# to export. This will be a STATIC library that barney and plugin can
# resolve its symbols from during linking. This can also use its own
# PRIVATE cmake fields for options, includr dirs, etc
# ==================================================================
add_library(barney_${backend}_${plugin_name} STATIC
  native/Splats.h
  native/Splats.cu
  anari/Splats.h
  anari/Splats.cpp
)
rtc_configure_source(native/Splats.cu)
target_include_directories(barney_${backend}_${plugin_name} PUBLIC
  ${CMAKE_CURRENT_LIST_DIR})

# ==================================================================
# step 2: link our object-lib plugin to anari_library_barney_static
# (mind the 'static'!), so our plugin can resolve itself to whatever
# functionality is compiled into barney, rtc, and banari
# ==================================================================
target_link_libraries(barney_${backend}_${plugin_name} PUBLIC
  #  barney_rtc_${backend}
  anari_library_barney_${backend}_static
)


# ==================================================================
# step 3: create a helper library for _registering_ this plugin. This
# need only contain a single file that does the `registerPlugin()`
# thing, but it has to be OBJECT linkage to work, so we'll use a
# separate library. This can then link to the static lib from steps 1
# and 2 to the actual objects being exported
# ==================================================================
add_library(barney_${backend}_plugin_${plugin_name} OBJECT
  registerPlugin.cpp)
target_link_libraries(barney_${backend}_plugin_${plugin_name} PRIVATE
  barney_${backend}_${plugin_name})

# step 3.2: link to 
#target_link_libraries(barney_${backend}_plugin_${plugin_name} PUBLIC
#  #  barney_rtc_${backend}
#  anari_library_barney_${backend}_static
#)

# ==================================================================
# step 4: link our plugin library to both of the generated shared
# libs: anari_libary_barney_{backend} for the single-backend one we're
# currently compiling, and anari_libary_barney for the multi-backend
# one that we're currently adding this backend to.
# ==================================================================
target_link_libraries(anari_library_barney_${backend}
  PUBLIC
  barney_${backend}_plugin_${plugin_name}  
)
target_link_libraries(anari_library_barney
  PUBLIC
  barney_${backend}_plugin_${plugin_name}  
)
