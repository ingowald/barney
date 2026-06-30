# SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA
# CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

check_language(CUDA)
if (NOT (CMAKE_CUDA_COMPILER))
  set(BARNEY_HAVE_CUDA OFF)
  set(CMAKE_CUDA_ARCHITECTURES)
  message(AUTHOR_WARNING "#barney: no CUDA compiler found; disabling cuda/optix backend")
  return()
endif()

enable_language(CUDA)
set(BARNEY_HAVE_CUDA ON)

if (WIN32)
  set(BARNEY_CUDA_ARCHITECTURES_INIT "all-major")
else()
  set(BARNEY_CUDA_ARCHITECTURES_INIT "native")
endif()

set(CMAKE_CUDA_ARCHITECTURES
  "${BARNEY_CUDA_ARCHITECTURES_INIT}" CACHE STRING
  "Which CUDA architecture to build for")
endif()

set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} --expt-relaxed-constexpr")

macro(set_library_properties tgt)
  set_target_properties(${tgt}
    PROPERTIES
    POSITION_INDEPENDENT_CODE    ON
    VISIBILITY_INLINES_HIDDEN    ON
    CUDA_SEPARABLE_COMPILATION   ON
    CUDA_USE_STATIC_CUDA_RUNTIME ON
    CUDA_RESOLVE_DEVICE_SYMBOLS  ON
    CUDA_VISIBILITY_PRESET       hidden
    CXX_VISIBILITY_PRESET        hidden
  )
  if (APPLE)
    set_target_properties(${tgt} PROPERTIES INSTALL_RPATH "$loader_path")
  else()
    set_target_properties(${tgt} PROPERTIES INSTALL_RPATH "$ORIGIN")
  endif()
endmacro()

