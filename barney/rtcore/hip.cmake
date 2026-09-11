if (USE_HIP)
  enable_language(HIP)
  # enable_language(HIP) above auto-detects the host GPU arch (and errors on a
  # no-GPU build host); pass -DCMAKE_HIP_ARCHITECTURES=... to override.
  message("#barney: building GPU code with HIP for CMAKE_HIP_ARCHITECTURES=${CMAKE_HIP_ARCHITECTURES}")
  # the cuda backend's GPU code is reused under HIP; OptiX is NVIDIA-only.
  set(BARNEY_HAVE_HIP ON)
  set(BARNEY_HAVE_CUDA OFF)
  set(CMAKE_CUDA_ARCHITECTURES)
  if (WIN32)
    # On Windows, CMake's Windows-Clang platform module injects -fuse-ld=lld-link
    # into HIP link commands, but the AMD clang driver rejects it when doing HIP
    # device-link (--hip-link); lld-link is the default host linker already.
    set(CMAKE_HIP_USING_LINKER_DEFAULT "")
  endif()

  # Optional hardware-RT backend on AMD GPUs via AMD HIPRT. HIPRT supplies the
  # BVH build + ray traversal (hardware-accelerated on RDNA2+, software on CDNA
  # such as gfx90a); barney keeps its function-pointer shading dispatch. HIPRT
  # is a discovered dependency (find via hiprt_ROOT / HIPRT_PATH), never
  # vendored -- mirrors how barney finds OptiX/OIDN.
  option(BARNEY_BACKEND_HIPRT "Enable HIPRT hardware-RT backend (AMD)?" OFF)
  if (BARNEY_BACKEND_HIPRT)
    find_package(hiprt REQUIRED)
    message("#barney: HIPRT backend enabled (hiprt at ${hiprt_LIBRARY})")
  endif()

  # AMD GPU support: compile barney's OptiX-free software ray-tracing backend
# (rtcore/cuda + rtcore/cudaCommon) through HIP/ROCm. When USE_HIP is ON the
# .cu sources are compiled with hipcc (LANGUAGE HIP); the CUDA/OptiX build is
# unaffected (this whole path is inert unless USE_HIP is requested).
option(USE_HIP "Build GPU code with HIP for AMD GPUs" OFF)


# On AMD GPUs the cuda backend's .cu sources are compiled as HIP. Mark the
# given sources LANGUAGE HIP so hipcc handles the device code (no effect on the
# CUDA/OptiX build, where this macro is never invoked).
macro(configure_source)
  foreach(src ${ARGN})
    get_filename_component(ext "${src}" EXT)
    if (ext STREQUAL ".cu")
      set_source_files_properties(${src} PROPERTIES
        LANGUAGE HIP
        COMPILE_OPTIONS "-fgpu-rdc")
    endif()
  endforeach()
endmacro()


