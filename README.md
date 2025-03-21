# Barney - A OptiX-CUDA Accelerated Path Tracer for Data-Parallel Sci-Vis Rendering


DISCLAIMER: Barney is a first prototype of a possibly-to-be data
parallel ray/path tracer for sci-vis content. It can actually do quite
a bit of "stuff" already; however, it is still experimental software.
In particular, `barney` is still *very much* "in flux": There are no
stable releases, nor are any of the feature-sets fully "spec'ed" or
even committed to; and any of the information in the remainder of this
document way well be outdated or even plain wrong by the time you are
going to read this. I will be happy about any feedback, bug reports,
reports about errors, broken documentation, etc, and will fix what I
can and who quickly I can - but do not expect this to be a finished
product in any way, shape, or form.

# Building and Running

Barney is not a stand-alone "renderer" or "vis-tool"; it is a library
with an API, and needs other applications to build towards it. As
such, it is never "run" on its own; it also needs to be run from another
application (e.g., `hayStack`, at http://github.org/ingowald/hayStack),
or from any application that supports the ANARI API (see https://www.khronos.org/anari/).


## Dependencies for building Barney

Barney requires the following additional tools and/or packages to build:

- `cmake`, for building

- `CUDA`, version 12 and up.

- `OWL` (https://github.com/owl-project/owl). Note OWL gets pulled in as a git
   submodule, no need to externally get and install.

- `OptiX`, as part of OWL. See documentation in OWL (https://github.com/owl-project/owl) for 
   where to get, and how to best install for OWL to easily find it)
   
- For data parallel multi-*node* rendering: MPI. *Running* barney
  requires a CUDA-aware MPI, for *building* this should not matter. We 
  typically develop under and test with OpenMPI 4.1.6.

## Building Barney - no ANARI

Barney is built via CMake, using the cmake build/install procedure
that works on all of Linux, Windows, and Mac; but how to build it depends
on whether you want to build _just_ barney, or also (more likely) the
"Banari" ANARI device through which ANARI apps can use it.

## Building Barney - no ANARI

For native barney apps - ie, that do not want to use ANARI - you
can build without the ANARI SDK as follows:

``` bash
cd barney
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=<same-install-dir-as-anari> [options]
cmake --build .. [ --config Release ]
cmake --install .. [ --config Release ]
```

Two notes for windows builds:

	- The "linux-like" calls to `cmake ...` etc will also work under windows, in
	both cmd.exe and powershell. You can of course also use the cmake and VS gui's, but you *will* have to then run the install target.
	
	- On windows you should specify the `--config Release` (or `Debug`, if you prefer) for both build and install; under Linux you can ignore this. For an ANARI build (see below) oyu want to use the same config as used for the ANARI SDK.

In the cmake config step, you may specify the following options:
   

- `-DOptiX_INSTALL_DIR=<path to optix>` . You can also set an env
  variable of the same name.

- `-DBARNEY_BACKEND_EMBREE=ON` Enables (slower) CPU rendering without
  a GPU. Off by deault

- `-DBARNEY_BACKEND_OPTIX=OFF` Optix is on by default, this will turn
  it off.

- `-DBARNEY_DISABLE_DENOISING=ON` Denoising is on by default (assuming
  a suitable denoiser can be found), this will turn it off

- `-DBARNEY_MPI=ON` Controls whether `barney_mpi` and
  `libanari_library_barney_mpi` will be build. Off by default,
  requires an appropriate (cuda aware!) MPI

## Building Barney - *with* ANARI

The ANARI build of barney works in pretty much the same way (and with
the same options), but requires a pre-built and installed `ANARI-SDK` from
https://github.com/KhronosGroup/ANARI-SDK. As to the time of this writing,
you need ANARI SDK version 0.13.2 (or `next_release` branch)

First, build and install the ANARI SDK:

``` bash
cd ANARI-SDK
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=<install-dir>
cmake --build .. [ --config Release ]
cmake --install .. [ --config Release ]
```

Then, build barney as described above, using same install dir:
``` bash
cd barney
mkdir build
cd build
cmake .. -DCMAKE_INSTALL_PREFIX=<same-install-dir-as-anari> [options]
cmake --build .. [ --config Release ]
cmake --install .. [ --config Release ]
```

To install into a different install dir than ANARI (you probably shouldn't?):
``` bash
cd barney
mkdir build
cd build
cmake .. \
   -DCMAKE_INSTALL_PREFIX=<same-install-dir-as-anari> \
   -DCMAKE_PREFIX_PATH=<anari-install-dir>,<whatever-other_paths> \
   [options]
cmake --build .. [ --config Release ]
cmake --install .. [ --config Release ]
```

This accepts the same options as above. Note the command-line cmake build/install 
calls will also work on windows (though of course you can also use cmake-gui
and visual studio gui if you prefer so).



# Examples of Supported Geometry and Volume Types 

## Triangle Meshes (including Instances and Color- and Opacity Textures)

Example: PBRT landscape in `miniScene` (http://github.com/ingowald/miniScene) format:

![](jpg/ls-collage.jpg)

Working:

- Image textures and texture coordinates are supported

- Alpha texturing for fully-transparent textures is supported (alpha
  channel, or dedicated alpha texture)

- Instancing is fully supported

Missing/incomplete:

- Material model is still very basic; reflection, refraction etc are not yet supported.

## Non-Triangular Surface Types for Sci-Vis

- prototypical support for ANARI-style `spheres` geometry (x,y,z,radius per sphere)

(README need updating)

- prototypical support for ANARI-style `cylinders` geometry (xyzradius for each endpoint)

(README need updating)


## Structured Volume Data

Structured Volume Data (`float`, `uint8` and `uint16` are supported,
and any volume can be distributed across different ranks by each rank
having different porions of that volume. `Barney` being intended for
sci-vis, every volume can have its own transfer function.

![](jpg/structured-collage.jpg)

a) `Engine` (`256x256x128_uint8`), Dell XPS Laptop

b) `rot-strat` (`4096x4096x4096_float`) data-parallel on 8x `RTX8000` (4 nodes, 2 GPUs each)

c) same, with a very dense transfer function:

## Unstructured-Mesh Data

- some first light exists, README needs updating.


## Block-Structed AMR Data

- some first light exists, README needs updating.




# ANARI / BARNARI

Though `barney` is not *limited to* ANARI (it is its own library, with
its own API), it can also be configured to build a (still every much
experimental!) `ANARI` "device" that exposes some of barney's
functionality. Once enabled in the cmake build, this builds a
`libanari_library_barney.so` that implemnets an ANARI device, and that
any ANARI-capable renderer can then load as the `"barney`" device.

Note: To distinguish between the (general) ANARI *API* and the
specific barney-based implementation of this API we typically refer to
this implementation as the `(B)ANARI` device, or simply as `banari`.

Disclaimer: if barney is still experimental, `banari` is even more so!
Not all `barney` functionality is exposed in `banari`, nor is every ANARI 
feature supported by `banari` - and even for the features that *are* supported, 
there may be some significant memory- or compute-overhead when going through this
device.

## Building BANARI:

- dependencies: `libgtk-3-dev`

- need to get, build, *and install* the ANARI-SDK:
  `git@github.com:KhronosGroup/ANARI-SDK`. Note the SDK *must*
  be installed for barney to properly find it.

- need to enable the `BARNEY_BUILD_ANARI` flag in barney's cmake
  config

- build `barney/anari` device in BARNEY build dir (not in haystack)

- add `barney/bin` dir (or whatever your build dir is called) to
  `LD_LIBRARY_PATH`, or link `libanari_library_baryney.so` into current dir

- `export ANARI_LIBRARY=barney`


# Version History

## v0.9.0

- major rework that allows 'rtcore' abstraction and multiple backends
- support both optix and embree (cpu only) backends
- completely reworked cmake build system (and in particular how stuff
  gets linked)
- reworked install/export/import system, exporting both `barney` and
  `barney_mpi`; should always get used through `find_package(barney)`
  and then linked as import target(s) `barney::barney` and (if found)
  `barney::barney_mpi`
- anari device split into `anari_library_barney` and
  `anari_library_barney_mpi`
- version used for pynari backend that works on all of
  linux/windows/mac and both cpu/gpu
  
Known limitations/issues:
- AMR support currently disabled
- umesh support only support cubql sampler

## v0.8.0

- updated to latest anari 0.12 SDK
- changed cuda arch detection to use 'native' by default, but allowing
  to override on cmdline.
- updated cuda arch detection in owl and cubql so as to allow barney
  to tell them which one to use (so it matches across all projects)
- reworked path tracing (and in particular MIS code) to (mostly-)pass
  a furnace test
