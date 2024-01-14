# Barney

# How to build and run

Unless you want to use the anari device (and know what you're doing):
build and run this through haystack; git clone as sibling project to
haystack (eg, `~/Projects/hayStack` and `~/Projects/barney`), then
when building haystack it should automatically find and build barney.

# TODOs

## add dedicated structured volume type

- barney: structured volume type (floats and uint8_t, maybe only
  floats for starters); re-use volume/MCAccelerator for macro cell
  infrastructure (build/computeranges/mapranges/...), but need to add
  DDA for traversal. only woodock, no blending.
  
- haystack: need 'loadablecontent' for structured volume that allows
  for splitting on the fly. for splitting use on-the-fly kd-tree
  (cheap for only N=numRanks parts); replicate boundary cells - maybe
  add full ghost cell at some later point.

## look into gridlets performance

- currently gridlets are handled 

## fix: object-space method(s) show artifacts for lander

## better/faster object-space prim intersection

look at interval subdivision methods? recursive element subdivsion
methods? subdivide (only) until num woodcock steps < const.

## look into "separate pass" for unstructured

problem right now is that unstructured meshes are "primitmives" in the
optix scene graph like any other - bu theyr'e much bigger (spatially)
and muuuuch more costly than others; and will suffer far more from
divergence issues; can also not do hardware-accelerated queries
because these would have to be done from in isec program which does't
work.

idea: use current optix world only for triangles, spheres, etc; and
have all umeshes be done in a separate pass. this pas could then do
restart-based marching from leaf to leaf; and do the rest either in
cuda (with full blocsk and compaction etcpp), or in its own raygen
program where we can trace rays for queries.

## Add direct openvdb rendering?


# ANARI / BARNARI

## building
- need to get, build, and "make install" ANARI-SDK: `git@github.com:KhronosGroup/ANARI-SDK`

- need to enable ANARI_DEVICE in barney ccmake config

- build barney/anari device in BARNEY build dir (not in haystack)

- add barney/bin dir to LD_LIBRARY_PATH, or link libanari_library_baryney.so to current dir

- `export ANARI_LIBRARY=barney`


## running amr data set:

- get and build `git@github.com:vtvamr/anari-volume-viewer.git`

- enable USE_HDF in build!

- get `envy:/space/zellmann/DR1/SILCC_hdf5_plt_cnt_0100`

- run `./anariVolumeViewer -l barney ~/models/SILCC_hdf5_plt_cnt_0100`

# UNSORTED

saved cmd-lines:

    mm -C ~/Projects/barney/bin install && mm && gdb ./anariViewer -l barney


    mm && cp ./hayThereQT ./hayThereOffline ~/ && scp ./hayThereQT ./hayThereOffline wally: && /home/wald/opt/bin/mpirun -n 1 -host hasky /home/wald/hayThereQT /home/wald/barney/jets-2*.umesh : -n 1 -host wally /home/wald/hayThereOffline /home/wald/barney/jets-2*umesh 


lander, per-rank split:
    mm && BARNEY_METHOD=object-space ./hayThereQT ~/per-rank/lander-small-vort_mag-9000.1*umesh  ~/per-rank/lander-small-vort_mag-9000.2*umesh -xf hayThere.xf  ~/per-rank/lander-small-vort_mag-9000.3*umesh  --camera -41.3633 18.2164 22.2561 -41.3629 18.2159 22.2554 0 1 0 -fovy 60






dependencies on new machine (for building and/or mpirun)
	cmake-curses-gui freeglut3-dev libglfw3-dev libhwloc-dev libibverbs-dev net-tools openssh-server



ander-perrank buggy view:
--camera 13.5086 12.7009 17.1061 16.0696 -0.166082 -0.711897 0 1 0 -fovy 60
