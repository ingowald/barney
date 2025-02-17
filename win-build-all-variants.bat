mkdir builddir-win-optix-only
mkdir builddir-win-embree-only
mkdir builddir-win-both

cmake . -DOptiX_INSTALL_DIR=c:/users/ingow/Projects/optix/ -DCMAKE_INSTALL_PREFIX=c:/users/ingow/opt -DBARNEY_MPI=OFF -DBARNEY_BACKEND_EMBREE=OFF -B builddir-win-optix-only
cmake . -DCMAKE_INSTALL_PREFIX=c:/users/ingow/opt -DBARNEY_MPI=OFF -DBARNEY_BACKEND_EMBREE=ON -DBARNEY_DISABLE_CUDA=ON -B builddir-win-embree-only
cmake . -DOptiX_INSTALL_DIR=c:/users/ingow/Projects/optix/ -DCMAKE_INSTALL_PREFIX=c:/users/ingow/opt -DBARNEY_MPI=OFF -DBARNEY_BACKEND_EMBREE=ON -B builddir-win-both

cmake --build build builddir-win-optix-only --config Release
cmake --build build builddir-win-optix-only --config Debug

cmake --build build builddir-win-embree-only --config Release
cmake --build build builddir-win-embree-only --config Debug

cmake --build build builddir-win-both --config Relase
cmake --build build builddir-win-both --config Debug
