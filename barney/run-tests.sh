#!/bin/bash

# THIS SCRIPT ASSUMES VARIOUS PRE-BUILT APPLICATIONS AND LIBRARIES
# THAT USE BARNEY; IF YOU DON'T HAVE THOSE (OR NOT IN THE PROPER
# DIRECTORIES) THIS SCRIPT WON'T DO YOU MUCH GOOD. USE AT YOUR
# OWN RISK

out=/tmp/barney-test
hs=$HOME/Projects/hayStack/bin/hsOffline
mini=$HOME/mini/

python=/usr/bin/python3
pynari=$HOME/Projects/pynari

mkdir $out


make -j 8 -C ~/Projects/hayStack/bin/ 
make -j 8 -C ~/Projects/barney/bin/ install
make -j 8 -C ~/Projects/pynari/bin

# landscape, path view
$hs $mini/ls.mini -os 2460 1080 -spp 16 -o $out/ls-path.png --camera -2976.901123 309.4327087 -3735.08252 1474.776611 1074.60376 4633.866699 0 1 0 -fovy 70

# landscape, top
$hs $mini/ls.mini -os 2460 1080 -spp 16 -o $out/ls-top.png --camera -1993.13 3033.55 -2069.94 -375.562 -599.161 1070.11 0 1 0 -fovy 70

# magnetic
$hs -os 1600 1200 -o $out/magnetic.png -spp 16 raw:///home/wald/models/structured/magnetic-512-volume.raw:format=float:dims=512,512,512 -xf ~/models/structured/magnetic-512-volume.xf --camera 78.808 713.978 453.142 263.274 319.72 176.829 0 0 1 -fovy 60

# jets
$hs -os 1024 1024 -o $out/jets.png -spp 16 ~/models/umesh/jets.umesh -xf ~/models/umesh/jets.xf --camera 94.0945 39.217 -0.826244 44.6431 58.8093 54.9847 1 0 0 -fovy 60
$hs -os 1024 1024 -o $out/anari-jets.png -spp 16 ~/models/umesh/jets.umesh -xf ~/models/umesh/jets.xf --camera 94.0945 39.217 -0.826244 44.6431 58.8093 54.9847 1 0 0 -fovy 60 -anari


# ==================================================================
# different env-lighting tests on suzy model
# ==================================================================
cam="--camera -0.528471 0.478023 2.93096 0 -0.00871944 0.00860333 0 1 0 -fovy 60"
for f in quarry-cloudy symm-garden empty-room; do
    $hs -o $out/suzy-$f".png" $mini/suzy-$f".mini" $cam -spp 16 -os 1024 1024
done



# ==================================================================
# pynari tests
# ==================================================================

PYTHONPATH=$pynari/bin ANARI_LIBRARY=barney $python $pynari/sample01.py -o $out/pynari01.png
PYTHONPATH=$pynari/bin ANARI_LIBRARY=barney $python $pynari/sample02.py -o $out/pynari02.png
PYTHONPATH=$pynari/bin ANARI_LIBRARY=barney $python $pynari/sample03.py -o $out/pynari03.png
PYTHONPATH=$pynari/bin ANARI_LIBRARY=barney $python $pynari/sample04.py -o $out/pynari04.png


