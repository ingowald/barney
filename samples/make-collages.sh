#!/bin/bash


#sample-testorb.jpg
#sample-headlight.png
#sample-landscape.png
convert  \
    sample-landscape.png -geometry x256\
    sample-headlight.png -geometry x256\
    sample-testorb.jpg -geometry x256\
    +append collage-triangles.jpg

#sample-capsules.png
#sample-curves.jpg
convert  \
    sample-capsules.png -geometry x256\
    sample-pynari1.jpg -geometry x256\
    +append row1.jpg

convert  \
    sample-cylinders.jpg -geometry x256\
    sample-cones.jpg -geometry x256\
    sample-spheres.png -geometry x256\
    sample-curves.jpg -geometry x256\
    +append row2.jpg

convert row1.jpg -geometry 1024x row2.jpg -geometry 1024x  -append collage-usergeom.jpg

#sample-kingsnake.png
#sample-chest.png
#sample-umesh1.png
convert  \
    sample-chest.png -geometry x256\
    sample-kingsnake.png -geometry x256\
    sample-umesh.png -geometry x256\
    +append collage-volumes.jpg
