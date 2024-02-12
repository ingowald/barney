#!/bin/bash
convert -resize x500! ls.jpg  -resize x500! ls-pond.jpg   -resize x500! +append ls-collage.jpg


convert -resize x500 engine.jpg  -resize x500 rotstrat-fuzzy.jpg  -resize x500 rotstrat-dense.jpg   -resize x500 +append structured-collage.jpg
