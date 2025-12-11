# SPDX-FileCopyrightText:
# Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier:
# Apache-2.0

import numpy as np
import pynari as anari
import random
import sys, getopt, PIL

from PIL import Image

fb_size = (1600,800)
look_from = (-3, -.2, 1)
look_at = (0, 0, 0)
fovy = 40.
look_up = (0.,0.,1.)
fovy = 50.
out_file_name = 'isosurface-umesh.jpg'

print('@pynari: -------------------------------------------------------')
print('@pynari: unstructured data, vertex centric scalars')
print('@pynari: -------------------------------------------------------')

device = anari.newDevice('default')

vertex_position = np.array([
    -1, -1, -1, 
    -1, -1, +1, 
    -1, +1, +1, 
    -1, +1, -1, 
    +1, -1, -1, 
    +1, -1, +1, 
    +1, +1, +1, 
    +1, +1, -1, 
    ],dtype=np.float32)
                    
index = np.array([
    #, hexes
#    0, 1, 3, 2,
#    4, 5, 7, 6
    0, 1, 2, 3, 4, 5, 6, 7
    ],dtype=np.uint32)

cell_index = np.array([
    0
                      ],dtype=np.uint32)
# vtk style numbering:
elt_tet=10
elt_hex=12
elt_wedge=13
elt_pyr=14
cell_type = np.array([elt_hex],dtype=np.uint8)
# for per vertex
vertex_data = np.array([
1,
0,
1,
0,
0,
1,
0,
1,
],dtype=np.float32)

spatial_field = device.newSpatialField('unstructured')
spatial_field.setParameterArray1D('cell.type',anari.UINT8,cell_type)
spatial_field.setParameterArray1D('cell.index',anari.UINT32,cell_index)
spatial_field.setParameterArray1D('index',anari.UINT32,index)
spatial_field.setParameterArray1D('vertex.position',anari.FLOAT32_VEC3,vertex_position)
spatial_field.setParameterArray1D('vertex.data',anari.FLOAT32,vertex_data)
spatial_field.commitParameters()

xf = np.array([0, 0, 1, 1,
               0, 1, 1, 1,
               0, 1, 0, 1,
               1, 1, 0, 1,
               1, 0, 0, 1,
               1, 0, 1, 1,
               0, 0, 1, 1,
               ],dtype=np.float32)
# for cell-centered data, make sure we see all the prims
xf_array = device.newArray1D(anari.float4,xf)

volume = device.newVolume('transferFunction1D')
volume.setParameter('color',anari.ARRAY1D,xf_array)
volume.setParameter('value',anari.SPATIAL_FIELD,spatial_field)
volume.setParameter('unitDistance',anari.FLOAT32,1.)
volume.commitParameters()

iso_geom = device.newGeometry('isosurface')
iso_geom.setParameter('isovalue',anari.FLOAT32,.45)
iso_geom.setParameter('field',anari.SPATIAL_FIELD,spatial_field)
iso_geom.commitParameters()

mat = device.newMaterial('physicallyBased')
mat.setParameter('baseColor',anari.float3,(.1,.1,.8))
mat.setParameter('ior',anari.FLOAT32,1.45)
mat.setParameter('metallic',anari.FLOAT32,.1)
mat.setParameter('specular',anari.FLOAT32,0.)
mat.setParameter('roughness',anari.FLOAT32,.35)
mat.setParameter('opacity',anari.FLOAT32,.85)
mat.commitParameters()

surf = device.newSurface()
surf.setParameter('geometry', anari.GEOMETRY, iso_geom)
surf.setParameter('material', anari.MATERIAL, mat)
surf.commitParameters()


world = device.newWorld()
#world.setParameterArray1D('volume', anari.VOLUME, [ volume ] )
world.setParameterArray1D('surface', anari.SURFACE, [ surf ] )
light = device.newLight('directional')
light.setParameter('direction', anari.float3, ( 1., -1., -1. ) )
light.commitParameters()

array = device.newArray1D(anari.LIGHT, [light])
world.setParameter('light', anari.ARRAY1D, array)
world.commitParameters()


camera = device.newCamera('perspective')
camera.setParameter('aspect', anari.FLOAT32, fb_size[0]/fb_size[1])
camera.setParameter('position',anari.FLOAT32_VEC3, look_from)
direction = [ look_at[0] - look_from[0],
              look_at[1] - look_from[1],
              look_at[2] - look_from[2] ] 
camera.setParameter('direction',anari.float3, direction)
camera.setParameter('up',anari.float3,look_up)
camera.setParameter('fovy',anari.float,fovy*3.14/180)
camera.commitParameters()


# background gradient: use an image of 1 pixel wide and 2 pixels high
bg_values = np.array(((.9,.9,.9,1.),(.15,.25,.8,1.)),
                     dtype=np.float32).reshape((2,1,4))
bg_gradient = device.newArray2D(anari.float4, bg_values)


renderer = device.newRenderer('default')
renderer.setParameter('ambientRadiance',anari.FLOAT32, 1.5)
renderer.setParameter('background', anari.ARRAY2D, bg_gradient)
renderer.setParameter('pixelSamples', anari.INT32, 4)
renderer.commitParameters()


frame = device.newFrame()

frame.setParameter('size', anari.uint2, fb_size)

frame.setParameter('channel.color', anari.DATA_TYPE, anari.UFIXED8_RGBA_SRGB)
frame.setParameter('renderer', anari.RENDERER, renderer)
frame.setParameter('camera', anari.CAMERA, camera)
frame.setParameter('world', anari.WORLD, world)
frame.commitParameters()

frame.render()
fb_color = frame.get('channel.color')
pixels = np.array(fb_color)

im = Image.fromarray(pixels)
im = im.transpose(PIL.Image.FLIP_TOP_BOTTOM)
im = im.convert('RGB')
print(f'@pynari: done. saving to {out_file_name}')
im.save(out_file_name)





