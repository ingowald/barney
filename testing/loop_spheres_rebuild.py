import matplotlib.pyplot as plt
import numpy as np
#from pynari import *
import pynari as anari
import random
import sys, getopt,PIL
import time

fb_size = (80,80)
num_paths_per_pixel = 1
look_from = (13., 2., 3.)
look_at = (0., 0., 0.)
look_up = (0.,1.,0.)
fovy = 20.

furn_value = .5

random.seed(80577)

geom = 0

# create a "Lambertian" (ie, diffuse) anari matterial, using ANARI's
# 'matte' material
def make_lambertian(r,g,b):
    mat = device.newMaterial('matte')
    mat.setParameter('color',anari.float3,(1.,1.,1.))
    mat.commitParameters()
    return mat


spheres = []

device = anari.newDevice('default')

pos = look_at
radius = 1.
material = make_lambertian(.5,.5,.5)

geom = device.newGeometry('sphere')
array = device.newArray(anari.FLOAT32_VEC3,np.array(pos,dtype=np.float32))
geom.setParameter('vertex.position',anari.ARRAY,array)
geom.setParameter('radius',anari.FLOAT32,radius)
geom.commitParameters()

surf = device.newSurface()
surf.setParameter('geometry', anari.GEOMETRY, geom)
surf.setParameter('material', anari.MATERIAL, material)
surf.commitParameters()
    
spheres.append(surf)



world = device.newWorld()
world.setParameterArray('surface', anari.SURFACE, spheres )
world.commitParameters()


camera = device.newCamera('perspective')
camera.setParameter('aspect', anari.FLOAT32, fb_size[0]/fb_size[1])
camera.setParameter('position',anari.FLOAT32_VEC3, look_from)
direction = [ look_at[0] - look_from[0],
              look_at[1] - look_from[1],
              look_at[2] - look_from[2] ] 
camera.setParameter('direction',anari.float3, direction)
camera.setParameter('up',anari.float3,look_up)
camera.setParameter('fovy',anari.FLOAT32,fovy*3.14/180)
camera.commitParameters()



renderer = device.newRenderer('default')
renderer.setParameter('ambientRadiance',anari.FLOAT32, .8)
renderer.setParameter('pixelSamples', anari.INT32, num_paths_per_pixel)
renderer.commitParameters()


frame = device.newFrame()

frame.setParameter('size', anari.uint2, fb_size)

frame.setParameter('channel.color', anari.DATA_TYPE, anari.UFIXED8_VEC4)
frame.setParameter('renderer', anari.OBJECT, renderer)
frame.setParameter('camera', anari.OBJECT, camera)
frame.setParameter('world', anari.OBJECT, world)
frame.commitParameters()

print("running 10k frames in a row, each frame we recommit world,"
      "recommit frame, render, and read frame buffer."
       "Look at top and nvidia-smi whether memory is growing...")
for _ in range(1000):
    print("setparam")
    geom.setParameter('radius',anari.FLOAT32,radius+_*.001)
    geom.commitParameters()
    surf.commitParameters()
    for s in spheres:
        s.commitParameters()
    world.commitParameters()
    frame.commitParameters()
    print("render")
    frame.render()
    time.sleep(.3)
    print("getfb")
    fb_color = frame.get('channel.color')

    pixels = np.array(fb_color)#.reshape([height, width, 4])

print('done')





