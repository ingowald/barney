# sample that tests 'curves' geometry by generating random 3D bezier
# curves

import matplotlib.pyplot as plt
import numpy as np
import pynari as anari
import random
import sys, getopt,PIL
import math

fb_size = (2*1024,1024)
camera_from = (-3,1.5,-2)
camera_dir = (1.,.1,.3)
camera_up = (0.,1.,0.)
fovy = 20.

random.seed(80577)

def add_sphere(pos, radius, material):
    geom = device.newGeometry('sphere')
    array = device.newArray1D(anari.FLOAT32_VEC3,np.array(pos,dtype=np.float32))
    geom.setParameter('vertex.position',anari.ARRAY1D,array)
    geom.setParameter('radius',anari.FLOAT32,radius)
    geom.commitParameters()

    surf = device.newSurface()
    surf.setParameter('geometry', anari.GEOMETRY, geom)
    surf.setParameter('material', anari.MATERIAL, material)
    surf.commitParameters()
    
    spheres.append(surf)

# create a "Lambertian" (ie, diffuse) anari matterial, using ANARI's
# 'matte' material
def make_lambertian(r,g,b):
    mat = device.newMaterial('matte')
    mat.setParameter('color',anari.float3,(r,g,b))
    mat.commitParameters()
    return mat

def mul(a,b):
    return (a[0]*b[0],a[1]*b[1],a[2]*b[2])

def add(a,b):
    return (a[0]+b[0],a[1]+b[1],a[2]+b[2])

def dot(a,b):
    return a[0]*b[0]+a[1]*b[1]+a[2]*b[2]

def cross(a,b):
    return (a[1]*b[2]-a[2]*b[1], a[2]*b[0]-a[0]*b[2], a[0]*b[1]-a[1]*b[0])

def make_vec(f):
    return (f,f,f);

def normalize(a):
    return mul(make_vec(1./math.sqrt(dot(a,a))), a)

def create_spheres():
    front = normalize(camera_dir)
    right = normalize(cross(front,camera_up))
    top   = normalize(cross(right,front))

    radius = .2
    add_sphere(add(camera_from,right),radius,make_lambertian(.7,.1,.1))
    add_sphere(add(camera_from,front),radius,make_lambertian(.1,.7,.1))
    add_sphere(add(camera_from,top),radius,make_lambertian(.1,.1,.7))


spheres = []

device = anari.newDevice('default')

create_spheres()

world = device.newWorld()
world.setParameterArray1D('surface', anari.SURFACE, spheres )
world.commitParameters()


camera = device.newCamera('omnidirectional')
camera.setParameter('position',anari.FLOAT32_VEC3, camera_from)
camera.setParameter('direction',anari.FLOAT32_VEC3, camera_dir)
camera.setParameter('up',anari.FLOAT32_VEC3, camera_up)
camera.commitParameters()


renderer = device.newRenderer('default')
renderer.setParameter('ambientRadiance',anari.FLOAT32, .8)
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

out_file_name = 'unitTest_omnicamera.png'

im = PIL.Image.fromarray(pixels)
im = im.transpose(PIL.Image.FLIP_TOP_BOTTOM)
im = im.convert('RGB')
print(f'@pynari: done. saving to {out_file_name}')
im.save(out_file_name)




