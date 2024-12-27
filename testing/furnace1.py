import matplotlib.pyplot as plt
import numpy as np
#from pynari import *
import pynari as anari
import random
import sys, getopt,PIL

fb_size = (800,800)
num_paths_per_pixel = 128
look_from = (13., 2., 3.)
look_at = (0., 0., 0.)
look_up = (0.,1.,0.)
fovy = 20.

furn_value = .5

random.seed(80577)

def add_sphere(pos, radius, material):
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

# create a "Lambertian" (ie, diffuse) anari matterial, using ANARI's
# 'matte' material
def make_lambertian(r,g,b):
    mat = device.newMaterial('matte')
    mat.setParameter('color',anari.float3,(1.,1.,1.))
    mat.commitParameters()
    return mat

def create_spheres():
    add_sphere(look_at, 1., make_lambertian(.5,.5,.5))

spheres = []

device = anari.newDevice('default')

create_spheres()


env_res_x = 128
env_res_y = 64
env_values = furn_value * np.ones(env_res_x*env_res_y*3, dtype=np.float32).reshape(3,env_res_x,env_res_y)
#todo: env_array = device.newArray(anari.FLOAT32_RGBA,env_ones)
env_array = device.newArray(anari.FLOAT32_VEC3,env_values)
env_light = device.newLight('hdri')
todo: env_light.setParameter('radiance',anari.ARRAY2D,env_array)
env_light.commitParameters()
    


world = device.newWorld()
world.setParameterArray('surface', anari.SURFACE, spheres )
array = device.newArray(anari.LIGHT, [env_light])
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
camera.setParameter('fovy',anari.FLOAT32,fovy*3.14/180)
camera.commitParameters()



# background gradient: use an image of 1 pixel wide and 2 pixels high
#bg_values = np.array(((.9,.9,.9,1.),(.15,.25,.8,1.)), dtype=np.float32).reshape((4,1,2))
#bg_gradient = device.newArray(anari.float4, bg_values)

renderer = device.newRenderer('default')
renderer.setParameter('ambientRadiance',anari.FLOAT32, .8)
renderer.setParameter('pixelSamples', anari.INT32, num_paths_per_pixel)
#renderer.setParameter('background', anari.float3, (.8,.8,.8) )
renderer.commitParameters()


frame = device.newFrame()

frame.setParameter('size', anari.uint2, fb_size)

frame.setParameter('channel.color', anari.DATA_TYPE, anari.UFIXED8_VEC4)
frame.setParameter('renderer', anari.OBJECT, renderer)
frame.setParameter('camera', anari.OBJECT, camera)
frame.setParameter('world', anari.OBJECT, world)
frame.commitParameters()

frame.render()
fb_color = frame.get('channel.color')

pixels = np.array(fb_color)#.reshape([height, width, 4])

out_file_name = ''
args = sys.argv[1:]
opts, args = getopt.getopt(args,"ho:",["help","output="])
for opt,arg in opts:
    if opt == '-h':
        printf('sample02.py [-o outfile.jpg]')
        sys.exit(0)
    elif opt == '-o':
        out_file_name = arg

if out_file_name == '':
    plt.imshow(pixels)
    plt.gca().invert_yaxis()
    plt.show()
else:
    im = PIL.Image.fromarray(pixels)
    im = im.transpose(PIL.Image.FLIP_TOP_BOTTOM)
    im = im.convert('RGB')
    im.save(out_file_name)




