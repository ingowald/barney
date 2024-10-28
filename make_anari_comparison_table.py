#!/bin/bash

import os

backends=('helide', 'visrtx', 'ospray', 'barney')
haystack='/home/wald/Projects/hayStack/bin/hsOffline'
data='/home/wald/anari-tests/'
haystack_tests=(
    f'-os 1600 1200 -spp 16 raw:///{data}/magnetic-512-volume.raw:format=float:dims=512,512,512 -xf {data}/magnetic-512-volume.xf --camera 78.808 713.978 453.142 263.274 319.72 176.829 0 0 1 -fovy 60',
    f'{data}/suzy-quarry-cloudy.mini -os 1024 1024 --camera -0.643881 0.961672 2.63918 0 -0.00871944 0.00860357 0 1 0 -fovy 60',
    f'{data}/sponza.mini -os 1600 1200 --camera -5.73079 5.81282 0.00632362 16.2995 4.8213 0.736332 0 1 0 -fovy 60',
    f'{data}/bmw.mini -os 1600 1200 --camera -287.689 224.94 239.668 -35.1839 30.1148 -15.9624 0 1 0 -fovy 60',
    f'{data}/ls.mini -os 2460 1080 -spp 16 --camera -2976.901123 309.4327087 -3735.08252 1474.776611 1074.60376 4633.866699 0 1 0 -fovy 70',
    f'{data}/rungholt.mini -os 1600 1200 --camera 160.627 83.3032 252.997 55.8672 0 109.937 0 1 0 -fovy 60',
    f'{data}/embree-headlight.mini --camera -753.732 517.383 -655.327 -315.048 234.48 -451.202 0 1 0 -fovy 60',
    f'{data}/pp.mini -os 1600 1200  --camera -229013 58629.4 136061 -74489 17471.9 64223 0 1 0 -fovy 60',
    )
def haystack_table(f):
    f.write('<table rules="all">\n')
    f.write('<tr>\n')
    f.write('<h2>pynari samples</h2>\n')
    f.write('<table rules="all">\n')
    f.write('<tr>\n')
    for backend in backends:
        f.write(f'<th>{backend}</th>\n')
    f.write(f'<th>(native)</th>\n')
    f.write('</tr>\n')
    idx = 0
    for test in haystack_tests:
        f.write('<tr>\n')
        idx = idx+1
        for backend in backends:
            print(f'=== running haystack test {test} on backend {backend}')
            cmd = f"ANARI_LIBRARY={backend} {haystack} {test} -anari -o backends/{backend}_haystack{idx}.png"
            print(f'running "{cmd}"')
            os.system(cmd)
            cmd = f"convert backends/{backend}_haystack{idx}.png backends/{backend}_haystack{idx}.jpg"
            os.system(cmd)
            f.write(f'<td><img src="{backend}_haystack{idx}.jpg" width="95%"></td>\n')
        cmd = f"{haystack} {test} -o backends/native_haystack{idx}.png"
        os.system(cmd)
        cmd = f"convert backends/native_haystack{idx}.png backends/native_haystack{idx}.jpg"
        os.system(cmd)
        f.write(f'<td><img src="native_haystack{idx}.jpg" width="95%"></td>\n')
        f.write('</tr>\n')
    f.write('</table>\n')

def pynari_samples_table(f):
    f.write('<h2>pynari samples</h2>\n')
    f.write('<table rules="all">\n')
    f.write('<tr>\n')
    for backend in backends:
        f.write(f'<th>{backend}</th>\n')
    f.write('</tr>\n')
    for id in range(1,7):
        f.write('<tr>\n')
        for backend in backends:
            print(f'=== running sample0{id} on backend {backend}')
            cmd = f"ANARI_LIBRARY={backend} python3 sample0{id}.py -o backends/{backend}_sample0{id}.png"
            os.system(cmd)
            cmd = f"convert backends/{backend}_sample0{id}.png backends/{backend}_sample0{id}.jpg"
            os.system(cmd)
            f.write(f'<td><img src="{backend}_sample0{id}.jpg" width="100%"></td>\n')
        f.write('</tr>\n')
    f.write('</table>\n')

os.system('mkdir backends')
with open('backends/index.html','w') as f:
    f.write("<html><body>\n")
    #pynari_samples_table(f)
    haystack_table(f)
    f.write('</body></html>\n')
os.system('chmod a+rX backends -R')
