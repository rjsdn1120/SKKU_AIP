import os
from glob import glob


def unzip():
    path ='/media/data2/rjsdn/AIP/data/img2/img'
    os.chdir(path)
    rlist=[]
    for x in os.walk('.'):
        for y in glob(os.path.join(x[0],'*.tar')):
            print(y)
            rlist.append(y)
            os.system(f'tar -xvf {y}')

def remove(l):
    for x in l:
        os.remove(x)
        
        
def mov(l):
    for x in l:
        p = os.path.join('/media/data2/rjsdn/AIP/data/img2/raw',x)
        os.system(f'mv {x} {p}')        