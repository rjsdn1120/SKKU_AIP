import os
from glob import glob
import shutil
import random

def remove(data):
    for x in data:
        if len(x)<10:
            shutil.rmtree('/'.join(x[0].split('/')[:-1])) 
            
def remove_zeros():           
for x in data:
    for y in x:
        if os.path.getsize(y)==0:
            os.remove(y)
            
def balance_data():
    path='/media/data2/rjsdn/AIP/data/pprm/'
    data = [sorted(glob(os.path.join(x[0],'*.jpg'))) for x in os.walk(path)]
    data = list(filter(None, data))

    cnt=0
    for x in data:
        cnt+=len(x)

    d={}
    for x in range(30):
        d[str(x)]=0

    for x in data:
        i = x[0].split('/')[-3].split('-')[0]
    #     print(x[0].split('/')[-3].split('-')[0])
        d[i]+=1

    m=1e9
    for x in d:
        if d[x]!=0:
            m=min(m,d[x])

    for x in d:
        d[x]-=m

        random.shuffle(data)   


    for x in data:
        i = x[0].split('/')[-3].split('-')[0]
        if d[i]<=0: continue
        d[i]-=1
        shutil.rmtree('/'.join(x[0].split('/')[:-1]))    
    
         