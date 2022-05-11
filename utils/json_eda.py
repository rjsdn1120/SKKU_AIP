import json
from PIL import Image
import os
import torch
from torch import nn
from glob import glob


def run():
    json_path = '/media/data2/rjsdn/AIP/data/annotation/2D/'
    jsons =[y for x in os.walk(json_path) for y in glob(os.path.join(x[0],'*.json'))]

    d ={}
    for x in range(100):
        d[str(x)]=0

    for x in jsons:
        with open(x) as json_file:
            json_data = json.load(json_file)
            json_annot=json_data["annotations"]
            json_image=json_data["images"]
            img_path=[]
            bbox=[]

            for j in json_annot:
                d[str(j['person_no'])]+=1             

if __name__=='__main__':
    run()
            
